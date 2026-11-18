from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers

# Frame conventions (ROS REP 103):
# - Body convention (base_link, and Gazebo sensor/link frames such as
#   thermal_link): x forward, y left, z up.
# - Optical convention: x right, y down, z forward (the image plane).
# Pixel coordinates (u, v): u to the right, v downward, pixel centres at
# integer coordinates (the convention of ThermalRegion.centroid_xy). Pixel i
# covers [i - 0.5, i + 0.5], so valid coordinates lie in
# [-0.5, width - 0.5] x [-0.5, height - 0.5].
#
# Failure behaviour: every invalid input raises CameraGeometryError (a
# ValueError). No function returns a malformed ray, and every returned
# number is finite (a computation that would overflow raises instead).
# Vectors and matrices may be sequences (tuple, list, array.array) or NumPy
# arrays of the right dimension; strings, mappings, sets and 0-d arrays are
# rejected.
#
# Only distortion-free calibration is supported: distortion model 'plumb_bob'
# with exactly five zero coefficients (the R1-validated configuration).
SUPPORTED_DISTORTION_MODEL = 'plumb_bob'
SUPPORTED_DISTORTION_COEFFICIENTS = 5

FRAME_CONVENTION_BODY = 'rep103_body'
FRAME_CONVENTION_OPTICAL = 'optical'
FRAME_CONVENTIONS = (FRAME_CONVENTION_BODY, FRAME_CONVENTION_OPTICAL)

# Principal-point correction validated in R1 for THIS configuration only: the
# warehouse robot's 320 x 240 thermal camera in gz-sim 8 (Harmonic) via
# ros_gz_bridge on ROS 2 Jazzy, whose published K puts the principal point at
# (width / 2, height / 2). Measured on-axis targets appear at (159.5, 119.5)
# in pixel-centre coordinates. Re-validate before using it for any other
# camera, resolution or simulator version. Evidence:
# test/data/r1_thermal_geometry/validation_record.json
WAREHOUSE_THERMAL_PRINCIPAL_POINT_SHIFT_PX = -0.5

MAX_PRINCIPAL_POINT_SHIFT_PX = 1.0  # larger corrections are rejected as implausible
ROTATION_TOLERANCE = 1e-9
QUATERNION_NORM_TOLERANCE = 1e-6

# body = OPTICAL_TO_BODY @ optical
OPTICAL_TO_BODY = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class CameraGeometryError(ValueError):
    """An input to the camera geometry functions is invalid."""


# --- Validation helpers ---------------------------------------------------------

def _real(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CameraGeometryError(f'{name} must be a real number, not {type(value).__name__}')
    try:
        value = float(value)
    except OverflowError:  # an integer too large for a float
        raise CameraGeometryError(f'{name} is too large to represent as a float') from None
    if not math.isfinite(value):
        raise CameraGeometryError(f'{name} must be finite, not {value}')
    return value


def _finite_output(name, values):
    if not all(math.isfinite(v) for v in values):
        raise CameraGeometryError(f'{name} is not finite ({values}); inputs are out of range')
    return values


def _items(name, value, ndim=1):
    # Elements of a sequence, or of a NumPy-style array with exactly ndim dimensions.
    if isinstance(value, (str, bytes, Mapping)):
        raise CameraGeometryError(f'{name} must be a sequence, not {type(value).__name__}')
    array_ndim = getattr(value, 'ndim', None)
    if array_ndim is not None:
        if array_ndim != ndim:
            raise CameraGeometryError(f'{name} must be a {ndim}-d array, not {array_ndim}-d')
    elif not isinstance(value, Sequence):
        raise CameraGeometryError(f'{name} must be a sequence, not {type(value).__name__}')
    return tuple(value)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise CameraGeometryError(f'{name} must be a positive integer, not {value!r}')
    return int(value)


def _frame_id(name, value):
    if not isinstance(value, str) or not value:
        raise CameraGeometryError(f'{name} must be a non-empty string, not {value!r}')
    return value


def _sequence(name, value, length):
    items = _items(name, value)
    if length is not None and len(items) != length:
        raise CameraGeometryError(f'{name} must have {length} elements, not {len(items)}')
    return tuple(_real(f'{name}[{i}]', v) for i, v in enumerate(items))


def validated_reals(name: str, value, length: int | None):
    """Return value as a tuple of finite floats (of the given length, if not None)."""
    return _sequence(name, value, length)


def _vector3(name, value):
    return _sequence(name, value, 3)


def _nonzero_vector3(name, value):
    vector = _vector3(name, value)
    if not any(vector):
        raise CameraGeometryError(f'{name} must not be the zero vector')
    return vector


def _rotation(name, value):
    outer = _items(name, value, ndim=2)
    if len(outer) != 3:
        raise CameraGeometryError(f'{name} must be a 3x3 matrix, not {len(outer)} rows')
    rows = tuple(_vector3(f'{name}[{i}]', row) for i, row in enumerate(outer))
    for i in range(3):
        for j in range(3):
            dot = sum(rows[i][k] * rows[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > ROTATION_TOLERANCE:
                raise CameraGeometryError(f'{name} is not orthonormal (row {i} . row {j} = {dot})')
    det = (rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
           - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
           + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]))
    if abs(det - 1.0) > ROTATION_TOLERANCE:
        raise CameraGeometryError(f'{name} is not a proper rotation (determinant {det})')
    return rows


def _rotate(rotation, vector):
    return _finite_output('rotated vector', tuple(
        sum(rotation[row][col] * vector[col] for col in range(3)) for row in range(3)))


# --- Calibration ------------------------------------------------------------------

@dataclass(frozen=True)
class RawCameraCalibration:
    """Camera calibration exactly as published (e.g. sensor_msgs/CameraInfo), uncorrected."""

    frame_id: str
    width: int
    height: int
    k: tuple
    distortion_model: str
    d: tuple

    def __post_init__(self) -> None:
        _frame_id('frame_id', self.frame_id)
        object.__setattr__(self, 'width', _positive_int('width', self.width))
        object.__setattr__(self, 'height', _positive_int('height', self.height))
        object.__setattr__(self, 'k', _sequence('k', self.k, 9))
        if not isinstance(self.distortion_model, str):
            raise CameraGeometryError('distortion_model must be a string')
        object.__setattr__(self, 'd', _sequence('d', self.d, None))


@dataclass(frozen=True)
class PinholeIntrinsics:
    """
    Corrected, distortion-free intrinsics in pixel-centre coordinates.

    frame_id is the camera frame the calibration belongs to; frame_convention
    says whether that frame is REP 103 body (x forward) or optical (z
    forward). principal_point_shift_px records the correction already applied
    to the published principal point; intrinsics_from_calibration is the only
    function that applies one.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_id: str
    frame_convention: str
    principal_point_shift_px: float

    def __post_init__(self) -> None:
        for name in ('fx', 'fy', 'cx', 'cy', 'principal_point_shift_px'):
            object.__setattr__(self, name, _real(name, getattr(self, name)))
        if self.fx <= 0 or self.fy <= 0:
            raise CameraGeometryError(f'fx and fy must be positive, not {self.fx}, {self.fy}')
        object.__setattr__(self, 'width', _positive_int('width', self.width))
        object.__setattr__(self, 'height', _positive_int('height', self.height))
        _frame_id('frame_id', self.frame_id)
        if self.frame_convention not in FRAME_CONVENTIONS:
            raise CameraGeometryError(
                f'frame_convention must be one of {FRAME_CONVENTIONS}, '
                f'not {self.frame_convention!r}')
        if not (-0.5 <= self.cx <= self.width - 0.5 and -0.5 <= self.cy <= self.height - 0.5):
            raise CameraGeometryError(
                f'principal point ({self.cx}, {self.cy}) lies outside the '
                f'{self.width} x {self.height} image')


def intrinsics_from_calibration(calibration: RawCameraCalibration, *,
                                principal_point_shift_px: float,
                                frame_convention: str) -> PinholeIntrinsics:
    """
    Convert published calibration into corrected internal intrinsics.

    This is the single calibration conversion boundary. The shift is added to
    the published cx and cy and must be given explicitly (0.0 for none). Only
    an uncorrected RawCameraCalibration is accepted, so the correction cannot
    be applied twice. Calibration with non-zero distortion is rejected.
    """
    if not isinstance(calibration, RawCameraCalibration):
        raise CameraGeometryError(
            f'expected RawCameraCalibration, not {type(calibration).__name__}; '
            'corrected intrinsics must not be converted again')
    shift = _real('principal_point_shift_px', principal_point_shift_px)
    if abs(shift) > MAX_PRINCIPAL_POINT_SHIFT_PX:
        raise CameraGeometryError(f'principal_point_shift_px {shift} exceeds '
                                  f'{MAX_PRINCIPAL_POINT_SHIFT_PX} px')
    k = calibration.k
    if k[1] != 0 or k[3] != 0 or k[6] != 0 or k[7] != 0 or k[8] != 1:
        raise CameraGeometryError(f'K is not a skew-free pinhole matrix: {list(k)}')
    if (calibration.distortion_model != SUPPORTED_DISTORTION_MODEL
            or len(calibration.d) != SUPPORTED_DISTORTION_COEFFICIENTS or any(calibration.d)):
        raise CameraGeometryError(
            f'unsupported distortion ({calibration.distortion_model!r}: {list(calibration.d)}); '
            f'only {SUPPORTED_DISTORTION_MODEL!r} with {SUPPORTED_DISTORTION_COEFFICIENTS} '
            'zero coefficients is supported')
    return PinholeIntrinsics(
        fx=k[0], fy=k[4], cx=k[2] + shift, cy=k[5] + shift,
        width=calibration.width, height=calibration.height,
        frame_id=calibration.frame_id, frame_convention=frame_convention,
        principal_point_shift_px=shift,
    )


def _pixel(u, v, intrinsics):
    if not isinstance(intrinsics, PinholeIntrinsics):
        raise CameraGeometryError(f'expected PinholeIntrinsics, not {type(intrinsics).__name__}')
    u, v = _real('u', u), _real('v', v)
    if not (-0.5 <= u <= intrinsics.width - 0.5 and -0.5 <= v <= intrinsics.height - 0.5):
        raise CameraGeometryError(
            f'pixel ({u}, {v}) lies outside the {intrinsics.width} x {intrinsics.height} image')
    return u, v


# --- Low-level, unnormalised directions --------------------------------------------

def pixel_to_optical_ray(u: float, v: float, intrinsics: PinholeIntrinsics):
    """Return the unnormalised optical-convention direction [(u-cx)/fx, (v-cy)/fy, 1]."""
    u, v = _pixel(u, v, intrinsics)
    return _finite_output('optical ray', (
        (u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0))


def optical_to_body(vector):
    """Rotate a vector from optical convention into body convention (unchanged length)."""
    return _rotate(OPTICAL_TO_BODY, _vector3('vector', vector))


def pixel_to_body_ray(u: float, v: float, intrinsics: PinholeIntrinsics):
    """Return the unnormalised body-convention direction through pixel (u, v)."""
    return optical_to_body(pixel_to_optical_ray(u, v, intrinsics))


def pixel_to_camera_direction(u: float, v: float, intrinsics: PinholeIntrinsics):
    """Return the unnormalised direction through (u, v) in intrinsics.frame_id."""
    optical = pixel_to_optical_ray(u, v, intrinsics)
    if intrinsics.frame_convention == FRAME_CONVENTION_OPTICAL:
        return optical
    return optical_to_body(optical)


# --- Public ray boundary ----------------------------------------------------------------

@dataclass(frozen=True)
class CameraRay:
    """
    A ray with an explicit frame: origin and unit-length direction in frame_id.

    The origin is the camera position. At finite range, the bearing of a
    target seen from frame_id's origin differs from the ray's own bearing.
    """

    frame_id: str
    origin: tuple
    direction: tuple

    def __post_init__(self) -> None:
        _frame_id('frame_id', self.frame_id)
        object.__setattr__(self, 'origin', _vector3('origin', self.origin))
        direction = _nonzero_vector3('direction', self.direction)
        norm = math.sqrt(sum(c * c for c in direction))
        if abs(norm - 1.0) > 1e-9:
            raise CameraGeometryError(f'direction must have unit length, not {norm}')
        object.__setattr__(self, 'direction', direction)


def validated_pose(frame_id: str, translation, rotation):
    """Return (frame_id, translation, rotation) checked and as float tuples."""
    return _frame_id('frame_id', frame_id), _vector3('translation', translation), \
        _rotation('rotation', rotation)


def pixel_to_ray(u: float, v: float, intrinsics: PinholeIntrinsics, *,
                 frame_id: str, translation, rotation) -> CameraRay:
    """
    Return the normalised ray through pixel (u, v), expressed in frame_id.

    translation and rotation are the pose of intrinsics.frame_id in frame_id
    (e.g. the base_link -> thermal_link transform). Both are required: pass
    (0, 0, 0) and IDENTITY explicitly for a ray in the camera frame itself.
    """
    frame_id, origin, rows = validated_pose(frame_id, translation, rotation)
    direction = _rotate(rows, pixel_to_camera_direction(u, v, intrinsics))
    norm = math.sqrt(sum(c * c for c in direction))
    _finite_output('ray direction norm', (norm,))
    return CameraRay(frame_id=frame_id, origin=origin,
                     direction=tuple(c / norm for c in direction))


def rotation_from_quaternion(x: float, y: float, z: float, w: float):
    """Return the rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (_real(n, val) for n, val in zip('xyzw', (x, y, z, w)))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise CameraGeometryError(f'quaternion must have unit norm, not {norm}')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


# --- Angles and projection (body convention) -------------------------------------------

def horizontal_bearing(direction) -> float:
    """Return the azimuth of a body-convention direction; positive means left."""
    x, y, _ = _nonzero_vector3('direction', direction)
    if x == 0 and y == 0:
        raise CameraGeometryError('horizontal bearing is undefined for a vertical direction')
    return math.atan2(y, x)


def elevation(direction) -> float:
    """Return the elevation of a body-convention direction; positive means up."""
    x, y, z = _nonzero_vector3('direction', direction)
    return math.atan2(z, math.hypot(x, y))


def body_point_to_pixel(point, intrinsics: PinholeIntrinsics):
    """
    Project a point given in the camera body frame (evaluation helper).

    Returns None if the point is not in front of the camera. The result may
    lie outside the image; callers check visibility themselves. Raises if the
    projection is not finite (a point extremely close to the image plane).
    """
    if not isinstance(intrinsics, PinholeIntrinsics):
        raise CameraGeometryError(f'expected PinholeIntrinsics, not {type(intrinsics).__name__}')
    x, y, z = _vector3('point', point)
    if x <= 0:
        return None
    return _finite_output('projected pixel', (
        intrinsics.cx - intrinsics.fx * y / x, intrinsics.cy - intrinsics.fy * z / x))
