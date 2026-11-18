import json
import math
from pathlib import Path

from hazard_perception.camera_geometry import (
    body_point_to_pixel,
    CameraGeometryError,
    CameraRay,
    elevation,
    FRAME_CONVENTION_BODY,
    FRAME_CONVENTION_OPTICAL,
    horizontal_bearing,
    IDENTITY,
    intrinsics_from_calibration,
    OPTICAL_TO_BODY,
    optical_to_body,
    PinholeIntrinsics,
    pixel_to_body_ray,
    pixel_to_camera_direction,
    pixel_to_optical_ray,
    pixel_to_ray,
    RawCameraCalibration,
    rotation_from_quaternion,
    validated_pose,
    WAREHOUSE_THERMAL_PRINCIPAL_POINT_SHIFT_PX,
)
import pytest

# R1 experimental validation data (simulator ground truth for offline evaluation only).
RECORD_PATH = Path(__file__).parent / 'data' / 'r1_thermal_geometry' / 'validation_record.json'
RECORD = json.loads(RECORD_PATH.read_text())
RAW = RECORD['raw_results']
INFO = RAW['camera_info']
CALIBRATION = RawCameraCalibration(
    frame_id=INFO['frame_id'], width=INFO['width'], height=INFO['height'], k=INFO['k'],
    distortion_model=INFO['distortion_model'], d=INFO['d'])
SHIFT = WAREHOUSE_THERMAL_PRINCIPAL_POINT_SHIFT_PX


def _intrinsics(shift=SHIFT, convention=FRAME_CONVENTION_BODY, calibration=CALIBRATION):
    return intrinsics_from_calibration(calibration, principal_point_shift_px=shift,
                                       frame_convention=convention)


K = _intrinsics()
SPHERES = [c for c in RAW['cases'] if c['target'] == 'r1_target']
CONES = [c for c in RAW['cases'] if c['target'] == 'fire_cone']
ZERO = (0.0, 0.0, 0.0)


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def _camera_pose(case):
    """World pose of thermal_link from Gazebo's robot and link poses (offline evaluation)."""
    (robot_p, robot_q), (link_p, link_q) = case['robot_world'], case['thermal_link_in_model']
    robot_r = rotation_from_quaternion(*robot_q)
    position = tuple(p + o for p, o in zip(robot_p, _apply(robot_r, link_p)))
    return position, _matmul(robot_r, rotation_from_quaternion(*link_q))


def _target_in_camera(case, on_floor_axis=False):
    position, rotation = _camera_pose(case)
    target = list(case['target_world'])
    if on_floor_axis:  # fire cone: compare with its axis at camera height
        target[2] = position[2]
    return _apply(_transpose(rotation), [t - p for t, p in zip(target, position)])


# --- Raw versus corrected calibration ----------------------------------------------

def test_recorded_calibration_is_what_gazebo_published():
    assert (CALIBRATION.width, CALIBRATION.height) == (320, 240)
    assert CALIBRATION.k[0] == CALIBRATION.k[4] == pytest.approx(160 / math.tan(1.047 / 2))
    assert (CALIBRATION.k[2], CALIBRATION.k[5]) == (160.0, 120.0)  # raw, pixel-edge
    assert CALIBRATION.distortion_model == 'plumb_bob' and not any(CALIBRATION.d)


def test_correction_is_applied_once_and_recorded():
    assert (K.cx, K.cy) == (159.5, 119.5)
    assert K.principal_point_shift_px == -0.5
    assert (CALIBRATION.k[2], CALIBRATION.k[5]) == (160.0, 120.0)  # raw data untouched
    uncorrected = _intrinsics(shift=0.0)
    assert (uncorrected.cx, uncorrected.cy, uncorrected.principal_point_shift_px) == \
        (160.0, 120.0, 0.0)


def test_corrected_intrinsics_cannot_be_converted_again():
    with pytest.raises(CameraGeometryError, match='must not be converted again'):
        intrinsics_from_calibration(K, principal_point_shift_px=SHIFT,
                                    frame_convention=FRAME_CONVENTION_BODY)


@pytest.mark.parametrize('kwargs', [
    {'principal_point_shift_px': float('nan'), 'frame_convention': FRAME_CONVENTION_BODY},
    {'principal_point_shift_px': True, 'frame_convention': FRAME_CONVENTION_BODY},
    {'principal_point_shift_px': -1.5, 'frame_convention': FRAME_CONVENTION_BODY},
    {'principal_point_shift_px': '-0.5', 'frame_convention': FRAME_CONVENTION_BODY},
    {'principal_point_shift_px': SHIFT, 'frame_convention': 'thermal_link'},
    {'principal_point_shift_px': SHIFT, 'frame_convention': None},
])
def test_ambiguous_conversion_configuration_is_rejected(kwargs):
    with pytest.raises(CameraGeometryError):
        intrinsics_from_calibration(CALIBRATION, **kwargs)


def test_conversion_requires_explicit_shift_and_convention():
    with pytest.raises(TypeError):
        intrinsics_from_calibration(CALIBRATION)


def _raw(**changes):
    fields = {'frame_id': 'thermal_link', 'width': 320, 'height': 240, 'k': list(INFO['k']),
              'distortion_model': 'plumb_bob', 'd': [0.0] * 5}
    fields.update(changes)
    return fields


@pytest.mark.parametrize('changes', [
    {'frame_id': ''}, {'frame_id': None},
    {'width': 0}, {'height': -240}, {'width': 320.0}, {'height': True},
    {'k': INFO['k'][:8]}, {'k': [float('inf')] + INFO['k'][1:]},
    {'d': [0.0, float('nan'), 0.0, 0.0, 0.0]}, {'d': 'none'}, {'distortion_model': None},
])
def test_malformed_published_calibration_is_rejected(changes):
    with pytest.raises(CameraGeometryError):
        RawCameraCalibration(**_raw(**changes))


@pytest.mark.parametrize('k', [
    [277.0, 1.0, 160.0, 0.0, 277.0, 120.0, 0.0, 0.0, 1.0],  # skew
    [277.0, 0.0, 160.0, 0.0, 277.0, 120.0, 0.0, 0.0, 2.0],  # not normalised
    [0.0, 0.0, 160.0, 0.0, 277.0, 120.0, 0.0, 0.0, 1.0],  # fx = 0
    [-277.0, 0.0, 160.0, 0.0, 277.0, 120.0, 0.0, 0.0, 1.0],  # fx < 0
    [277.0, 0.0, 400.0, 0.0, 277.0, 120.0, 0.0, 0.0, 1.0],  # principal point outside image
])
def test_invalid_camera_matrices_are_rejected(k):
    with pytest.raises(CameraGeometryError):
        _intrinsics(calibration=RawCameraCalibration(**_raw(k=k)))


def test_distorted_calibration_is_rejected_not_ignored():
    distorted = RawCameraCalibration(**_raw(d=[-0.1, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(CameraGeometryError, match='distortion'):
        _intrinsics(calibration=distorted)


@pytest.mark.parametrize('name, value', [
    ('fx', float('nan')), ('fy', 0.0), ('fx', -1.0), ('cx', float('inf')), ('cy', 240.0),
    ('width', 0), ('frame_id', ''), ('frame_convention', 'body'),
    ('principal_point_shift_px', float('nan')),
])
def test_invalid_intrinsics_are_rejected(name, value):
    fields = {'fx': 277.0, 'fy': 277.0, 'cx': 159.5, 'cy': 119.5, 'width': 320, 'height': 240,
              'frame_id': 'thermal_link', 'frame_convention': FRAME_CONVENTION_BODY,
              'principal_point_shift_px': 0.0}
    fields[name] = value
    with pytest.raises(CameraGeometryError):
        PinholeIntrinsics(**fields)


# --- R1 experimental results remain valid -------------------------------------------

def test_r1_record_is_complete():
    assert len(SPHERES) == 16 and len(CONES) == 3
    assert all(set(c['region_counts']) == {1} for c in SPHERES + CONES)
    assert 'never use it as an input' in RECORD['notice'].lower()


@pytest.mark.parametrize('case', SPHERES, ids=[c['label'] for c in SPHERES])
def test_r1_sphere_bearing_and_elevation(case):
    truth = _target_in_camera(case)
    ray = pixel_to_ray(case['u'], case['v'], K, frame_id='thermal_link',
                       translation=ZERO, rotation=IDENTITY)
    assert horizontal_bearing(ray.direction) == pytest.approx(horizontal_bearing(truth), abs=7e-4)
    assert elevation(ray.direction) == pytest.approx(elevation(truth), abs=4e-4)
    expected = body_point_to_pixel(truth, K)
    assert abs(case['u'] - expected[0]) < 0.25 and abs(case['v'] - expected[1]) < 0.25


@pytest.mark.parametrize('case', SPHERES, ids=[c['label'] for c in SPHERES])
def test_r1_sphere_world_ray_passes_through_target(case):
    # Non-identity rotations (robot yaw 180 and 149 degrees) and non-zero
    # camera offsets, from the recorded Gazebo poses.
    position, rotation = _camera_pose(case)
    ray = pixel_to_ray(case['u'], case['v'], K, frame_id='world',
                       translation=position, rotation=rotation)
    assert ray.origin == pytest.approx(position)
    offset = [t - o for t, o in zip(case['target_world'], ray.origin)]
    along = sum(a * b for a, b in zip(offset, ray.direction))
    closest = [o + along * d for o, d in zip(ray.origin, ray.direction)]
    miss = math.dist(case['target_world'], closest)
    assert along > 0
    assert miss < 0.003  # metres, at 1.2 - 5 m range


def test_uncorrected_calibration_has_systematic_bias():
    uncorrected = _intrinsics(shift=0.0)
    errors = [horizontal_bearing(pixel_to_body_ray(c['u'], c['v'], uncorrected))
              - horizontal_bearing(_target_in_camera(c)) for c in SPHERES]
    assert min(errors) > 0.001  # every case biased the same way (+0.0017 rad mean)


def test_duplicated_correction_is_detectable_against_r1():
    pre_shifted = RawCameraCalibration(**_raw(k=[
        INFO['k'][0], 0.0, INFO['k'][2] + SHIFT, 0.0, INFO['k'][4], INFO['k'][5] + SHIFT,
        0.0, 0.0, 1.0]))
    double = _intrinsics(calibration=pre_shifted)
    assert (double.cx, double.cy) == (159.0, 119.0)
    errors = [horizontal_bearing(pixel_to_body_ray(c['u'], c['v'], double))
              - horizontal_bearing(_target_in_camera(c)) for c in SPHERES]
    # Every case biased negatively and outside the corrected +-0.0007 rad band.
    assert max(errors) < -0.0009
    assert sum(errors) / len(errors) < -0.0015


@pytest.mark.parametrize('case', CONES, ids=[c['label'] for c in CONES])
def test_r1_fire_cone_bearing_shape_bias(case):
    truth = _target_in_camera(case, on_floor_axis=True)
    ray = pixel_to_body_ray(case['u'], case['v'], K)
    assert horizontal_bearing(ray) == pytest.approx(horizontal_bearing(truth), abs=2.5e-3)


# --- Conventions -----------------------------------------------------------------------

def test_optical_to_body_is_the_rep_103_rotation():
    assert _matmul(OPTICAL_TO_BODY, _transpose(OPTICAL_TO_BODY)) == IDENTITY
    validated_pose('base_link', ZERO, OPTICAL_TO_BODY)  # proper rotation, det +1
    roll, yaw = -math.pi / 2, -math.pi / 2  # body -> optical static transform
    rz = ((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1))
    rx = ((1, 0, 0), (0, math.cos(roll), -math.sin(roll)), (0, math.sin(roll), math.cos(roll)))
    for row, expected in zip(_matmul(rz, rx), OPTICAL_TO_BODY):
        assert row == pytest.approx(expected, abs=1e-12)


def test_principal_point_maps_forward_left_is_positive_up_is_positive():
    assert pixel_to_body_ray(K.cx, K.cy, K) == pytest.approx((1.0, 0.0, 0.0))
    assert horizontal_bearing(pixel_to_body_ray(K.cx - 50, K.cy, K)) > 0
    assert horizontal_bearing(pixel_to_body_ray(K.cx + 50, K.cy, K)) < 0
    assert elevation(pixel_to_body_ray(K.cx, K.cy - 50, K)) > 0
    assert elevation(pixel_to_body_ray(K.cx, K.cy + 50, K)) < 0


def test_optical_convention_calibration_gives_the_same_world_ray():
    optical = _intrinsics(convention=FRAME_CONVENTION_OPTICAL)
    assert pixel_to_camera_direction(10.0, 20.0, optical) == pixel_to_optical_ray(10.0, 20.0, K)
    a = pixel_to_ray(10.0, 20.0, K, frame_id='base_link', translation=ZERO, rotation=IDENTITY)
    b = pixel_to_ray(10.0, 20.0, optical, frame_id='base_link', translation=ZERO,
                     rotation=OPTICAL_TO_BODY)  # pose of an optical frame in the body frame
    assert b.direction == pytest.approx(a.direction)


@pytest.mark.parametrize('point', [(2.5, 0.0, 0.0), (2.0, 0.8, -0.1), (5.0, -1.4, 0.6)])
def test_projection_and_back_projection_agree(point):
    u, v = body_point_to_pixel(point, K)
    ray = pixel_to_body_ray(u, v, K)
    assert horizontal_bearing(ray) == pytest.approx(math.atan2(point[1], point[0]))
    assert elevation(ray) == pytest.approx(math.atan2(point[2], math.hypot(point[0], point[1])))


def test_points_behind_the_camera_do_not_project():
    assert body_point_to_pixel((0.0, 0.0, 1.0), K) is None
    assert body_point_to_pixel((-1.0, 0.0, 0.0), K) is None


# --- Public ray: normalisation, rotation and translation --------------------------------

def _rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


@pytest.mark.parametrize('rotation', [
    IDENTITY,
    _rpy(0.0, 0.0, math.pi / 2),
    _rpy(0.3, -0.2, 2.6),
    _matmul(_rpy(0.0, 0.0, 2.6), _rpy(0.1, 0.4, -0.7)),  # compound rotation
])
@pytest.mark.parametrize('translation', [ZERO, (0.185, -0.04, 0.13), (-3.1, 2.0, 0.5)])
def test_ray_rotates_direction_and_translates_origin(rotation, translation):
    u, v = 73.6, 75.6
    ray = pixel_to_ray(u, v, K, frame_id='parent', translation=translation, rotation=rotation)
    body = pixel_to_body_ray(u, v, K)
    norm = math.sqrt(sum(c * c for c in body))
    assert ray.frame_id == 'parent'
    assert ray.origin == pytest.approx(translation)
    assert math.sqrt(sum(c * c for c in ray.direction)) == pytest.approx(1.0, abs=1e-12)
    assert ray.direction == pytest.approx(_apply(rotation, [c / norm for c in body]))
    # A point along the ray, mapped back into the camera frame, projects to (u, v).
    point = [o + 3.0 * d for o, d in zip(ray.origin, ray.direction)]
    in_camera = _apply(_transpose(rotation), [p - t for p, t in zip(point, translation)])
    assert body_point_to_pixel(in_camera, K) == pytest.approx((u, v))


def test_yaw_90_mount_turns_forward_into_left():
    ray = pixel_to_ray(K.cx, K.cy, K, frame_id='base_link', translation=ZERO,
                       rotation=_rpy(0.0, 0.0, math.pi / 2))
    assert ray.direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_parent_bearing_differs_from_ray_bearing_at_finite_range():
    ray = pixel_to_ray(73.5519, 119.5, K, frame_id='base_link',
                       translation=(0.185, -0.04, 0.13), rotation=IDENTITY)
    assert horizontal_bearing(ray.direction) == pytest.approx(0.30, abs=1e-3)
    target = [o + 1.2 * d for o, d in zip(ray.origin, ray.direction)]
    assert math.atan2(target[1], target[0]) == pytest.approx(0.2321, abs=1e-3)  # R1


@pytest.mark.parametrize('q, expected', [
    ((0.0, 0.0, 0.0, 1.0), IDENTITY),
    ((0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)), _rpy(0.0, 0.0, math.pi / 2)),
    ((math.sin(0.15), 0.0, 0.0, math.cos(0.15)), _rpy(0.3, 0.0, 0.0)),
])
def test_rotation_from_quaternion(q, expected):
    for row, expected_row in zip(rotation_from_quaternion(*q), expected):
        assert row == pytest.approx(expected_row, abs=1e-12)


@pytest.mark.parametrize('q', [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0),
                               (float('nan'), 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, True)])
def test_invalid_quaternions_are_rejected(q):
    with pytest.raises(CameraGeometryError):
        rotation_from_quaternion(*q)


# --- Invalid inputs ------------------------------------------------------------------

@pytest.mark.parametrize('u, v', [
    (float('nan'), 100.0), (100.0, float('inf')), (None, 1.0), ('1', 1.0), (True, 1.0),
    (-0.51, 100.0), (319.51, 100.0), (100.0, -0.51), (100.0, 239.51),
])
def test_invalid_or_out_of_image_pixels_are_rejected(u, v):
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(u, v, K, frame_id='thermal_link', translation=ZERO, rotation=IDENTITY)
    with pytest.raises(CameraGeometryError):
        pixel_to_body_ray(u, v, K)


def test_pixel_bounds_are_inclusive_at_the_outer_pixel_edges():
    for u, v in ((-0.5, -0.5), (319.5, 239.5)):
        pixel_to_ray(u, v, K, frame_id='thermal_link', translation=ZERO, rotation=IDENTITY)


@pytest.mark.parametrize('translation', [
    (0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, float('nan'), 0.0), 'abc', None, 1.0,
    (0.0, True, 0.0),
])
def test_malformed_translations_are_rejected(translation):
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(100.0, 100.0, K, frame_id='base_link', translation=translation,
                     rotation=IDENTITY)


@pytest.mark.parametrize('rotation', [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # 2 x 3
    ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),  # 3 x 2
    ((1.0, 0.0, 0.0), (0.0, float('nan'), 0.0), (0.0, 0.0, 1.0)),
    ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # scaled
    ((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # sheared
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),  # reflection
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # reflection (axis swap)
    'IDENTITY', None,
])
def test_invalid_rotations_are_rejected(rotation):
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(100.0, 100.0, K, frame_id='base_link', translation=ZERO, rotation=rotation)


@pytest.mark.parametrize('frame_id', ['', None, 3])
def test_invalid_ray_frames_are_rejected(frame_id):
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(100.0, 100.0, K, frame_id=frame_id, translation=ZERO, rotation=IDENTITY)


def test_pose_must_be_given_explicitly():
    with pytest.raises(TypeError):
        pixel_to_ray(100.0, 100.0, K, frame_id='base_link')


@pytest.mark.parametrize('direction', [
    (0.0, 0.0, 0.0), (float('nan'), 0.0, 0.0), (1.0, 0.0), (0.0, 0.0, 2.0), (0.6, 0.6, 0.0),
])
def test_camera_ray_requires_a_finite_unit_direction(direction):
    with pytest.raises(CameraGeometryError):
        CameraRay(frame_id='base_link', origin=ZERO, direction=direction)


@pytest.mark.parametrize('direction', [(0.0, 0.0, 0.0), (float('inf'), 0.0, 0.0), (1.0,)])
def test_degenerate_directions_are_rejected_by_angle_helpers(direction):
    with pytest.raises(CameraGeometryError):
        horizontal_bearing(direction)
    with pytest.raises(CameraGeometryError):
        elevation(direction)


def test_bearing_of_a_vertical_direction_is_undefined():
    with pytest.raises(CameraGeometryError):
        horizontal_bearing((0.0, 0.0, 1.0))
    assert elevation((0.0, 0.0, 1.0)) == pytest.approx(math.pi / 2)


def test_non_intrinsics_and_malformed_vectors_are_rejected():
    with pytest.raises(CameraGeometryError):
        pixel_to_optical_ray(1.0, 1.0, {'fx': 277.0})
    with pytest.raises(CameraGeometryError):
        optical_to_body((1.0, 2.0))
    with pytest.raises(CameraGeometryError):
        body_point_to_pixel((1.0, float('nan'), 0.0), K)


def test_numpy_scalars_are_accepted_as_numbers():
    np = pytest.importorskip('numpy')
    ray = pixel_to_ray(np.float64(73.6), np.float32(119.5), K, frame_id='thermal_link',
                       translation=np.zeros(3), rotation=np.eye(3))
    assert all(type(c) is float for c in ray.origin + ray.direction)


# --- R2.1 regressions (Codex counterexamples) ---------------------------------

@pytest.mark.parametrize('model, d', [
    ('equidistant', [0.0] * 4),  # R21-1: other model, zero coefficients
    ('mystery_model', []),  # R21-1: unknown model, no coefficients
    ('plumb_bob', [0.0]),  # R21-1: wrong coefficient count
    ('plumb_bob', [0.0] * 8),
    ('rational_polynomial', [0.0] * 8),
    ('', []),
])
def test_r21_1_only_zero_plumb_bob_calibration_is_supported(model, d):
    calibration = RawCameraCalibration(**_raw(distortion_model=model, d=d))
    with pytest.raises(CameraGeometryError, match='unsupported distortion'):
        _intrinsics(calibration=calibration)


def test_r21_1_ros_container_types_remain_accepted():
    array = pytest.importorskip('array')
    np = pytest.importorskip('numpy')
    # sensor_msgs/CameraInfo uses a 1-d ndarray for k and array.array for d.
    calibration = RawCameraCalibration(
        **_raw(k=np.array(INFO['k']), d=array.array('d', [0.0] * 5)))
    assert _intrinsics(calibration=calibration) == K


@pytest.mark.parametrize('call', [
    lambda: pixel_to_ray(10**400, 10.0, K, frame_id='b', translation=ZERO, rotation=IDENTITY),
    lambda: pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=(10**400, 0, 0),
                         rotation=IDENTITY),
    lambda: rotation_from_quaternion(0, 0, 0, 10**400),
    lambda: RawCameraCalibration(**_raw(width=320, k=[10**400] + INFO['k'][1:])),
])
def test_r21_4_huge_integers_raise_geometry_error_not_overflow(call):
    with pytest.raises(CameraGeometryError, match='too large'):
        call()


@pytest.mark.parametrize('translation', [
    {0: 1.0, 1: 2.0, 2: 3.0},  # R21-4: was silently read as its keys (0, 1, 2)
    {1.0, 2.0, 3.0},
    (x for x in (1.0, 2.0, 3.0)),
    b'abc',
])
def test_r21_4_mappings_and_unsupported_containers_are_rejected(translation):
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=translation, rotation=IDENTITY)


def test_r21_4_array_dimensions_are_checked():
    np = pytest.importorskip('numpy')
    with pytest.raises(CameraGeometryError, match='0-d'):
        pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=np.array(1.0), rotation=IDENTITY)
    with pytest.raises(CameraGeometryError, match='2-d'):
        pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=np.zeros((3, 1)), rotation=IDENTITY)
    with pytest.raises(CameraGeometryError, match='2-d'):
        pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=ZERO, rotation=np.eye(3).ravel())
    with pytest.raises(CameraGeometryError):
        pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=ZERO, rotation={0: IDENTITY[0]})
    ray = pixel_to_ray(10.0, 10.0, K, frame_id='b', translation=np.zeros(3), rotation=np.eye(3))
    assert ray.origin == ZERO


def test_low_level_outputs_are_always_finite():
    tiny = PinholeIntrinsics(fx=1e-310, fy=1e-310, cx=159.5, cy=119.5, width=320, height=240,
                             frame_id='thermal_link', frame_convention=FRAME_CONVENTION_BODY,
                             principal_point_shift_px=0.0)
    with pytest.raises(CameraGeometryError, match='not finite'):
        pixel_to_optical_ray(0.0, 0.0, tiny)
    with pytest.raises(CameraGeometryError, match='not finite'):
        body_point_to_pixel((1e-320, 1.0, 0.0), K)
    extreme = PinholeIntrinsics(fx=1e-300, fy=1e-300, cx=159.5, cy=119.5, width=320,
                                height=240, frame_id='thermal_link',
                                frame_convention=FRAME_CONVENTION_BODY,
                                principal_point_shift_px=0.0)
    with pytest.raises(CameraGeometryError):  # direction norm would overflow
        pixel_to_ray(0.0, 0.0, extreme, frame_id='b', translation=ZERO, rotation=IDENTITY)
    big = pixel_to_ray(K.cx, K.cy, K, frame_id='b', translation=(1e300, 0.0, 0.0),
                       rotation=IDENTITY)
    assert big.direction == pytest.approx((1.0, 0.0, 0.0))
