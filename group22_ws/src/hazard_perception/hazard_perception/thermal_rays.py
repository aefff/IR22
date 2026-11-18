from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numbers

from hazard_perception.camera_geometry import (
    CameraGeometryError,
    CameraRay,
    PinholeIntrinsics,
    pixel_to_ray,
    validated_pose,
    validated_reals,
)
from hazard_perception.thermal_regions import ThermalFrameObservation, ThermalRegion

# Converts image-space thermal evidence into camera rays. A ray is a bearing
# from the camera, nothing more: no range, map position, identity or
# probability. Region indices are only positions within one frame.


def _int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise CameraGeometryError(f'{name} must be an integer >= {minimum}, not {value!r}')
    return int(value)


def _sequence(name, value):
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise CameraGeometryError(f'{name} must be a sequence, not {type(value).__name__}')
    return tuple(value)


@dataclass(frozen=True)
class RegionRay:
    """
    The ray through one thermal region's centroid.

    Construction validates and copies every field (centroid_xy becomes a tuple
    of floats). That the centroid and the ray correspond is guaranteed only
    for records produced by observation_to_rays.
    """

    region_index: int
    centroid_xy: tuple
    touches_image_boundary: bool  # the region, and so its centroid, may be truncated
    ray: CameraRay

    def __post_init__(self) -> None:
        object.__setattr__(self, 'region_index', _int('region_index', self.region_index, 0))
        object.__setattr__(self, 'centroid_xy',
                           validated_reals('centroid_xy', self.centroid_xy, 2))
        if not isinstance(self.touches_image_boundary, bool):
            raise CameraGeometryError('touches_image_boundary must be a bool')
        if not isinstance(self.ray, CameraRay):
            raise CameraGeometryError(f'ray must be a CameraRay, not {type(self.ray).__name__}')


@dataclass(frozen=True)
class ThermalObservationRays:
    """Rays for every region of one thermal frame, with the frame's source metadata."""

    source_stamp_ns: int | None
    source_frame_id: str
    image_width: int
    image_height: int
    rays: tuple  # RegionRay entries, in the observation's region order; empty if none

    def __post_init__(self) -> None:
        if self.source_stamp_ns is not None:
            _int('source_stamp_ns', self.source_stamp_ns, 0)
        if not isinstance(self.source_frame_id, str) or not self.source_frame_id:
            raise CameraGeometryError('source_frame_id must be a non-empty string')
        object.__setattr__(self, 'image_width', _int('image_width', self.image_width, 1))
        object.__setattr__(self, 'image_height', _int('image_height', self.image_height, 1))
        rays = _sequence('rays', self.rays)
        if not all(isinstance(r, RegionRay) for r in rays):
            raise CameraGeometryError('rays must contain only RegionRay records')
        indices = [r.region_index for r in rays]
        if indices != sorted(set(indices)):
            raise CameraGeometryError(f'region indices must be unique and ascending: {indices}')
        if len({r.ray.frame_id for r in rays}) > 1:
            raise CameraGeometryError('all rays must be expressed in the same frame')
        object.__setattr__(self, 'rays', rays)


def _validated_region(index, region, width, height):
    # Checks a region against the extractor's contract (thermal_regions.py):
    # half-open bbox inside the image, centroid inside the bbox's pixels, and
    # touches_image_boundary exactly when the bbox reaches an image edge.
    name = f'regions[{index}]'
    if not isinstance(region, ThermalRegion):
        raise CameraGeometryError(f'{name} must be a ThermalRegion, not {type(region).__name__}')
    bbox = _sequence(f'{name}.bbox_xywh', region.bbox_xywh)
    if len(bbox) != 4:
        raise CameraGeometryError(f'{name}.bbox_xywh must have 4 elements, not {len(bbox)}')
    x, y = _int(f'{name}.bbox x', bbox[0], 0), _int(f'{name}.bbox y', bbox[1], 0)
    w, h = _int(f'{name}.bbox width', bbox[2], 1), _int(f'{name}.bbox height', bbox[3], 1)
    if x + w > width or y + h > height:
        raise CameraGeometryError(
            f'{name}.bbox_xywh {(x, y, w, h)} extends outside the {width} x {height} image')
    cx, cy = validated_reals(f'{name}.centroid_xy', region.centroid_xy, 2)
    if not (x <= cx <= x + w - 1 and y <= cy <= y + h - 1):
        raise CameraGeometryError(
            f'{name}.centroid_xy {(cx, cy)} lies outside its bbox {(x, y, w, h)}')
    touches = region.touches_image_boundary
    if not isinstance(touches, bool):
        raise CameraGeometryError(f'{name}.touches_image_boundary must be a bool')
    if touches != (x == 0 or y == 0 or x + w == width or y + h == height):
        raise CameraGeometryError(
            f'{name}.touches_image_boundary={touches} is inconsistent with bbox {(x, y, w, h)}')
    return (cx, cy), touches


def observation_to_rays(observation: ThermalFrameObservation, intrinsics: PinholeIntrinsics, *,
                        frame_id: str, translation, rotation) -> ThermalObservationRays:
    """
    Return one ray per region, expressed in frame_id (all or nothing).

    translation and rotation are the pose of the camera frame in frame_id at
    the observation's time. The observation must come from the camera that
    intrinsics describes (same frame and image size), and every region must
    satisfy the extractor's contract. If anything is invalid, the whole
    observation is rejected with CameraGeometryError and no rays are
    returned. A valid observation without regions gives an empty result.
    """
    if not isinstance(observation, ThermalFrameObservation):
        raise CameraGeometryError(
            f'expected ThermalFrameObservation, not {type(observation).__name__}')
    if not isinstance(intrinsics, PinholeIntrinsics):
        raise CameraGeometryError(f'expected PinholeIntrinsics, not {type(intrinsics).__name__}')
    # Validated up front so that an invalid pose fails even for an empty frame.
    frame_id, translation, rotation = validated_pose(frame_id, translation, rotation)
    if observation.source_frame_id != intrinsics.frame_id:
        raise CameraGeometryError(
            f'observation frame {observation.source_frame_id!r} does not match the '
            f'calibration frame {intrinsics.frame_id!r}')
    extraction = observation.extraction
    width = _int('image_width', extraction.image_width, 1)
    height = _int('image_height', extraction.image_height, 1)
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise CameraGeometryError(
            f'observation image {width} x {height} does not '
            f'match the calibration {intrinsics.width} x {intrinsics.height}')
    # Validate and copy every region before building any ray.
    validated = [_validated_region(index, region, width, height)
                 for index, region in enumerate(_sequence('regions', extraction.regions))]
    rays = tuple(
        RegionRay(
            region_index=index,
            centroid_xy=centroid,
            touches_image_boundary=touches,
            ray=pixel_to_ray(centroid[0], centroid[1], intrinsics, frame_id=frame_id,
                             translation=translation, rotation=rotation),
        )
        for index, (centroid, touches) in enumerate(validated)
    )
    return ThermalObservationRays(
        source_stamp_ns=observation.source_stamp_ns,
        source_frame_id=observation.source_frame_id,
        image_width=width,
        image_height=height,
        rays=rays,
    )
