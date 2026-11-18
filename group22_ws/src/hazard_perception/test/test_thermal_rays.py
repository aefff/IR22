import dataclasses
import math

from hazard_perception.camera_geometry import (
    CameraGeometryError,
    elevation,
    FRAME_CONVENTION_BODY,
    horizontal_bearing,
    IDENTITY,
    intrinsics_from_calibration,
    RawCameraCalibration,
    WAREHOUSE_THERMAL_PRINCIPAL_POINT_SHIFT_PX,
)
from hazard_perception.thermal_rays import observation_to_rays, RegionRay, ThermalObservationRays
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalFrameObservation,
    ThermalRegion,
)
import numpy as np
import pytest

FX = 160 / math.tan(1.047 / 2)
K = intrinsics_from_calibration(
    RawCameraCalibration(frame_id='thermal_link', width=320, height=240,
                         k=[FX, 0.0, 160.0, 0.0, FX, 120.0, 0.0, 0.0, 1.0],
                         distortion_model='plumb_bob', d=[0.0] * 5),
    principal_point_shift_px=WAREHOUSE_THERMAL_PRINCIPAL_POINT_SHIFT_PX,
    frame_convention=FRAME_CONVENTION_BODY)
MOUNT = {'frame_id': 'base_link', 'translation': (0.185, -0.04, 0.13), 'rotation': IDENTITY}


def _observation(image, stamp_ns=12_000_000_345, frame_id='thermal_link'):
    return ThermalFrameObservation(stamp_ns, frame_id, extract_thermal_regions(image))


def _image():
    image = np.full((240, 320), 96, dtype=np.uint8)
    image[100:110, 155:165] = 200  # centred region
    image[10:16, 0:4] = 220  # region touching the left edge, upper left
    return image


def test_empty_frame_gives_no_rays_but_keeps_metadata():
    rays = observation_to_rays(_observation(np.full((240, 320), 96, np.uint8)), K, **MOUNT)
    assert rays == ThermalObservationRays(12_000_000_345, 'thermal_link', 320, 240, ())


def test_multiple_regions_keep_order_index_centroid_and_boundary_contact():
    observation = _observation(_image())
    rays = observation_to_rays(observation, K, **MOUNT)
    regions = observation.extraction.regions
    assert [r.region_index for r in rays.rays] == [0, 1]
    assert [r.centroid_xy for r in rays.rays] == [region.centroid_xy for region in regions]
    assert [r.touches_image_boundary for r in rays.rays] == [True, False]
    assert all(isinstance(r, RegionRay) for r in rays.rays)
    for region_ray in rays.rays:
        assert region_ray.ray.frame_id == 'base_link'
        assert region_ray.ray.origin == (0.185, -0.04, 0.13)
        assert math.sqrt(sum(c * c for c in region_ray.ray.direction)) == pytest.approx(1.0)
    edge, centre = rays.rays
    assert horizontal_bearing(centre.ray.direction) == pytest.approx(0.0)  # centroid 159.5
    assert horizontal_bearing(edge.ray.direction) > 0  # image left = robot left
    assert elevation(edge.ray.direction) > 0  # image top = up


def test_zero_timestamp_and_missing_stamp_are_preserved():
    assert observation_to_rays(_observation(_image(), stamp_ns=0), K, **MOUNT).source_stamp_ns == 0
    assert observation_to_rays(_observation(_image(), stamp_ns=None), K, **MOUNT) \
        .source_stamp_ns is None


@pytest.mark.parametrize('frame_id', [None, 'camera_link', 'thermal_optical'])
def test_observation_from_another_frame_is_rejected(frame_id):
    with pytest.raises(CameraGeometryError, match='calibration frame'):
        observation_to_rays(_observation(_image(), frame_id=frame_id), K, **MOUNT)


def test_observation_with_another_image_size_is_rejected():
    small = ThermalFrameObservation(1, 'thermal_link',
                                    extract_thermal_regions(np.zeros((120, 160), np.uint8)))
    with pytest.raises(CameraGeometryError, match='does not match the calibration'):
        observation_to_rays(small, K, **MOUNT)


@pytest.mark.parametrize('pose', [
    {'frame_id': '', 'translation': (0.0, 0.0, 0.0), 'rotation': IDENTITY},
    {'frame_id': 'base_link', 'translation': (0.0, 0.0), 'rotation': IDENTITY},
    {'frame_id': 'base_link', 'translation': (0.0, 0.0, 0.0),
     'rotation': ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))},
])
def test_invalid_pose_is_rejected_even_for_an_empty_frame(pose):
    empty = _observation(np.full((240, 320), 96, np.uint8))
    with pytest.raises(CameraGeometryError):
        observation_to_rays(empty, K, **pose)


def test_out_of_image_centroid_is_rejected():
    bad = ThermalRegion((0, 0, 1, 1), (320.0, 10.0), 1, 200.0, 200, True)
    extraction = dataclasses.replace(extract_thermal_regions(_image()), regions=(bad,))
    with pytest.raises(CameraGeometryError, match='outside'):
        observation_to_rays(ThermalFrameObservation(1, 'thermal_link', extraction), K, **MOUNT)


@pytest.mark.parametrize('observation, intrinsics', [
    ('not an observation', K),
    (None, K),
])
def test_wrong_input_types_are_rejected(observation, intrinsics):
    with pytest.raises(CameraGeometryError):
        observation_to_rays(observation, intrinsics, **MOUNT)


def test_raw_calibration_is_not_accepted_as_intrinsics():
    raw = RawCameraCalibration(frame_id='thermal_link', width=320, height=240,
                               k=[FX, 0.0, 160.0, 0.0, FX, 120.0, 0.0, 0.0, 1.0],
                               distortion_model='plumb_bob', d=[0.0] * 5)
    with pytest.raises(CameraGeometryError, match='PinholeIntrinsics'):
        observation_to_rays(_observation(_image()), raw, **MOUNT)


# --- R2.1 regressions (Codex counterexamples) ---------------------------------

def _with_regions(*regions, width=320, height=240):
    base = extract_thermal_regions(np.full((240, 320), 96, np.uint8))
    extraction = dataclasses.replace(base, image_width=width, image_height=height,
                                     regions=regions)
    return ThermalFrameObservation(7, 'thermal_link', extraction)


def _region(bbox, centroid, touches):
    return ThermalRegion(bbox, centroid, 1, 200.0, 200, touches)


VALID = _region((10, 10, 2, 2), (10.5, 10.5), False)


@pytest.mark.parametrize('region, message', [
    (_region((-1, 5, 2, 1), (0.0, 5.0), True), 'bbox x'),  # negative bbox x
    (_region((5, -1, 1, 2), (5.0, 0.0), True), 'bbox y'),  # negative bbox y
    (_region((318, 5, 5, 1), (319.0, 5.0), True), 'outside the 320 x 240'),  # right edge
    (_region((5, 238, 1, 5), (5.0, 239.0), True), 'outside the 320 x 240'),  # bottom edge
    (_region((10, 10, 0, 2), (10.0, 10.5), False), 'bbox width'),  # zero width
    (_region((10, 10, 2, -2), (10.5, 10.5), False), 'bbox height'),
    (_region((10.0, 10, 2, 2), (10.5, 10.5), False), 'bbox x'),  # float position
    (_region((10, 10, 2.0, 2), (10.5, 10.5), False), 'bbox width'),  # float size
    (_region((10, 10, 2), (10.5, 10.5), False), '4 elements'),
    (_region((10, 10, 2, 2), (50.0, 50.0), False), 'outside its bbox'),  # centroid outside
    (_region((10, 10, 2, 2), (11.5, 10.5), False), 'outside its bbox'),  # past last pixel
    (_region((0, 5, 1, 1), (-0.25, 5.0), True), 'outside its bbox'),  # negative centroid
    (_region((10, 10, 2, 2), (10.5, 10.5, 0.0), False), '2 elements'),  # R21-4 3-element
    (_region((10, 10, 2, 2), {0: 10.5, 1: 10.5}, False), 'sequence'),
    (_region((10, 10, 2, 2), (10.5, 10.5), 1), 'must be a bool'),  # non-bool flag
    (_region((10, 10, 2, 2), (10.5, 10.5), True), 'inconsistent'),  # interior marked True
    (_region((0, 10, 2, 2), (0.5, 10.5), False), 'inconsistent'),  # edge marked False
    (_region((318, 10, 2, 2), (318.5, 10.5), False), 'inconsistent'),  # right edge
])
def test_r21_2_inconsistent_region_geometry_is_rejected(region, message):
    with pytest.raises(CameraGeometryError, match=message):
        observation_to_rays(_with_regions(region), K, **MOUNT)


@pytest.mark.parametrize('width, height', [(320.0, 240), (320, 240.0), (True, 240), (0, 240)])
def test_r21_2_invalid_image_dimensions_are_rejected(width, height):
    with pytest.raises(CameraGeometryError, match='image_'):
        observation_to_rays(_with_regions(VALID, width=width, height=height), K, **MOUNT)


def test_r21_2_detector_output_is_accepted_including_edges():
    rng = np.random.default_rng(2101)
    regions = 0
    for _ in range(3):
        image = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
        image[:, rng.integers(0, 320)] = 255  # full-height column: touches top and bottom
        for connectivity in (4, 8):
            for min_area in (1, 5):
                extraction = extract_thermal_regions(
                    image, min_area_px=min_area, connectivity=connectivity)
                rays = observation_to_rays(
                    ThermalFrameObservation(1, 'thermal_link', extraction), K, **MOUNT)
                assert len(rays.rays) == len(extraction.regions)
                regions += len(extraction.regions)
    assert regions > 1_000


def test_r21_2_single_pixel_regions_at_every_corner_are_accepted():
    image = np.zeros((240, 320), np.uint8)
    for r, c in ((0, 0), (0, 319), (239, 0), (239, 319), (120, 160)):
        image[r, c] = 255
    rays = observation_to_rays(_observation(image), K, **MOUNT)
    assert [r.touches_image_boundary for r in rays.rays] == [True, True, False, True, True]


def test_r21_3_output_is_independent_of_caller_owned_containers():
    centroid = [10.5, 10.5]
    bbox = [10, 10, 2, 2]
    rays = observation_to_rays(_with_regions(_region(bbox, centroid, False)), K, **MOUNT)
    record = rays.rays[0]
    direction = record.ray.direction
    centroid[0], bbox[0] = 300.0, 200
    assert record.centroid_xy == (10.5, 10.5)
    assert type(record.centroid_xy) is tuple
    assert all(type(c) is float for c in record.centroid_xy)
    assert record.ray.direction == direction
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.centroid_xy = (1.0, 1.0)


def _a_ray():
    return observation_to_rays(_with_regions(VALID), K, **MOUNT).rays[0].ray


@pytest.mark.parametrize('kwargs', [
    {'region_index': -1}, {'region_index': True}, {'region_index': 1.0},
    {'centroid_xy': (1.0,)}, {'centroid_xy': {0: 1.0, 1: 2.0}},
    {'centroid_xy': (float('nan'), 1.0)},
    {'touches_image_boundary': 0}, {'ray': (0.0, 0.0, 1.0)},
])
def test_r21_3_region_ray_enforces_field_invariants(kwargs):
    fields = {'region_index': 0, 'centroid_xy': (10.5, 10.5), 'touches_image_boundary': False,
              'ray': _a_ray()}
    fields.update(kwargs)
    with pytest.raises(CameraGeometryError):
        RegionRay(**fields)


def test_r21_3_region_ray_copies_its_centroid():
    centroid = [10.5, 10.5]
    record = RegionRay(0, centroid, False, _a_ray())
    centroid[0] = 99.0
    assert record.centroid_xy == (10.5, 10.5)


def _record(index, frame='base_link'):
    ray = _a_ray()
    return RegionRay(index, (10.5, 10.5), False,
                     dataclasses.replace(ray, frame_id=frame) if frame != ray.frame_id else ray)


@pytest.mark.parametrize('kwargs', [
    {'source_stamp_ns': -1}, {'source_stamp_ns': True}, {'source_frame_id': ''},
    {'source_frame_id': None}, {'image_width': 320.0}, {'image_height': 0},
    {'rays': [_a_ray()]}, {'rays': (_record(1), _record(0))}, {'rays': (_record(0), _record(0))},
    {'rays': (_record(0), _record(1, 'odom'))}, {'rays': {0: _record(0)}},
])
def test_r21_3_frame_record_enforces_invariants(kwargs):
    fields = {'source_stamp_ns': 1, 'source_frame_id': 'thermal_link', 'image_width': 320,
              'image_height': 240, 'rays': ()}
    fields.update(kwargs)
    with pytest.raises(CameraGeometryError):
        ThermalObservationRays(**fields)


def test_r21_3_frame_record_copies_its_ray_list():
    rays = [_record(0)]
    frame = ThermalObservationRays(1, 'thermal_link', 320, 240, rays)
    rays.append(_record(1))
    assert len(frame.rays) == 1 and type(frame.rays) is tuple


def test_all_or_nothing_valid_region_followed_by_invalid_region():
    invalid = _region((10, 10, 2, 2), (50.0, 50.0), False)
    observation = _with_regions(VALID, invalid)
    with pytest.raises(CameraGeometryError, match=r'regions\[1\]'):
        observation_to_rays(observation, K, **MOUNT)
    # The valid region alone converts; the empty frame converts to an empty result.
    assert len(observation_to_rays(_with_regions(VALID), K, **MOUNT).rays) == 1
    assert observation_to_rays(_with_regions(), K, **MOUNT).rays == ()


def test_all_or_nothing_regions_container_must_be_a_sequence():
    observation = _with_regions(VALID)
    bad = dataclasses.replace(observation.extraction, regions={0: VALID})
    with pytest.raises(CameraGeometryError, match='regions must be a sequence'):
        observation_to_rays(dataclasses.replace(observation, extraction=bad), K, **MOUNT)
