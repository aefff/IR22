import dataclasses
import json

import cv2
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalExtractionResult,
    ThermalFrameObservation,
    ThermalRegion,
)
import numpy as np
import pytest

HOT = 200


def _image(height, width):
    return np.zeros((height, width), dtype=np.uint8)


def _legacy_hot_pixel_count(image, threshold=180):
    # Mirrors the thresholding of the original (pre-M2.2) fire_detection.py.
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(mask)


# 1. Completely cold image.
def test_cold_image_is_valid_empty_result():
    result = extract_thermal_regions(_image(10, 12))
    assert result == ThermalExtractionResult(
        image_width=12,
        image_height=10,
        threshold=180,
        min_area_px=1,
        connectivity=8,
        hot_pixel_count=0,
        regions=(),
    )


# 2 and 4. One rectangle with exact bounding box, centroid and statistics.
def test_single_rectangle_exact_measurements():
    image = _image(10, 12)
    image[2:5, 3:7] = HOT  # rows 2-4, columns 3-6
    result = extract_thermal_regions(image)
    assert result.hot_pixel_count == 12
    assert result.regions == (
        ThermalRegion(
            bbox_xywh=(3, 2, 4, 3),
            centroid_xy=(4.5, 3.0),
            area_px=12,
            mean_intensity=200.0,
            max_intensity=200,
            touches_image_boundary=False,
        ),
    )


# 3. Multiple disconnected regions, in raster order of their first pixel.
def test_multiple_regions_ordered_by_first_pixel():
    image = _image(10, 12)
    image[1:3, 1:3] = 190  # A: first pixel (row 1, col 1)
    image[6:9, 8:11] = 250  # B: first pixel (row 6, col 8)
    image[1, 9] = 181  # C: first pixel (row 1, col 9)
    result = extract_thermal_regions(image)
    assert result.hot_pixel_count == 4 + 9 + 1
    assert [r.bbox_xywh for r in result.regions] == [(1, 1, 2, 2), (9, 1, 1, 1), (8, 6, 3, 3)]
    assert [r.max_intensity for r in result.regions] == [190, 181, 250]


# 5. Irregular component: statistics use component pixels only, not the bounding box,
# which here also contains cold pixels and a separate hot component.
def test_irregular_component_statistics_exclude_other_bbox_pixels():
    image = _image(7, 7)
    for y, value in zip(range(1, 5), (190, 200, 210, 220)):
        image[y, 1] = value  # vertical bar, x = 1
    for x, value in zip(range(2, 5), (230, 240, 250)):
        image[4, x] = value  # horizontal bar, y = 4
    image[1, 4] = 255  # separate pixel inside the L's bounding box, not adjacent to it

    result = extract_thermal_regions(image)
    assert result.hot_pixel_count == 8
    l_shape, single = result.regions
    assert l_shape.bbox_xywh == (1, 1, 4, 4)
    assert l_shape.area_px == 7
    assert l_shape.mean_intensity == pytest.approx(1540 / 7)
    assert l_shape.max_intensity == 250
    assert l_shape.centroid_xy == pytest.approx((13 / 7, 22 / 7))
    assert single == ThermalRegion((4, 1, 1, 1), (4.0, 1.0), 1, 255.0, 255, False)


# 6. Strict threshold semantics at 179, 180 and 181.
@pytest.mark.parametrize('threshold, expected_hot_columns', [
    (178, [0, 1, 2]),
    (179, [1, 2]),
    (180, [2]),
    (181, []),
])
def test_threshold_is_strictly_greater_than(threshold, expected_hot_columns):
    image = np.array([[179, 180, 181]], dtype=np.uint8)
    result = extract_thermal_regions(image, threshold=threshold)
    assert result.hot_pixel_count == len(expected_hot_columns)
    assert result.hot_pixel_count == _legacy_hot_pixel_count(image, threshold)
    if expected_hot_columns:
        (region,) = result.regions
        assert region.bbox_xywh == (expected_hot_columns[0], 0, len(expected_hot_columns), 1)
    else:
        assert result.regions == ()


# 7. Whole-image counts of 50 and 51 reproduce the original detector's alarm.
@pytest.mark.parametrize('hot_pixels, legacy_alarm', [(50, False), (51, True)])
def test_legacy_alarm_boundary(hot_pixels, legacy_alarm):
    image = _image(10, 10)
    image.reshape(-1)[:hot_pixels] = HOT
    result = extract_thermal_regions(image)
    assert result.hot_pixel_count == hot_pixels == _legacy_hot_pixel_count(image)
    assert (result.hot_pixel_count > 50) is legacy_alarm


def test_hot_pixel_count_matches_legacy_detector_on_random_images():
    rng = np.random.default_rng(22)
    for _ in range(20):
        image = rng.integers(0, 256, size=(24, 32), dtype=np.uint8)
        result = extract_thermal_regions(image)
        assert result.hot_pixel_count == _legacy_hot_pixel_count(image)


# 8 and 9. Whole-image count is independent of per-region filtering, and
# minimum-area filtering is inclusive.
@pytest.mark.parametrize('min_area_px, expected_regions', [(30, 2), (31, 0), (51, 0)])
def test_two_30_pixel_regions_count_vs_area_filter(min_area_px, expected_regions):
    image = _image(20, 20)
    image[1:4, 1:11] = HOT  # 3 x 10 = 30 pixels
    image[10:13, 1:11] = HOT  # 3 x 10 = 30 pixels, not adjacent to the first
    result = extract_thermal_regions(image, min_area_px=min_area_px)
    assert result.hot_pixel_count == 60
    assert result.hot_pixel_count > 50  # original alarm fires...
    assert len(result.regions) == expected_regions  # ...whatever the region filter keeps
    assert all(r.area_px == 30 for r in result.regions)


# 10. Diagonal adjacency joins with 8-connectivity but not with 4-connectivity.
def test_diagonal_pixels_and_connectivity():
    image = _image(3, 3)
    image[0, 0] = HOT
    image[1, 1] = HOT
    eight = extract_thermal_regions(image, connectivity=8)
    four = extract_thermal_regions(image, connectivity=4)
    assert [r.bbox_xywh for r in eight.regions] == [(0, 0, 2, 2)]
    assert [r.bbox_xywh for r in four.regions] == [(0, 0, 1, 1), (1, 1, 1, 1)]
    assert eight.hot_pixel_count == four.hot_pixel_count == 2


# 11. Boundary contact on each edge, plus an interior control.
def test_regions_touching_each_image_edge():
    image = _image(5, 6)
    image[0, 2] = HOT  # top
    image[2, 0] = HOT  # left
    image[2, 2] = HOT  # interior
    image[2, 5] = HOT  # right
    image[4, 2] = HOT  # bottom
    result = extract_thermal_regions(image)
    assert [(r.bbox_xywh[:2], r.touches_image_boundary) for r in result.regions] == [
        ((2, 0), True),
        ((0, 2), True),
        ((2, 2), False),
        ((5, 2), True),
        ((2, 4), True),
    ]


# 12. All-hot image.
def test_all_hot_image_is_one_region():
    image = np.full((4, 5), 255, dtype=np.uint8)
    result = extract_thermal_regions(image)
    assert result.hot_pixel_count == 20
    assert result.regions == (
        ThermalRegion((0, 0, 5, 4), (2.0, 1.5), 20, 255.0, 255, True),
    )


# 13. Invalid images and parameters are rejected, never converted.
@pytest.mark.parametrize('image, error', [
    ([[0, 200], [200, 0]], TypeError),
    (np.ma.masked_array(np.zeros((2, 2), dtype=np.uint8)), TypeError),
    (np.zeros((2, 2), dtype=np.uint16), TypeError),
    (np.zeros((2, 2), dtype=np.float32), TypeError),
    (np.zeros((2, 2), dtype=np.int8), TypeError),
    (np.zeros((2, 2), dtype=bool), TypeError),
    (np.zeros(4, dtype=np.uint8), ValueError),
    (np.zeros((2, 2, 1), dtype=np.uint8), ValueError),
    (np.zeros((2, 2, 3), dtype=np.uint8), ValueError),
    (np.zeros((0, 5), dtype=np.uint8), ValueError),
])
def test_invalid_images_raise(image, error):
    with pytest.raises(error):
        extract_thermal_regions(image)


@pytest.mark.parametrize('kwargs, error', [
    ({'threshold': -1}, ValueError),
    ({'threshold': 255}, ValueError),
    ({'threshold': 180.0}, TypeError),
    ({'threshold': True}, TypeError),
    ({'min_area_px': 0}, ValueError),
    ({'min_area_px': 1.5}, TypeError),
    ({'connectivity': 6}, ValueError),
    ({'connectivity': '8'}, TypeError),
])
def test_invalid_parameters_raise(kwargs, error):
    with pytest.raises(error):
        extract_thermal_regions(_image(3, 3), **kwargs)


def test_numpy_integer_parameters_are_stored_as_plain_ints():
    result = extract_thermal_regions(_image(3, 3), threshold=np.int64(100))
    assert type(result.threshold) is int


# 14. Non-contiguous and Fortran-ordered inputs give the same result as a C-contiguous copy.
def test_non_contiguous_and_fortran_inputs():
    base = _image(10, 20)
    base[2:4, 4:8] = HOT
    view = base[:, ::2]
    assert not view.flags.c_contiguous
    expected = extract_thermal_regions(np.ascontiguousarray(view))
    assert extract_thermal_regions(view) == expected
    assert extract_thermal_regions(np.asfortranarray(view)) == expected
    assert [r.bbox_xywh for r in expected.regions] == [(2, 2, 2, 2)]


# 15 and 16. Determinism, input immutability and immutable outputs.
def test_deterministic_and_does_not_modify_input():
    rng = np.random.default_rng(7)
    image = rng.integers(150, 256, size=(30, 40), dtype=np.uint8)
    original = image.copy()
    image.flags.writeable = False  # any write by the function would raise

    first = extract_thermal_regions(image)
    second = extract_thermal_regions(image)
    assert first == second
    assert np.array_equal(image, original)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.hot_pixel_count = 0
    assert isinstance(first.regions, tuple)


def test_results_contain_only_plain_python_values():
    image = _image(6, 6)
    image[1:3, 1:4] = np.array([[181, 190, 200], [210, 220, 255]], dtype=np.uint8)
    result = extract_thermal_regions(image)
    for value in dataclasses.astuple(result.regions[0]):
        leaves = value if isinstance(value, tuple) else (value,)
        assert all(type(leaf) in (int, float, bool) for leaf in leaves)
    # Round-trips through JSON without custom encoders.
    assert json.loads(json.dumps(dataclasses.asdict(result)))['hot_pixel_count'] == 6


# 17. Observation metadata: None means missing, zero is a valid timestamp.
def test_observation_preserves_missing_metadata_and_zero_stamp():
    extraction = extract_thermal_regions(_image(2, 2))
    missing = ThermalFrameObservation(None, None, extraction)
    assert missing.source_stamp_ns is None
    assert missing.source_frame_id is None

    zero = ThermalFrameObservation(0, 'thermal_link', extraction)
    assert zero.source_stamp_ns == 0
    assert zero.source_stamp_ns is not None
    assert zero.extraction is extraction

    assert not hasattr(extraction, 'source_stamp_ns')  # extraction never invents metadata


@pytest.mark.parametrize('stamp, frame_id, error', [
    (-1, None, ValueError),
    (True, None, TypeError),
    (1.5, None, TypeError),
    (np.int64(5), None, TypeError),
    (None, '', ValueError),
    (None, 5, TypeError),
])
def test_observation_rejects_invalid_metadata(stamp, frame_id, error):
    extraction = extract_thermal_regions(_image(2, 2))
    with pytest.raises(error):
        ThermalFrameObservation(stamp, frame_id, extraction)


def test_observation_requires_an_extraction_result():
    with pytest.raises(TypeError):
        ThermalFrameObservation(0, 'thermal_link', None)
