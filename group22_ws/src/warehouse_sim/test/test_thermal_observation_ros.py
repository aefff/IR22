import copy
import dataclasses
from types import SimpleNamespace

from fake_interfaces import read_msg_fields
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalExtractionResult,
    ThermalFrameObservation,
    ThermalRegion,
)
import numpy as np
import pytest
from ros_stand_ins import any_stand_ins_installed
from warehouse_sim import thermal_observation_ros
from warehouse_sim.thermal_observation_ros import (
    MAX_STAMP_NS,
    observation_to_message,
    ObservationConversionError,
    stamp_to_ns,
)

REGION_FIELDS = (
    'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'centroid_x', 'centroid_y',
    'area_px', 'mean_intensity', 'max_intensity', 'touches_image_boundary',
)
FRAME_FIELDS = (
    'image_width', 'image_height', 'threshold', 'min_area_px', 'connectivity',
    'hot_pixel_count',
)


@pytest.fixture(params=['fake', 'real'])
def message_classes(request):
    if request.param == 'fake':
        return request.getfixturevalue('fake_interface_messages')
    if any_stand_ins_installed():
        pytest.skip('real generated messages need a built and sourced ROS 2 workspace')
    return (
        thermal_observation_ros.ThermalFrameObservationMsg,
        thermal_observation_ros.ThermalRegionMsg,
    )


def _two_region_observation(stamp_ns=12_000_000_345, frame_id='thermal_link'):
    image = np.zeros((8, 12), dtype=np.uint8)
    image[1, 1:4] = (181, 200, 201)  # region A: an L shape with a fractional mean
    image[2, 1] = 255
    image[4:7, 11] = (190, 195, 200)  # region B: on the right edge
    return ThermalFrameObservation(stamp_ns, frame_id, extract_thermal_regions(image))


def _as_dict(message):
    return {
        'stamp': (message.header.stamp.sec, message.header.stamp.nanosec),
        'frame_id': message.header.frame_id,
        **{name: getattr(message, name) for name in FRAME_FIELDS},
        'regions': [
            {name: getattr(region, name) for name in REGION_FIELDS}
            for region in message.regions
        ],
    }


def _with_extraction(observation, **changes):
    extraction = dataclasses.replace(observation.extraction, **changes)
    return dataclasses.replace(observation, extraction=extraction)


def _with_region(observation, **changes):
    region = dataclasses.replace(observation.extraction.regions[0], **changes)
    return _with_extraction(observation, regions=(region,))


# --- The interface definitions ------------------------------------------------

def test_msg_files_define_the_agreed_fields():
    assert read_msg_fields('ThermalRegion') == {
        'bbox_x': 'uint32', 'bbox_y': 'uint32', 'bbox_width': 'uint32',
        'bbox_height': 'uint32', 'centroid_x': 'float64', 'centroid_y': 'float64',
        'area_px': 'uint32', 'mean_intensity': 'float64', 'max_intensity': 'uint8',
        'touches_image_boundary': 'bool',
    }
    assert read_msg_fields('ThermalFrameObservation') == {
        'header': 'std_msgs/Header', 'image_width': 'uint32', 'image_height': 'uint32',
        'threshold': 'uint8', 'min_area_px': 'uint32', 'connectivity': 'uint8',
        'hot_pixel_count': 'uint32', 'regions': 'ThermalRegion[]',
    }


# --- Conversion ---------------------------------------------------------------

def test_every_field_maps_exactly(message_classes):
    frame_class, region_class = message_classes
    message = observation_to_message(_two_region_observation())

    assert type(message) is frame_class
    assert all(type(region) is region_class for region in message.regions)
    assert _as_dict(message) == {
        'stamp': (12, 345),
        'frame_id': 'thermal_link',
        'image_width': 12, 'image_height': 8,
        'threshold': 180, 'min_area_px': 1, 'connectivity': 8,
        'hot_pixel_count': 7,
        'regions': [
            {'bbox_x': 1, 'bbox_y': 1, 'bbox_width': 3, 'bbox_height': 2,
             'centroid_x': 1.75, 'centroid_y': 1.25, 'area_px': 4,
             'mean_intensity': 209.25, 'max_intensity': 255,
             'touches_image_boundary': False},
            {'bbox_x': 11, 'bbox_y': 4, 'bbox_width': 1, 'bbox_height': 3,
             'centroid_x': 11.0, 'centroid_y': 5.0, 'area_px': 3,
             'mean_intensity': 195.0, 'max_intensity': 200,
             'touches_image_boundary': True},
        ],
    }


def test_fractional_values_keep_full_precision(message_classes):
    observation = _with_region(
        _two_region_observation(),
        centroid_xy=(1 / 3, 2 / 7), mean_intensity=1540 / 7,
    )
    region = observation_to_message(observation).regions[0]
    assert (region.centroid_x, region.centroid_y) == (1 / 3, 2 / 7)
    assert region.mean_intensity == 1540 / 7


def test_cold_frame_gives_empty_region_array(message_classes):
    image = np.full((240, 320), 96, dtype=np.uint8)
    message = observation_to_message(
        ThermalFrameObservation(0, 'thermal_link', extract_thermal_regions(image)),
    )
    assert list(message.regions) == []
    assert message.hot_pixel_count == 0
    assert (message.image_width, message.image_height) == (320, 240)


def test_region_order_is_preserved_not_sorted(message_classes):
    observation = _two_region_observation()
    first, second = observation.extraction.regions
    reversed_observation = _with_extraction(observation, regions=(second, first))
    message = observation_to_message(reversed_observation)
    assert [region.bbox_x for region in message.regions] == [11, 1]


@pytest.mark.parametrize('stamp_ns, sec, nanosec', [
    (0, 0, 0),
    (1, 0, 1),
    (1_999_999_999, 1, 999_999_999),
    (MAX_STAMP_NS, 2_147_483_647, 999_999_999),
])
def test_timestamps_are_split_exactly(message_classes, stamp_ns, sec, nanosec):
    message = observation_to_message(_two_region_observation(stamp_ns=stamp_ns))
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (sec, nanosec)


def test_largest_supported_timestamp_is_the_agreed_value():
    assert MAX_STAMP_NS == 2_147_483_647_999_999_999


def test_timestamp_beyond_ros_range_is_rejected(message_classes):
    with pytest.raises(ObservationConversionError, match='source_stamp_ns'):
        observation_to_message(_two_region_observation(stamp_ns=MAX_STAMP_NS + 1))


def test_missing_timestamp_is_rejected(message_classes):
    with pytest.raises(ObservationConversionError, match='acquisition timestamp'):
        observation_to_message(_two_region_observation(stamp_ns=None))


def test_missing_frame_id_becomes_empty_string(message_classes):
    message = observation_to_message(_two_region_observation(frame_id=None))
    assert message.header.frame_id == ''


@pytest.mark.parametrize('stamp', [True, 1.5, np.int64(5)])
def test_invalid_timestamp_types_are_rejected(message_classes, stamp):
    observation = _two_region_observation()
    # The dataclass already rejects these; bypass it to test the converter's own check.
    object.__setattr__(observation, 'source_stamp_ns', stamp)
    with pytest.raises(ObservationConversionError, match='source_stamp_ns'):
        observation_to_message(observation)


@pytest.mark.parametrize('field, value', [
    ('threshold', True),
    ('threshold', 180.0),
    ('threshold', 256),
    ('connectivity', -1),
    ('min_area_px', 1.5),
    ('image_width', np.int64(320)),
    ('hot_pixel_count', -1),
    ('hot_pixel_count', 2**32),
])
def test_invalid_frame_integers_are_rejected(message_classes, field, value):
    observation = _with_extraction(_two_region_observation(), **{field: value})
    with pytest.raises(ObservationConversionError, match=field):
        observation_to_message(observation)


@pytest.mark.parametrize('changes, field', [
    ({'bbox_xywh': (True, 1, 3, 2)}, 'bbox_x'),
    ({'bbox_xywh': (1, 1, -3, 2)}, 'bbox_width'),
    ({'bbox_xywh': (1, 1, 3, 2**32)}, 'bbox_height'),
    ({'area_px': 4.0}, 'area_px'),
    ({'max_intensity': 256}, 'max_intensity'),
    ({'touches_image_boundary': 1}, 'touches_image_boundary'),
])
def test_invalid_region_values_are_rejected(message_classes, changes, field):
    observation = _with_region(_two_region_observation(), **changes)
    with pytest.raises(ObservationConversionError, match=rf'regions\[0\]\.{field}'):
        observation_to_message(observation)


@pytest.mark.parametrize('changes, field', [
    ({'centroid_xy': (float('nan'), 1.0)}, 'centroid_x'),
    ({'centroid_xy': (1.0, float('inf'))}, 'centroid_y'),
    ({'mean_intensity': float('-inf')}, 'mean_intensity'),
    ({'mean_intensity': float('nan')}, 'mean_intensity'),
    ({'mean_intensity': 200}, 'mean_intensity'),  # an int is not silently converted
])
def test_invalid_float_values_are_rejected(message_classes, changes, field):
    observation = _with_region(_two_region_observation(), **changes)
    with pytest.raises(ObservationConversionError, match=rf'regions\[0\]\.{field}'):
        observation_to_message(observation)


def test_non_observation_input_is_a_type_error(message_classes):
    with pytest.raises(TypeError):
        observation_to_message(SimpleNamespace(source_stamp_ns=0))


def test_each_call_returns_new_objects_and_leaves_input_unchanged(message_classes):
    observation = _two_region_observation()
    snapshot = copy.deepcopy(observation)

    first = observation_to_message(observation)
    second = observation_to_message(observation)
    assert first is not second
    assert first.header is not second.header
    assert first.regions is not second.regions
    assert all(a is not b for a, b in zip(first.regions, second.regions))

    first.regions[0].area_px = 999
    first.regions.pop()
    first.header.frame_id = 'changed'
    assert _as_dict(second) == _as_dict(observation_to_message(snapshot))
    assert observation == snapshot


# --- Source timestamp validation ---------------------------------------------

def _stamp(sec, nanosec):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


@pytest.mark.parametrize('sec, nanosec, expected_ns', [
    (0, 0, 0),
    (12, 345, 12_000_000_345),
    (0, 999_999_999, 999_999_999),
    (2_147_483_647, 999_999_999, MAX_STAMP_NS),
])
def test_valid_source_stamps_are_exact(sec, nanosec, expected_ns):
    ns = stamp_to_ns(_stamp(sec, nanosec))
    assert ns == expected_ns
    assert type(ns) is int


def test_decreasing_source_stamps_are_both_valid():
    # Simulation time can jump backwards after a reset; no monotonicity check.
    assert [stamp_to_ns(_stamp(s, 0)) for s in (10, 5)] == [10 * 10**9, 5 * 10**9]


@pytest.mark.parametrize('sec, nanosec', [
    (5, 1_000_000_000),  # must not become six seconds
    (5, -1),
    (-1, 0),
    (-1, 999_999_999),
    (2**31, 0),
    (-2**31 - 1, 0),
])
def test_out_of_range_source_stamps_are_rejected(sec, nanosec):
    with pytest.raises(ValueError):
        stamp_to_ns(_stamp(sec, nanosec))


@pytest.mark.parametrize('sec, nanosec', [
    (True, 0),
    (0, False),
    (1.0, 0),
    (0, 0.5),
    (np.int32(1), 0),
    ('1', 0),
])
def test_wrongly_typed_source_stamps_are_rejected(sec, nanosec):
    with pytest.raises(TypeError):
        stamp_to_ns(_stamp(sec, nanosec))


def test_region_dataclass_fields_are_all_converted():
    # Guards against a dataclass field being added without a message field.
    converted = {
        'bbox_xywh': {'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'},
        'centroid_xy': {'centroid_x', 'centroid_y'},
    }
    region_fields = set()
    for field in dataclasses.fields(ThermalRegion):
        region_fields |= converted.get(field.name, {field.name})
    assert region_fields == set(read_msg_fields('ThermalRegion'))

    frame_fields = {f.name for f in dataclasses.fields(ThermalExtractionResult)}
    assert frame_fields == set(read_msg_fields('ThermalFrameObservation')) - {'header'}
