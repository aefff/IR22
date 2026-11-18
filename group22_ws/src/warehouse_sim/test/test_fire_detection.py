from types import SimpleNamespace

import cv2
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalExtractionResult,
    ThermalFrameObservation,
    ThermalRegion,
)
import numpy as np
import pytest
from ros_stand_ins import any_stand_ins_installed
from warehouse_sim import fire_detection
from warehouse_sim.fire_detection import CvBridgeError, FireDetection, stamp_to_ns
from warehouse_sim.thermal_observation_ros import ObservationConversionError

HOT = 200
WARNING_51 = 'FIRE HAZARD DETECTED! Hot pixels: 51'

ROS_STUBBED = any_stand_ins_installed()


@pytest.fixture(autouse=True)
def _fake_messages(fake_interface_messages):
    # Callback tests publish fake messages; real ones are covered in test_ros_interfaces.py.
    return fake_interface_messages


class RecordingLogger:

    def __init__(self):
        self.records = []

    def info(self, message, **kwargs):
        self.records.append(('info', message))

    def warning(self, message, **kwargs):
        self.records.append(('warning', message))

    def error(self, message, **kwargs):
        self.records.append(('error', message))


class FakeBridge:
    """Returns a prepared mono8 image, or raises a prepared conversion error."""

    def __init__(self, image=None, error=None):
        self.image = image
        self.error = error
        self.requested_encodings = []

    def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
        self.requested_encodings.append(desired_encoding)
        if self.error is not None:
            raise self.error
        return self.image


class FakePublisher:
    """Records published messages; this is not evidence of DDS delivery."""

    def __init__(self, error=None):
        self.error = error
        self.published = []

    def publish(self, message):
        if self.error is not None:
            raise self.error
        self.published.append(message)


class FakeNode:
    """Runs the real FireDetection methods without rclpy.init() or Gazebo."""

    image_callback = FireDetection.image_callback
    process_thermal_image = FireDetection.process_thermal_image

    def __init__(self, bridge, publisher=None):
        self.bridge = bridge
        self.observation_publisher = publisher if publisher is not None else FakePublisher()
        self.logger = RecordingLogger()

    def get_logger(self):
        return self.logger


def _fake_node(bridge, publisher=None):
    node = FakeNode(bridge, publisher)
    return node, node.logger


def _msg(sec=12, nanosec=345, frame_id='thermal_link', width=320, height=240):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id=frame_id,
        ),
        encoding='mono8',
        width=width,
        height=height,
    )


def _run_callback(node, msg=None):
    return node.image_callback(msg if msg is not None else _msg())


def _process(image, stamp_ns, frame_id):
    # The processing boundary, given an already converted mono8 image.
    node, logger = _fake_node(None)
    return node.process_thermal_image(image, stamp_ns, frame_id), logger


def _image_with_hot_pixels(count, shape=(10, 10)):
    image = np.zeros(shape, dtype=np.uint8)
    image.reshape(-1)[:count] = HOT
    return image


def _legacy_image_callback(self, msg):
    # The detector's callback before M2.2, verbatim apart from formatting.
    image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
    threshold = 180
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    hot_pixels = cv2.countNonZero(mask)
    if hot_pixels > 50:
        self.get_logger().warning(f'FIRE HAZARD DETECTED! Hot pixels: {hot_pixels}')


# --- Processing boundary: converted image in, observation out ---------------

# 1 and 2. A valid image produces an observation with exact measurements.
def test_valid_image_produces_observation_with_expected_regions():
    image = np.zeros((10, 12), dtype=np.uint8)
    image[2:5, 3:7] = HOT
    observation, logger = _process(image, 7, 'thermal_link')
    assert logger.records == []
    assert observation.source_stamp_ns == 7
    assert observation.source_frame_id == 'thermal_link'
    assert isinstance(observation, ThermalFrameObservation)
    assert observation.extraction.threshold == 180
    assert observation.extraction.hot_pixel_count == 12
    assert observation.extraction.regions == (
        ThermalRegion((3, 2, 4, 3), (4.5, 3.0), 12, 200.0, 200, False),
    )


def test_original_detector_parameters_are_unchanged():
    assert fire_detection.HOT_PIXEL_THRESHOLD == 180
    assert fire_detection.MIN_HOT_PIXELS == 50


@pytest.mark.parametrize('sec, nanosec, expected_ns', [
    (12, 345, 12_000_000_345),
    (0, 0, 0),
    (0, 999_999_999, 999_999_999),
    (1_762_600_000, 123_456_789, 1_762_600_000_123_456_789),
])
def test_stamp_to_ns_is_exact(sec, nanosec, expected_ns):
    ns = stamp_to_ns(SimpleNamespace(sec=sec, nanosec=nanosec))
    assert ns == expected_ns
    assert type(ns) is int


@pytest.mark.parametrize('image, stamp_ns, error', [
    (np.zeros((0, 5), dtype=np.uint8), 0, ValueError),
    (np.zeros((4, 4), dtype=np.uint16), 0, TypeError),
    (np.zeros((4, 4), dtype=np.uint8), -1, ValueError),
])
def test_processing_failure_raises_instead_of_returning_empty_observation(
    image, stamp_ns, error,
):
    with pytest.raises(error):
        _process(image, stamp_ns, None)


# --- Callback, with a fake node and bridge -----------------------------------

# 1, 3 and 5. The callback returns the observation with the image's own metadata.
def test_callback_returns_observation_with_source_stamp_and_frame():
    bridge = FakeBridge(_image_with_hot_pixels(3))
    node, logger = _fake_node(bridge)
    observation = _run_callback(node, _msg(sec=12, nanosec=345, frame_id='thermal_link'))
    assert isinstance(observation, ThermalFrameObservation)
    assert observation.source_stamp_ns == 12_000_000_345
    assert observation.source_frame_id == 'thermal_link'
    assert observation.extraction.hot_pixel_count == 3
    assert bridge.requested_encodings == ['mono8']
    assert logger.records == []


# 4. Timestamp zero is a valid acquisition time, not "missing".
def test_callback_preserves_zero_timestamp():
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(0)))
    observation = _run_callback(node, _msg(sec=0, nanosec=0))
    assert observation is not None
    assert observation.source_stamp_ns == 0


# 5. An unset ROS frame ('') becomes None, as the M2.1 contract requires.
def test_callback_maps_empty_frame_id_to_none():
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(0)))
    assert _run_callback(node, _msg(frame_id='')).source_frame_id is None


# 6 and 7. The original whole-image boundary: 50 does not warn, 51 does.
@pytest.mark.parametrize('hot_pixels, expected_records', [
    (50, []),
    (51, [('warning', WARNING_51)]),
])
def test_callback_legacy_alarm_boundary(hot_pixels, expected_records):
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(hot_pixels)))
    observation = _run_callback(node)
    assert observation.extraction.hot_pixel_count == hot_pixels
    assert logger.records == expected_records


# 8. Two separate 30-pixel components still warn: the alarm is not per region.
def test_callback_two_30_pixel_components_still_warn():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[1:4, 1:11] = HOT
    image[10:13, 1:11] = HOT
    node, logger = _fake_node(FakeBridge(image))
    observation = _run_callback(node)
    assert [r.area_px for r in observation.extraction.regions] == [30, 30]
    assert logger.records == [('warning', 'FIRE HAZARD DETECTED! Hot pixels: 60')]


# 9. A cold frame is a successful observation with no regions, not a failure.
def test_callback_cold_frame_is_successful_empty_observation():
    node, logger = _fake_node(FakeBridge(np.full((240, 320), 96, dtype=np.uint8)))
    observation = _run_callback(node)
    assert isinstance(observation, ThermalFrameObservation)
    assert observation.extraction.hot_pixel_count == 0
    assert observation.extraction.regions == ()
    assert logger.records == []


# 10. Conversion and processing failures give no observation and an error log.
@pytest.mark.parametrize('error', [
    CvBridgeError('encoding not supported'),
    TypeError('buffer is too small for requested array'),
    ValueError('cannot reshape array'),
])
def test_callback_conversion_failure_returns_none(error):
    node, logger = _fake_node(FakeBridge(error=error))
    assert _run_callback(node) is None
    assert len(logger.records) == 1
    level, message = logger.records[0]
    assert level == 'error'
    assert 'Thermal frame dropped' in message
    assert str(error) in message
    assert type(error).__name__ in message


def test_callback_empty_image_returns_none():
    node, logger = _fake_node(FakeBridge(np.zeros((0, 320), dtype=np.uint8)))
    assert _run_callback(node) is None
    assert [level for level, _ in logger.records] == ['error']


def test_callback_invalid_stamp_drops_observation_but_keeps_fire_warning():
    # The original warning depends only on the image, so bad metadata must not
    # suppress it; only the observation is dropped.
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(51)))
    assert _run_callback(node, _msg(sec=-1)) is None
    assert logger.records[0] == ('warning', WARNING_51)
    assert [level for level, _ in logger.records] == ['warning', 'error']


def test_callback_propagates_exceptions_outside_the_handled_types():
    # Only ValueError is handled around processing. A TypeError there (e.g. a
    # non-uint8 image after mono8 conversion) and any exception type the
    # conversion handler does not list propagate instead of being dropped.
    node, _ = _fake_node(FakeBridge(np.zeros((4, 4), dtype=np.float32)))
    with pytest.raises(TypeError):
        _run_callback(node)
    node, _ = _fake_node(FakeBridge(error=AttributeError('defect')))
    with pytest.raises(AttributeError):
        _run_callback(node)


def test_callback_value_error_inside_processing_is_logged_not_hidden(monkeypatch):
    # Known limitation: the ValueError handler cannot tell an expected input
    # failure from a ValueError raised by a defect inside processing. Such a
    # defect still yields an error log with its message and no observation;
    # it is never turned into an empty successful observation.
    def defective_extraction(image, threshold):
        raise ValueError('defect inside extraction')

    monkeypatch.setattr(fire_detection, 'extract_thermal_regions', defective_extraction)
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(0)))
    assert _run_callback(node) is None
    assert logger.records == [('error', 'Thermal frame dropped: defect inside extraction')]


# 11. Warning text and per-frame logging are identical to the original callback.
def test_callback_logs_exactly_like_original_detector_frame_by_frame():
    rng = np.random.default_rng(22)
    frames = [_image_with_hot_pixels(n) for n in (0, 50, 51, 51, 100)]
    frames += [rng.integers(0, 256, size=(24, 32), dtype=np.uint8) for _ in range(10)]
    frames += [rng.integers(170, 186, size=(12, 12), dtype=np.uint8) for _ in range(10)]

    new_node, new_logger = _fake_node(None)
    old_node, old_logger = _fake_node(None)
    for image in frames:
        new_node.bridge = FakeBridge(image)
        old_node.bridge = FakeBridge(image)
        assert _run_callback(new_node) is not None
        _legacy_image_callback(old_node, _msg())

    assert new_logger.records == old_logger.records
    warnings = [r for r in new_logger.records if r[0] == 'warning']
    assert warnings[:3] == [('warning', WARNING_51), ('warning', WARNING_51),
                            ('warning', 'FIRE HAZARD DETECTED! Hot pixels: 100')]


# --- Publication (fake publisher: call counting only, not DDS delivery) ------

def _published(node):
    return node.observation_publisher.published


def _stamps(messages):
    return [(m.header.stamp.sec, m.header.stamp.nanosec) for m in messages]


def test_topic_and_qos_constants():
    assert fire_detection.OBSERVATION_TOPIC == 'thermal/observations'
    assert fire_detection.OBSERVATION_QOS_DEPTH == 5


def test_valid_frame_publishes_exactly_once():
    image = np.zeros((10, 12), dtype=np.uint8)
    image[2:5, 3:7] = HOT  # 12 hot pixels: below the legacy warning, still published
    node, logger = _fake_node(FakeBridge(image))
    observation = _run_callback(node, _msg(sec=12, nanosec=345, frame_id='thermal_link'))

    (message,) = _published(node)
    assert _stamps([message]) == [(12, 345)]
    assert message.header.frame_id == 'thermal_link'
    assert message.hot_pixel_count == observation.extraction.hot_pixel_count == 12
    assert [(r.bbox_x, r.bbox_y, r.bbox_width, r.bbox_height) for r in message.regions] == [
        (3, 2, 4, 3),
    ]
    assert logger.records == []


def test_cold_frame_publishes_empty_regions():
    node, logger = _fake_node(FakeBridge(np.full((240, 320), 96, dtype=np.uint8)))
    _run_callback(node)
    (message,) = _published(node)
    assert list(message.regions) == []
    assert message.hot_pixel_count == 0
    assert logger.records == []


@pytest.mark.parametrize('hot_pixels, warned', [(0, False), (50, False), (51, True)])
def test_legacy_warning_does_not_gate_publication(hot_pixels, warned):
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(hot_pixels)))
    _run_callback(node)
    assert len(_published(node)) == 1
    assert (('warning', f'FIRE HAZARD DETECTED! Hot pixels: {hot_pixels}') in logger.records) \
        is warned


def test_repeated_identical_frames_publish_repeatedly():
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(3)))
    msg = _msg()
    _run_callback(node, msg)
    _run_callback(node, msg)
    first, second = _published(node)
    assert first is not second
    assert _stamps([first, second]) == [(12, 345), (12, 345)]


def test_decreasing_and_zero_timestamps_are_published_unchanged():
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(0)))
    for sec in (10, 5, 0):
        _run_callback(node, _msg(sec=sec, nanosec=0))
    assert _stamps(_published(node)) == [(10, 0), (5, 0), (0, 0)]


def test_empty_frame_id_is_published_as_empty_string():
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(0)))
    _run_callback(node, _msg(frame_id=''))
    assert _published(node)[0].header.frame_id == ''


def test_extraction_and_conversion_run_once_per_frame(monkeypatch):
    calls = {'extract': 0, 'convert': 0}

    def counting_extract(image, threshold):
        calls['extract'] += 1
        return extract_thermal_regions(image, threshold=threshold)

    def counting_convert(observation):
        calls['convert'] += 1
        return real_convert(observation)

    real_convert = fire_detection.observation_to_message
    monkeypatch.setattr(fire_detection, 'extract_thermal_regions', counting_extract)
    monkeypatch.setattr(fire_detection, 'observation_to_message', counting_convert)
    node, _ = _fake_node(FakeBridge(_image_with_hot_pixels(60)))
    _run_callback(node)
    assert calls == {'extract': 1, 'convert': 1}
    assert len(_published(node)) == 1


@pytest.mark.parametrize('bridge', [
    FakeBridge(error=CvBridgeError('encoding not supported')),
    FakeBridge(error=TypeError('buffer is too small for requested array')),
    FakeBridge(np.zeros((0, 320), dtype=np.uint8)),
])
def test_processing_failures_publish_nothing(bridge):
    node, logger = _fake_node(bridge)
    assert _run_callback(node) is None
    assert _published(node) == []
    assert [level for level, _ in logger.records] == ['error']


@pytest.mark.parametrize('sec, nanosec', [
    (5, 1_000_000_000),
    (-1, 0),
    (2**31, 0),
    (True, 0),
    (1.5, 0),
])
def test_invalid_source_stamps_publish_nothing_but_keep_the_warning(sec, nanosec):
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(51)))
    assert _run_callback(node, _msg(sec=sec, nanosec=nanosec)) is None
    assert _published(node) == []
    assert logger.records[0] == ('warning', WARNING_51)
    level, message = logger.records[1]
    assert level == 'error'
    assert message.startswith('Thermal frame dropped: invalid source timestamp')
    assert len(logger.records) == 2


def test_converter_validation_failure_publishes_nothing(monkeypatch):
    nan_region = ThermalRegion((0, 0, 1, 1), (0.0, 0.0), 1, float('nan'), 200, True)

    def nan_extraction(image, threshold):
        return ThermalExtractionResult(10, 10, threshold, 1, 8, 1, (nan_region,))

    monkeypatch.setattr(fire_detection, 'extract_thermal_regions', nan_extraction)
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(1)))
    assert _run_callback(node) is None
    assert _published(node) == []
    assert logger.records == [(
        'error',
        'Thermal observation not published: regions[0].mean_intensity=nan is not finite',
    )]


def test_publisher_failure_propagates_and_is_not_logged_as_a_bad_frame():
    publisher = FakePublisher(error=RuntimeError('publisher failed'))
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(3)), publisher)
    with pytest.raises(RuntimeError, match='publisher failed'):
        _run_callback(node)
    assert logger.records == []


def test_unexpected_converter_exception_propagates(monkeypatch):
    def broken_convert(observation):
        raise AttributeError('defect in converter')

    monkeypatch.setattr(fire_detection, 'observation_to_message', broken_convert)
    node, logger = _fake_node(FakeBridge(_image_with_hot_pixels(3)))
    with pytest.raises(AttributeError):
        _run_callback(node)
    assert _published(node) == []
    assert logger.records == []


def test_conversion_error_is_a_value_error_subclass():
    # Lets generic callers catch it as ValueError, while the callback catches
    # only this type, so other ValueErrors from conversion are not hidden.
    assert issubclass(ObservationConversionError, ValueError)


# --- Operational boundary ------------------------------------------------------

def test_detector_has_no_ground_truth_or_second_extraction_path():
    # Discovery against simulator ground truth lives in hazard_discovery; the
    # detector extracts once (see test_extraction_and_conversion_run_once_per_frame).
    for name in ('cv2', '_detect_discovery', 'PoseArray', 'Odometry'):
        assert not hasattr(fire_detection, name) and not hasattr(FireDetection, name), name


# --- Real ROS message conversion (skipped where ROS 2 is not sourced) -------

@pytest.mark.skipif(ROS_STUBBED, reason='requires a sourced ROS 2 environment')
def test_real_cv_bridge_message_round_trip():
    from cv_bridge import CvBridge

    bridge = CvBridge()
    image = _image_with_hot_pixels(51, shape=(240, 320))
    msg = bridge.cv2_to_imgmsg(image, encoding='mono8')
    msg.header.stamp.sec = 3
    msg.header.stamp.nanosec = 4
    msg.header.frame_id = 'thermal_link'

    node, logger = _fake_node(bridge)
    observation = _run_callback(node, msg)
    assert observation.source_stamp_ns == 3_000_000_004
    assert observation.source_frame_id == 'thermal_link'
    assert observation.extraction.hot_pixel_count == 51
    assert logger.records == [('warning', WARNING_51)]

    msg.data = msg.data[: len(msg.data) // 2]  # truncated payload
    assert _run_callback(node, msg) is None
    assert logger.records[-1][0] == 'error'
