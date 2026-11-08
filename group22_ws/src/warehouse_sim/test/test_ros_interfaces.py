import time

from fake_interfaces import introspection_type, read_msg_fields
from hazard_perception.thermal_regions import extract_thermal_regions, ThermalFrameObservation
import numpy as np
import pytest
from ros_stand_ins import any_stand_ins_installed

requires_ros = pytest.mark.skipif(
    any_stand_ins_installed(),
    reason='requires a built and sourced ROS 2 Jazzy workspace',
)

# Isolated from the default domain (0) and from a simulation run on another domain.
DDS_TEST_DOMAIN_ID = 77
DISCOVERY_TIMEOUT_SEC = 10.0
DELIVERY_TIMEOUT_SEC = 10.0
DRAIN_SEC = 0.5

COLD = np.full((24, 32), 96, dtype=np.uint8)
HOT = COLD.copy()
HOT[2:5, 3:7] = 200  # 12 hot pixels: below the legacy warning threshold
SENTINEL_STAMP = (9, 0)


# --- Helpers usable without ROS -------------------------------------------------

@pytest.mark.parametrize('msg_type, expected', [
    ('float64', 'double'),
    ('float32', 'float'),
    ('bool', 'boolean'),
    ('uint8', 'uint8'),
    ('uint32', 'uint32'),
    ('string', 'string'),
    ('std_msgs/Header', 'std_msgs/Header'),
    ('ThermalRegion', 'hazard_interfaces/ThermalRegion'),
    ('ThermalRegion[]', 'sequence<hazard_interfaces/ThermalRegion>'),
    ('float64[]', 'sequence<double>'),
    ('bool[]', 'sequence<boolean>'),
])
def test_introspection_type_spelling(msg_type, expected):
    assert introspection_type(msg_type) == expected


@pytest.mark.parametrize('msg_type', ['uint8[4]', 'float64[<=3]', 'string<=8', 'thermalregion'])
def test_introspection_type_rejects_unsupported_forms(msg_type):
    with pytest.raises(ValueError):
        introspection_type(msg_type)


def test_expected_contract_still_detects_mismatches():
    fields = read_msg_fields('ThermalRegion')
    expected = {name: introspection_type(t) for name, t in fields.items()}
    assert expected['mean_intensity'] == 'double'
    assert expected['touches_image_boundary'] == 'boolean'
    wrong_type = {**expected, 'mean_intensity': 'float'}
    renamed = {**{k: v for k, v in expected.items() if k != 'area_px'}, 'area': 'uint32'}
    assert wrong_type != expected
    assert renamed != expected


def _spin_until(executor, condition, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    while not condition():
        if time.monotonic() >= deadline:
            return False
        executor.spin_once(timeout_sec=0.05)
    return True


def _drain(executor, duration_sec):
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


class _CountingExecutor:

    def __init__(self):
        self.spins = 0

    def spin_once(self, timeout_sec):
        self.spins += 1


def test_spin_until_is_bounded():
    executor = _CountingExecutor()
    started = time.monotonic()
    assert _spin_until(executor, lambda: False, timeout_sec=0.2) is False
    assert time.monotonic() - started < 2.0
    assert executor.spins > 0
    assert _spin_until(_CountingExecutor(), lambda: True, timeout_sec=0.2) is True


# --- Generated messages ---------------------------------------------------------

@requires_ros
@pytest.mark.parametrize('name', ['ThermalRegion', 'ThermalFrameObservation'])
def test_generated_messages_match_the_contract_and_have_type_support(name):
    from hazard_interfaces import msg
    from rclpy.type_support import check_for_type_support

    message_class = getattr(msg, name)
    # Importing a generated message does not load its compiled type support;
    # this does, and raises if it is missing.
    check_for_type_support(message_class)
    expected = {field: introspection_type(t) for field, t in read_msg_fields(name).items()}
    assert message_class.get_fields_and_field_types() == expected


@requires_ros
def test_serialisation_round_trip_preserves_every_field():
    from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
    from rclpy.serialization import deserialize_message, serialize_message
    from warehouse_sim.thermal_observation_ros import MAX_STAMP_NS, observation_to_message

    image = np.zeros((8, 12), dtype=np.uint8)
    image[1, 1:4] = (181, 200, 201)
    image[2, 1] = 255
    image[4:7, 11] = (190, 195, 200)
    for stamp_ns in (0, 12_000_000_345, MAX_STAMP_NS):
        observation = ThermalFrameObservation(
            stamp_ns, 'thermal_link', extract_thermal_regions(image),
        )
        message = observation_to_message(observation)
        restored = deserialize_message(serialize_message(message), ThermalFrameObservationMsg)
        assert restored == message
        assert len(restored.regions) == 2


# --- Real node and DDS delivery -------------------------------------------------

@pytest.fixture
def ros_context(monkeypatch):
    import rclpy

    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    rclpy.init(domain_id=DDS_TEST_DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.try_shutdown()


def _single_endpoint(infos, node_name):
    matches = [i for i in infos if i.node_name == node_name and i.node_namespace == '/']
    return matches[0] if len(matches) == 1 else None


def _describe(infos):
    return [
        f'{i.node_namespace.rstrip("/")}/{i.node_name}'
        f'(reliability={i.qos_profile.reliability.name}, '
        f'durability={i.qos_profile.durability.name})'
        for i in infos
    ] or 'none'


def _stamp(message):
    return (message.header.stamp.sec, message.header.stamp.nanosec)


@requires_ros
def test_detector_publishes_one_observation_per_valid_frame_over_dds(ros_context):
    from cv_bridge import CvBridge
    from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from warehouse_sim.fire_detection import FireDetection

    detector = harness = executor = None
    try:
        detector = FireDetection()
        harness = Node('m23_test_harness')
        received = []
        harness.create_subscription(
            ThermalFrameObservationMsg, '/thermal/observations', received.append,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        image_publisher = harness.create_publisher(Image, '/thermal/image_raw', 10)
        executor = SingleThreadedExecutor()
        executor.add_node(detector)
        executor.add_node(harness)

        # Discovery is asynchronous. Stage 1: wait for the detector's own
        # endpoints (identified by node name) to appear in the graph.
        def detector_endpoints():
            return (
                _single_endpoint(
                    harness.get_publishers_info_by_topic('/thermal/observations'),
                    'fire_detection',
                ),
                _single_endpoint(
                    harness.get_subscriptions_info_by_topic('/thermal/image_raw'),
                    'fire_detection',
                ),
            )

        assert _spin_until(
            executor, lambda: None not in detector_endpoints(), DISCOVERY_TIMEOUT_SEC,
        ), (
            'fire_detection endpoints not discovered within '
            f'{DISCOVERY_TIMEOUT_SEC}s. Publishers on /thermal/observations: '
            f'{_describe(harness.get_publishers_info_by_topic("/thermal/observations"))}; '
            'subscriptions on /thermal/image_raw: '
            f'{_describe(harness.get_subscriptions_info_by_topic("/thermal/image_raw"))}'
        )

        # Stage 2: the discovered publisher endpoint must offer the agreed QoS.
        endpoint, _ = detector_endpoints()
        assert endpoint.topic_type == 'hazard_interfaces/msg/ThermalFrameObservation'
        assert endpoint.qos_profile.reliability == ReliabilityPolicy.RELIABLE, (
            f'observation publisher QoS: {_describe([endpoint])}, expected RELIABLE'
        )
        assert endpoint.qos_profile.durability == DurabilityPolicy.VOLATILE, (
            f'observation publisher QoS: {_describe([endpoint])}, expected VOLATILE'
        )
        profile = detector.observation_publisher.qos_profile
        assert (profile.history, profile.depth) == (HistoryPolicy.KEEP_LAST, 5), profile
        assert detector.observation_publisher.topic_name == '/thermal/observations'

        # Stage 3: both connections must actually match (compatible QoS).
        def matched():
            return (
                image_publisher.get_subscription_count() > 0
                and detector.observation_publisher.get_subscription_count() > 0
            )

        assert _spin_until(executor, matched, DISCOVERY_TIMEOUT_SEC), (
            'endpoints discovered but not connected (incompatible QoS?): '
            f'image publisher matches={image_publisher.get_subscription_count()}, '
            'observation publisher matches='
            f'{detector.observation_publisher.get_subscription_count()}'
        )

        bridge = CvBridge()

        def frame(image, sec, nanosec=0):
            msg = bridge.cv2_to_imgmsg(image, encoding='mono8')
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = 'thermal_link'
            return msg

        truncated = frame(HOT, 3)
        truncated.data = truncated.data[:10]

        # Invalid frames sit between valid ones; a repeated valid frame must
        # publish again; the last frame is the sentinel. Delivery over this one
        # reliable connection is in order, so once the sentinel arrives every
        # earlier observation has arrived too.
        for msg in (
            frame(COLD, 1), frame(HOT, 2), truncated, frame(HOT, 4, 1_000_000_000),
            frame(HOT, -1), frame(COLD, 1), frame(COLD, *SENTINEL_STAMP),
        ):
            image_publisher.publish(msg)
            executor.spin_once(timeout_sec=0.05)

        sentinel_arrived = _spin_until(
            executor, lambda: any(_stamp(m) == SENTINEL_STAMP for m in received),
            DELIVERY_TIMEOUT_SEC,
        )
        assert sentinel_arrived, f'sentinel not received; got {[_stamp(m) for m in received]}'
        _drain(executor, DRAIN_SEC)  # bounded: catches trailing duplicates

        assert [_stamp(m) for m in received] == [(1, 0), (2, 0), (1, 0), SENTINEL_STAMP]
        assert all(m.header.frame_id == 'thermal_link' for m in received)
        cold, hot = received[0], received[1]
        assert list(cold.regions) == []
        assert cold.hot_pixel_count == 0
        assert hot.hot_pixel_count == 12
        assert [(r.bbox_x, r.bbox_y, r.bbox_width, r.bbox_height) for r in hot.regions] == [
            (3, 2, 4, 3),
        ]
    finally:
        if executor is not None:
            executor.shutdown()
        for node in (detector, harness):
            if node is not None:
                node.destroy_node()
