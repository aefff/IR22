import math
import time
from types import SimpleNamespace

import cv2
from hazard_perception.thermal_regions import extract_thermal_regions, ThermalFrameObservation
import numpy as np
import pytest
from ros_stand_ins import any_stand_ins_installed

if any_stand_ins_installed():
    # hazard_discovery imports geometry_msgs, nav_msgs and std_msgs, which have
    # no stand-ins. In strict mode stand-ins are forbidden, so this never skips.
    pytest.skip('requires a built and sourced ROS 2 workspace', allow_module_level=True)

from warehouse_sim import hazard_discovery  # noqa: E402
from warehouse_sim.hazard_discovery import (  # noqa: E402
    discovered_hazard_indices,
    HazardDiscovery,
)
from warehouse_sim.thermal_observation_ros import observation_to_message  # noqa: E402

DDS_TEST_DOMAIN_ID = 77
TIMEOUT_SEC = 10.0
FX = 277.0
WIDTH, HEIGHT = 320, 240


class FakePublisher:

    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message.data)


class RecordingLogger:

    def __init__(self):
        self.records = []

    def warning(self, message, **kwargs):
        self.records.append(('warning', message))


class FakeDiscoveryNode:
    """Runs the real HazardDiscovery callback without rclpy.init()."""

    observation_callback = HazardDiscovery.observation_callback

    def __init__(self, hazards=(), odom=None, camera_info=None):
        self._hazards = list(hazards)
        self._odom = odom
        self._camera_info = camera_info
        self._reported = set()
        self.discovery_publisher = FakePublisher()
        self.logger = RecordingLogger()

    def get_logger(self):
        return self.logger


def _legacy_detect_discovery(self, image):
    # FireDetection._detect_discovery as it was before R0.1, verbatim apart
    # from the hard-coded constants; used only as a reference implementation.
    if self._odom is None or self._camera_info is None or not self._hazards:
        return
    binary = (image > 180).astype('uint8')
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    pose = self._odom
    q = pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    camera_x = pose.position.x + 0.185 * math.cos(yaw) + 0.04 * math.sin(yaw)
    camera_y = pose.position.y + 0.185 * math.sin(yaw) - 0.04 * math.cos(yaw)
    fx = self._camera_info.k[0] if len(self._camera_info.k) >= 1 else 0.0
    width = self._camera_info.width or image.shape[1]
    if fx <= 0.0 or width <= 0:
        return
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        pixel_x = float(centroids[component][0])
        region_bearing = math.atan2((pixel_x - (width - 1) / 2.0), fx)
        candidates = []
        for index, (hx, hy) in enumerate(self._hazards, start=1):
            dx, dy = hx - camera_x, hy - camera_y
            bearing = math.atan2(dy, dx) - yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing - region_bearing) < 0.12:
                candidates.append((math.hypot(dx, dy), index))
        if candidates:
            _, index = min(candidates)
            name = f'fire_hazard_{index:03d}'
            if name not in self._reported:
                self._reported.add(name)
                self.discovery_publisher.publish(SimpleNamespace(data=name))


def _pose(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
    )


def _camera_info(fx=FX, width=WIDTH):
    return SimpleNamespace(k=[fx, 0.0, 159.5, 0.0, fx, 119.5, 0.0, 0.0, 1.0], width=width)


def _observation_message(image, stamp_ns=1_000_000_000):
    extraction = extract_thermal_regions(image, threshold=180)
    return observation_to_message(ThermalFrameObservation(stamp_ns, 'thermal_link', extraction))


def _centred_hot_image(area_rows=6):
    image = np.full((HEIGHT, WIDTH), 96, dtype=np.uint8)
    image[100:100 + area_rows, 155:165] = 200  # 10 px wide, centroid x = 159.5
    return image


AHEAD = [(5.185, -0.04)]  # straight ahead of the lens for a robot at the origin facing +x


# --- Equivalence with the previous in-detector implementation ----------------

def _random_scenario(rng):
    yaw = rng.uniform(-math.pi, math.pi)
    pose = _pose(rng.uniform(-3, 3), rng.uniform(-3, 3), yaw)
    choice = rng.random()
    if choice < 0.7:
        camera_info = _camera_info()
    elif choice < 0.8:
        camera_info = _camera_info(width=0)  # falls back to the image width
    elif choice < 0.9:
        camera_info = _camera_info(width=2 * WIDTH)  # calibration width takes precedence
    elif choice < 0.95:
        camera_info = _camera_info(fx=0.0)  # invalid calibration
    else:
        camera_info = SimpleNamespace(k=[], width=WIDTH)  # missing intrinsics

    image = rng.integers(0, 181, size=(HEIGHT, WIDTH), dtype=np.uint8)  # all <= 180: cold
    image[rng.random(image.shape) < 0.01] = 180  # exactly at threshold: not hot
    for _ in range(rng.integers(0, 5)):
        h, w = int(rng.integers(2, 14)), int(rng.integers(2, 14))  # areas around 50
        r, c = int(rng.integers(0, HEIGHT - h + 1)), int(rng.integers(0, WIDTH - w + 1))
        image[r:r + h, c:c + w] = rng.integers(181, 256)
    for _ in range(rng.integers(0, 30)):  # isolated and diagonal hot specks
        r, c = int(rng.integers(0, HEIGHT - 1)), int(rng.integers(0, WIDTH - 1))
        image[r, c] = image[r + 1, c + 1] = 230

    # Hazards near the bearing each large region would match under the existing
    # sign convention (so matches actually occur), jittered across the
    # tolerance, plus unrelated hazards.
    camera_x = pose.position.x + 0.185 * math.cos(yaw) + 0.04 * math.sin(yaw)
    camera_y = pose.position.y + 0.185 * math.sin(yaw) - 0.04 * math.cos(yaw)
    width = camera_info.width or WIDTH  # the width the algorithm will use
    hazards = []
    for region in extract_thermal_regions(image).regions:
        if region.area_px >= 50:
            b = math.atan2(region.centroid_xy[0] - (width - 1) / 2.0, FX)
            b += rng.uniform(-0.2, 0.2)
            dist = rng.uniform(0.5, 8.0)
            hazards.append((camera_x + dist * math.cos(yaw + b),
                            camera_y + dist * math.sin(yaw + b)))
    for _ in range(rng.integers(0, 3)):
        hazards.append((rng.uniform(-6, 6), rng.uniform(-4, 4)))
    rng.shuffle(hazards)
    return pose, camera_info, hazards, image


def test_discovery_matches_previous_implementation_on_random_scenarios():
    rng = np.random.default_rng(2201)
    events = order_differences = 0
    for _ in range(400):
        pose, camera_info, hazards, image = _random_scenario(rng)
        legacy = FakeDiscoveryNode(hazards, pose, camera_info)
        new = FakeDiscoveryNode(hazards, pose, camera_info)
        for frame in (image, np.flipud(image).copy()):  # two frames: deduplication too
            _legacy_detect_discovery(legacy, frame)
            new.observation_callback(_observation_message(frame))
        assert sorted(new.discovery_publisher.published) == \
            sorted(legacy.discovery_publisher.published)
        assert new._reported == legacy._reported
        order_differences += new.discovery_publisher.published != \
            legacy.discovery_publisher.published
        events += len(new.discovery_publisher.published)
    assert events > 200, 'too few discoveries for the comparison to be meaningful'
    assert order_differences == 0


@pytest.mark.parametrize('rows, discovered', [(4, []), (5, ['fire_hazard_001'])])
def test_area_threshold_is_inclusive_at_50_pixels(rows, discovered):
    # 10 px wide: 4 rows = 40 px, 5 rows = 50 px (the old `area < 50` cut-off).
    node = FakeDiscoveryNode(AHEAD, _pose(), _camera_info())
    node.observation_callback(_observation_message(_centred_hot_image(area_rows=rows)))
    assert node.discovery_publisher.published == discovered


def test_names_are_one_based_spawn_order_and_reported_once():
    hazards = [(-3.0, 2.0), (5.185, -0.04)]  # the second spawned hazard is ahead
    node = FakeDiscoveryNode(hazards, _pose(), _camera_info())
    for stamp in (1, 2, 3):
        node.observation_callback(_observation_message(_centred_hot_image(), stamp))
    assert node.discovery_publisher.published == ['fire_hazard_002']
    assert node.logger.records == [('warning', 'FIRE HAZARD DISCOVERED: fire_hazard_002')]


def test_nearest_of_several_bearing_matches_is_chosen():
    hazards = [(8.185, -0.04), (2.185, -0.04), (5.185, -0.04)]
    node = FakeDiscoveryNode(hazards, _pose(), _camera_info())
    node.observation_callback(_observation_message(_centred_hot_image()))
    assert node.discovery_publisher.published == ['fire_hazard_002']


# --- Missing, empty and invalid inputs ---------------------------------------

@pytest.mark.parametrize('hazards, odom, camera_info', [
    ([], _pose(), _camera_info()),
    (AHEAD, None, _camera_info()),
    (AHEAD, _pose(), None),
])
def test_missing_inputs_publish_nothing(hazards, odom, camera_info):
    node = FakeDiscoveryNode(hazards, odom, camera_info)
    node.observation_callback(_observation_message(_centred_hot_image()))
    assert node.discovery_publisher.published == []


@pytest.mark.parametrize('camera_info', [
    _camera_info(fx=0.0),
    _camera_info(fx=-1.0),
    SimpleNamespace(k=[], width=WIDTH),
])
def test_invalid_calibration_publishes_nothing(camera_info):
    node = FakeDiscoveryNode(AHEAD, _pose(), camera_info)
    node.observation_callback(_observation_message(_centred_hot_image()))
    assert node.discovery_publisher.published == []


def test_zero_calibration_width_falls_back_to_observation_width():
    node = FakeDiscoveryNode(AHEAD, _pose(), _camera_info(width=0))
    node.observation_callback(_observation_message(_centred_hot_image()))
    assert node.discovery_publisher.published == ['fire_hazard_001']


def test_zero_width_everywhere_publishes_nothing():
    observation = SimpleNamespace(image_width=0, regions=[
        SimpleNamespace(area_px=60, centroid_x=0.0)])
    assert discovered_hazard_indices(observation, _camera_info(width=0), _pose(), AHEAD) == []


def test_cold_and_empty_observations_publish_nothing():
    node = FakeDiscoveryNode(AHEAD, _pose(), _camera_info())
    node.observation_callback(_observation_message(np.full((HEIGHT, WIDTH), 96, np.uint8)))
    assert node.discovery_publisher.published == []


# --- No repeated connected-component extraction ------------------------------

def test_discovery_uses_published_regions_without_extracting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('discovery must not extract regions itself')

    message = _observation_message(_centred_hot_image())
    monkeypatch.setattr(cv2, 'connectedComponentsWithStats', forbidden)
    monkeypatch.setattr(cv2, 'connectedComponents', forbidden)
    node = FakeDiscoveryNode(AHEAD, _pose(), _camera_info())
    node.observation_callback(message)
    assert node.discovery_publisher.published == ['fire_hazard_001']
    assert not hasattr(hazard_discovery, 'cv2')
    assert not hasattr(hazard_discovery, 'extract_thermal_regions')


# --- Real ROS --------------------------------------------------------------------

@pytest.fixture
def ros_context(monkeypatch):
    import rclpy

    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    rclpy.init(domain_id=DDS_TEST_DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.try_shutdown()


def _spin_until(executor, condition, timeout_sec=TIMEOUT_SEC):
    deadline = time.monotonic() + timeout_sec
    while not condition():
        if time.monotonic() >= deadline:
            return False
        executor.spin_once(timeout_sec=0.05)
    return True


def _drain(executor, duration_sec=0.5):
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


def _ground_truth_messages():
    from geometry_msgs.msg import Pose, PoseArray
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo

    hazards = PoseArray()
    hazards.header.frame_id = 'world'
    for x, y in [(-3.0, 2.0), (5.185, -0.04)]:
        pose = Pose()
        pose.position.x, pose.position.y = x, y
        pose.orientation.w = 1.0
        hazards.poses.append(pose)
    odom = Odometry()
    odom.pose.pose.orientation.w = 1.0
    info = CameraInfo()
    info.width, info.height = WIDTH, HEIGHT
    info.k = [FX, 0.0, 159.5, 0.0, FX, 119.5, 0.0, 0.0, 1.0]
    return hazards, odom, info


@pytest.mark.parametrize('with_detector', [False, True], ids=['observations', 'end_to_end'])
def test_discovery_over_dds(ros_context, with_detector):
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseArray
    from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from warehouse_sim.fire_detection import FireDetection

    nodes, executor = [], None
    try:
        discovery = HazardDiscovery()
        harness = Node('r01_test_harness')
        nodes += [discovery, harness]
        detector = FireDetection() if with_detector else None
        if detector is not None:
            nodes.append(detector)
        received = []
        harness.create_subscription(String, '/hazards/discovered',
                                    lambda m: received.append(m.data), 10)
        pubs = {
            'poses': harness.create_publisher(PoseArray, '/hazards/poses', 10),
            'odom': harness.create_publisher(Odometry, '/odom', 10),
            'info': harness.create_publisher(CameraInfo, '/thermal/camera_info', 10),
            'obs': harness.create_publisher(
                ThermalFrameObservationMsg, '/thermal/observations', 10),
            'image': harness.create_publisher(Image, '/thermal/image_raw', 10),
        }
        executor = SingleThreadedExecutor()
        for node in nodes:
            executor.add_node(node)

        needed = ['poses', 'odom', 'info', 'image' if with_detector else 'obs']

        def ready():
            # End to end, the detector's observation publisher must also have
            # matched the discovery subscription (the only subscriber to it).
            return (all(pubs[k].get_subscription_count() > 0 for k in needed)
                    and discovery.discovery_publisher.get_subscription_count() > 0
                    and (detector is None
                         or detector.observation_publisher.get_subscription_count() > 0))

        assert _spin_until(executor, ready), {
            **{k: pubs[k].get_subscription_count() for k in needed},
            'detector->discovery': (detector.observation_publisher.get_subscription_count()
                                    if detector is not None else 'n/a'),
        }

        hazards, odom, info = _ground_truth_messages()
        pubs['poses'].publish(hazards)
        pubs['odom'].publish(odom)
        pubs['info'].publish(info)
        assert _spin_until(executor, lambda: discovery._hazards and discovery._odom is not None
                           and discovery._camera_info is not None), 'ground truth not received'

        for stamp in (1, 2):  # the same hazard seen twice is reported once
            if with_detector:
                msg = CvBridge().cv2_to_imgmsg(_centred_hot_image(), encoding='mono8')
                msg.header.stamp.sec = stamp
                msg.header.frame_id = 'thermal_link'
                pubs['image'].publish(msg)
            else:
                pubs['obs'].publish(_observation_message(_centred_hot_image(), stamp * 10**9))
        assert _spin_until(executor, lambda: received), 'no discovery event received'
        _drain(executor)
        assert received == ['fire_hazard_002']
    finally:
        if executor is not None:
            executor.shutdown()
        for node in nodes:
            node.destroy_node()
