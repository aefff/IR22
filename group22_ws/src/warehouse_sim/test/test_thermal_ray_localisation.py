import math
import time

import numpy as np
import pytest
from ros_stand_ins import any_stand_ins_installed

if any_stand_ins_installed():
    # The node needs tf2_ros and generated messages, which have no stand-ins.
    # Strict mode forbids stand-ins, so this never skips there.
    pytest.skip('requires a built and sourced ROS 2 workspace', allow_module_level=True)

# ROS imports follow the skip on purpose (they fail without ROS).
from geometry_msgs.msg import TransformStamped  # noqa: E402,I100
from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg  # noqa: E402,E501
from hazard_interfaces.msg import ThermalRayFrame  # noqa: E402
from hazard_perception.camera_geometry import CameraGeometryError, horizontal_bearing  # noqa: E402
from hazard_perception.thermal_regions import (  # noqa: E402
    extract_thermal_regions,
    ThermalFrameObservation,
)
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402
import tf2_ros  # noqa: E402
from warehouse_sim import thermal_ray_localisation  # noqa: E402
from warehouse_sim.thermal_observation_ros import observation_to_message  # noqa: E402
from warehouse_sim.thermal_ray_localisation import (  # noqa: E402
    calibration_from_camera_info,
    rays_to_message,
    ThermalRayLocalisation,
)

DOMAIN_ID = 77
FX = 160 / math.tan(1.047 / 2)
GAZEBO_K = [FX, 0.0, 160.0, 0.0, FX, 120.0, 0.0, 0.0, 1.0]
MOUNT = (0.185, -0.04, 0.13)
WAIT_SEC = 5.0


def camera_info(width=320, height=240, frame='thermal_link', model='plumb_bob', d=None,
                k=None):
    msg = CameraInfo()
    msg.header.frame_id = frame
    msg.width, msg.height = width, height
    msg.distortion_model = model
    msg.d = [0.0] * 5 if d is None else d
    msg.k = GAZEBO_K if k is None else k
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    kk = list(msg.k)
    msg.p = kk[0:3] + [0.0] + kk[3:6] + [0.0] + kk[6:9] + [0.0]
    return msg


def image_with(*blocks):
    image = np.full((240, 320), 96, dtype=np.uint8)
    for rows, cols in blocks:
        image[rows, cols] = 200
    return image


CENTRED = (slice(100, 140), slice(155, 165))  # centroid (159.5, 119.5): on the optical axis
LEFT_EDGE = (slice(10, 16), slice(0, 4))  # touches the left edge, upper left


def observation_msg(stamp_ns, image=None, frame='thermal_link'):
    image = image_with(CENTRED) if image is None else image
    return observation_to_message(
        ThermalFrameObservation(stamp_ns, frame, extract_thermal_regions(image)))


def transform(sec, yaw=0.0, translation=MOUNT, nanosec=0, parent='base_link',
              child='thermal_link'):
    t = TransformStamped()
    t.header.stamp.sec, t.header.stamp.nanosec = sec, nanosec
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = translation
    t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return t


# --- Pure conversion helpers -------------------------------------------------------------

def test_camera_info_becomes_raw_calibration_unchanged():
    raw = calibration_from_camera_info(camera_info())
    assert raw.frame_id == 'thermal_link' and (raw.width, raw.height) == (320, 240)
    assert raw.k == tuple(GAZEBO_K)  # published principal point (160, 120), not corrected
    assert raw.distortion_model == 'plumb_bob' and raw.d == (0.0,) * 5


def _modified(**changes):
    msg = camera_info()
    for name, value in changes.items():
        target = msg
        *parents, attr = name.split('.')
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, attr, value)
    return msg


@pytest.mark.parametrize('changes', [
    {'r': [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]},  # rectification
    {'p': [FX, 0.0, 160.0, -10.0, 0.0, FX, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]},  # stereo Tx
    {'p': [FX, 0.0, 159.5, 0.0, 0.0, FX, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]},  # P != K
    {'binning_x': 2}, {'binning_y': 4},
    {'roi.x_offset': 10, 'roi.width': 100, 'roi.height': 100},
    {'roi.do_rectify': True},
])
def test_unsupported_camera_info_configurations_are_rejected(changes):
    with pytest.raises(CameraGeometryError):
        calibration_from_camera_info(_modified(**changes))


def test_ray_frame_message_preserves_stamp_and_serialises():
    from hazard_perception.camera_geometry import (
        FRAME_CONVENTION_BODY, intrinsics_from_calibration, rotation_from_quaternion)
    from hazard_perception.thermal_rays import observation_to_rays
    from warehouse_sim.thermal_observation_ros import message_to_observation

    source = observation_msg(12_000_000_345, image_with(CENTRED, LEFT_EDGE))
    intrinsics = intrinsics_from_calibration(
        calibration_from_camera_info(camera_info()), principal_point_shift_px=-0.5,
        frame_convention=FRAME_CONVENTION_BODY)
    rays = observation_to_rays(message_to_observation(source), intrinsics, frame_id='base_link',
                               translation=MOUNT, rotation=rotation_from_quaternion(0, 0, 0, 1))
    message = rays_to_message(source.header.stamp, 'base_link', rays)
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (12, 345)
    assert message.header.frame_id == 'base_link' and message.source_frame_id == 'thermal_link'
    assert [r.region_index for r in message.rays] == [0, 1]
    assert [r.touches_image_boundary for r in message.rays] == [True, False]
    restored = deserialize_message(serialize_message(message), ThermalRayFrame)
    assert restored == message


def test_localisation_does_not_extract_or_use_ground_truth():
    for name in ('cv2', 'extract_thermal_regions', 'PoseArray', 'Odometry', 'String'):
        assert not hasattr(thermal_ray_localisation, name), name


# --- Real ROS node -----------------------------------------------------------------------

@pytest.fixture
def ros(monkeypatch):
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    rclpy.init(domain_id=DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.try_shutdown()


class Rig:
    """The node under test plus a harness node publishing its inputs."""

    def __init__(self, shift=-0.5, timeout=0.3, sim_time=False, **extra):
        overrides = [Parameter('principal_point_shift_px', value=shift),
                     Parameter('camera_frame_convention', value='rep103_body'),
                     Parameter('transform_timeout_sec', value=timeout)]
        overrides += [Parameter(name, value=value) for name, value in extra.items()]
        if sim_time:
            overrides.append(Parameter('use_sim_time', value=True))
        self.node = ThermalRayLocalisation(parameter_overrides=overrides)
        self.harness = Node('r3_test_harness')
        self.received = []
        self.harness.create_subscription(ThermalRayFrame, '/thermal/rays',
                                         self.received.append, 10)
        self.observations = self.harness.create_publisher(
            ThermalFrameObservationMsg, '/thermal/observations', 10)
        self.camera_info = self.harness.create_publisher(CameraInfo, '/thermal/camera_info', 10)
        self.clock = self.harness.create_publisher(Clock, '/clock', 10)
        self.static_tf = tf2_ros.StaticTransformBroadcaster(self.harness)
        self.dynamic_tf = tf2_ros.TransformBroadcaster(self.harness)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.harness)
        assert self.spin_until(lambda: (
            self.observations.get_subscription_count() > 0
            and self.camera_info.get_subscription_count() > 0
            and self.node.publisher.get_subscription_count() > 0)), 'endpoints did not match'

    def spin_until(self, condition, timeout=WAIT_SEC):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                return False
            self.executor.spin_once(timeout_sec=0.02)
        return True

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def calibrate(self, msg=None):
        msg = msg or camera_info()
        expected = calibration_from_camera_info(msg)
        self.camera_info.publish(msg)
        assert self.spin_until(lambda: self.node.intrinsics is not None
                               and self.node.raw_calibration == expected), 'no calibration'

    def static_mount(self, yaw=0.0, translation=MOUNT):
        self.static_tf.sendTransform(transform(0, yaw, translation))
        assert self.spin_until(lambda: self.node.tf_buffer.can_transform(
            'base_link', 'thermal_link', rclpy.time.Time())), 'static transform not received'

    def send(self, *messages):
        for message in messages:
            self.observations.publish(message)
            self.executor.spin_once(timeout_sec=0.02)

    def set_clock(self, sec, nanosec=0):
        message = Clock()
        message.clock.sec, message.clock.nanosec = sec, nanosec
        self.clock.publish(message)
        target = sec * 1_000_000_000 + nanosec
        assert self.spin_until(lambda: self.node.get_clock().now().nanoseconds == target), \
            'simulated clock not received'

    def pending_stamps(self):
        return [entry[1].source_stamp_ns for entry in self.node.pending]

    def stamps(self):
        return [(m.header.stamp.sec, m.header.stamp.nanosec) for m in self.received]

    def wait_for(self, count):
        return self.spin_until(lambda: len(self.received) >= count)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.harness.destroy_node()


@pytest.fixture
def rig(ros):
    rigs = []

    def make(**kwargs):
        rigs.append(Rig(**kwargs))
        return rigs[-1]

    yield make
    for r in rigs:
        r.close()


def test_rays_keep_stamp_frames_metadata_origin_and_direction(rig):
    r = rig()
    r.calibrate()
    r.static_mount()
    r.send(observation_msg(12_000_000_345, image_with(CENTRED, LEFT_EDGE)))
    assert r.wait_for(1)
    (frame,) = r.received
    assert (frame.header.stamp.sec, frame.header.stamp.nanosec) == (12, 345)
    assert frame.header.frame_id == 'base_link' and frame.source_frame_id == 'thermal_link'
    assert (frame.image_width, frame.image_height) == (320, 240)
    edge, centre = frame.rays
    assert (edge.region_index, centre.region_index) == (0, 1)
    assert (edge.touches_image_boundary, centre.touches_image_boundary) == (True, False)
    assert (centre.centroid_x, centre.centroid_y) == (159.5, 119.5)
    for ray in frame.rays:
        assert (ray.origin.x, ray.origin.y, ray.origin.z) == pytest.approx(MOUNT)
        norm = math.sqrt(ray.direction.x ** 2 + ray.direction.y ** 2 + ray.direction.z ** 2)
        assert norm == pytest.approx(1.0, abs=1e-12)
    # Corrected principal point: the axis centroid maps exactly forward.
    assert (centre.direction.x, centre.direction.y, centre.direction.z) == \
        pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert edge.direction.y > 0 and edge.direction.z > 0  # image upper left = left and up


def test_correction_is_applied_once_and_only_from_the_parameter(rig):
    corrected = rig()
    corrected.calibrate()
    assert (corrected.node.intrinsics.cx, corrected.node.intrinsics.cy) == (159.5, 119.5)
    assert corrected.node.raw_calibration.k[2] == 160.0  # raw value kept
    corrected.close()
    uncorrected = Rig(shift=0.0)
    try:
        uncorrected.calibrate()
        uncorrected.static_mount()
        uncorrected.send(observation_msg(1_000_000_000))
        assert uncorrected.wait_for(1)
        d = uncorrected.received[0].rays[0].direction
        assert horizontal_bearing((d.x, d.y, d.z)) == pytest.approx(math.atan(0.5 / FX), rel=1e-12)
    finally:
        uncorrected.close()


def test_valid_empty_frame_versus_rejected_frames(rig):
    r = rig()
    r.calibrate()
    r.static_mount()
    bad_geometry = observation_msg(2_000_000_000)
    bad_geometry.regions[0].centroid_x = 300.0  # outside its bbox
    bad_content = observation_msg(3_000_000_000)
    bad_content.connectivity = 6
    wrong_frame = observation_msg(4_000_000_000, frame='camera_link')
    r.send(observation_msg(1_000_000_000, image_with()), bad_geometry, bad_content, wrong_frame,
           observation_msg(5_000_000_000))  # the last one is a sentinel
    assert r.spin_until(lambda: (5, 0) in r.stamps())
    r.spin_for(0.3)
    assert r.stamps() == [(1, 0), (5, 0)]
    assert list(r.received[0].rays) == []  # valid frame, no regions
    assert r.node.rejections == {'geometry': 1, 'observation': 1, 'frame': 1}


def test_inconsistent_hot_pixel_count_never_becomes_an_empty_ray_frame(rig):
    r = rig()
    r.calibrate()
    r.static_mount()
    malformed = observation_msg(2_000_000_000, image_with())  # min_area_px=1, no regions
    malformed.hot_pixel_count = 1
    r.send(malformed, observation_msg(3_000_000_000))  # the second one is a sentinel
    assert r.spin_until(lambda: (3, 0) in r.stamps())
    r.spin_for(0.3)
    assert r.stamps() == [(3, 0)] and r.node.rejections == {'observation': 1}


def test_calibration_must_arrive_first_and_identical_repeats_are_accepted(rig):
    r = rig()
    r.static_mount()
    r.send(observation_msg(1_000_000_000))  # before any CameraInfo: rejected, not queued
    r.spin_for(0.2)
    r.camera_info.publish(camera_info(model='equidistant', d=[0.0] * 4))  # invalid first
    r.camera_info.publish(camera_info(frame='camera_link'))  # another camera
    r.spin_for(0.2)
    assert r.node.intrinsics is None and not r.node.calibration_locked
    r.calibrate()
    for _ in range(3):
        r.camera_info.publish(camera_info())  # identical repeats
    r.spin_for(0.2)
    r.send(observation_msg(2_000_000_000))
    assert r.wait_for(1)
    assert r.stamps() == [(2, 0)] and not r.node.calibration_locked
    assert r.node.rejections == {'calibration': 1, 'camera_info': 2}


@pytest.mark.parametrize('second', [
    camera_info(k=[FX, 0.0, 150.0, 0.0, FX, 120.0, 0.0, 0.0, 1.0]),  # different calibration
    camera_info(model='equidistant', d=[0.0] * 4),  # invalid
    camera_info(frame='camera_link'),  # another camera on the topic
])
def test_r31_2_calibration_change_while_queued_fails_closed(rig, second):
    r = rig(timeout=5.0)
    r.calibrate()  # calibration A
    r.send(observation_msg(40_000_000_000))  # queued: no transform yet
    r.spin_for(0.1)
    assert r.pending_stamps() == [40_000_000_000]
    r.camera_info.publish(second)  # calibration B
    assert r.spin_until(lambda: r.node.calibration_locked)
    assert r.pending_stamps() == [] and r.node.intrinsics is None
    r.dynamic_tf.sendTransform(transform(39))
    r.dynamic_tf.sendTransform(transform(41))  # the delayed transform now exists
    r.spin_for(0.3)
    r.camera_info.publish(camera_info())  # even calibration A again: restart required
    r.send(observation_msg(40_500_000_000))
    r.spin_for(0.3)
    assert r.received == []
    assert r.node.calibration_locked
    assert r.node.rejections == {'calibration_changed': 1, 'calibration_locked': 1}


def test_incompatible_first_calibration_rejects_observations(rig):
    r = rig()
    r.calibrate(camera_info(width=160, height=120,
                            k=[FX / 2, 0.0, 80.0, 0.0, FX / 2, 60.0, 0.0, 0.0, 1.0]))
    r.static_mount()
    r.send(observation_msg(1_000_000_000))  # 320 x 240 vs 160 x 120
    r.spin_for(0.2)
    assert r.received == [] and r.node.rejections == {'geometry': 1}


def test_non_identity_mount_rotates_direction_and_keeps_origin(rig):
    r = rig()
    r.calibrate()
    r.static_mount(yaw=math.pi / 2, translation=(1.0, 2.0, 3.0))
    r.send(observation_msg(1_000_000_000))
    assert r.wait_for(1)
    ray = r.received[0].rays[0]
    assert (ray.origin.x, ray.origin.y, ray.origin.z) == pytest.approx((1.0, 2.0, 3.0))
    assert (ray.direction.x, ray.direction.y, ray.direction.z) == \
        pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_transform_is_looked_up_at_the_observation_time_not_the_latest(rig):
    r = rig()
    r.calibrate()
    for sec, yaw in ((10, 0.0), (20, math.pi / 2), (30, math.pi / 2)):
        r.dynamic_tf.sendTransform(transform(sec, yaw))
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=30)))
    r.send(observation_msg(10_000_000_000), observation_msg(15_000_000_000))
    assert r.wait_for(2)
    first, second = (f.rays[0].direction for f in r.received)
    assert (first.x, first.y) == pytest.approx((1.0, 0.0), abs=1e-9)  # not the latest 90 deg
    assert horizontal_bearing((second.x, second.y, second.z)) == pytest.approx(math.pi / 4)


def test_delayed_transform_arriving_before_the_deadline_is_used(rig):
    r = rig(timeout=0.4)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(30))
    r.send(observation_msg(40_000_000_000))  # transform for t=40 s not available yet
    r.spin_for(0.15)
    assert r.received == [] and r.pending_stamps() == [40_000_000_000]
    r.dynamic_tf.sendTransform(transform(41))  # arrives within the deadline
    assert r.wait_for(1)
    assert r.stamps() == [(40, 0)] and r.node.rejections == {}


def test_r31_4_transform_arriving_after_the_deadline_is_not_used(rig):
    r = rig(timeout=0.2)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(30))
    r.send(observation_msg(40_000_000_000))
    assert r.spin_until(lambda: r.pending_stamps() == [40_000_000_000], timeout=0.15)
    time.sleep(0.4)  # not spinning: the deadline passes
    r.node.tf_buffer.set_transform(transform(41), 'test')  # as if TF were delivered first
    r.node.drain()  # the timer callback
    r.spin_for(0.3)
    assert r.received == []
    assert r.node.rejections == {'expired': 1} and r.pending_stamps() == []


def test_fifo_continues_after_an_expired_entry(rig):
    r = rig(timeout=0.4)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(30))
    r.dynamic_tf.sendTransform(transform(41))
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=41)))
    r.send(observation_msg(41_400_000_000))  # beyond the newest transform: stuck
    r.spin_for(0.2)
    r.send(observation_msg(41_000_000_000))  # available, but waits behind the stuck one
    r.spin_for(0.1)
    assert r.received == [] and len(r.node.pending) == 2
    assert r.wait_for(1)  # the stuck entry expires; the next one is processed at once
    assert r.stamps() == [(41, 0)]
    assert r.node.rejections == {'expired': 1}


def test_entries_blocked_until_their_own_deadline_also_expire(rig):
    r = rig(timeout=0.3)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(41))
    r.send(observation_msg(41_400_000_000), observation_msg(41_300_000_000))  # both stuck
    r.spin_for(0.6)
    assert r.received == [] and r.pending_stamps() == []
    assert r.node.rejections == {'expired': 2}


def test_bounded_queue_drops_the_oldest(rig):
    r = rig(timeout=5.0, max_pending_observations=2)
    r.calibrate()
    r.send(*(observation_msg(s * 100_000_000) for s in (1, 2, 3)))  # no transform at all
    r.spin_for(0.2)
    assert r.pending_stamps() == [200_000_000, 300_000_000]
    assert r.node.rejections == {'queue_full': 1}


@pytest.mark.parametrize('static_tfs, dynamic_tfs, published, rejections', [
    # Static-only direct mount: accepted, stamp 0 preserved.
    ([transform(0)], [], [(0, 0)], {}),
    # Static-only two-hop chain: accepted.
    ([transform(0, parent='base_link', child='mount_link', translation=(0.1, 0.0, 0.0)),
      transform(0, parent='mount_link', child='thermal_link', translation=(0.085, -0.04, 0.13))],
     [], [(0, 0)], {}),
    # R31-1: a DYNAMIC transform stamped zero, which tf2 returns for Time(0).
    ([], [transform(0)], [], {'zero_stamp_unverified': 1}),
    # Dynamic transforms at other times: Time(0) would return the latest.
    ([], [transform(10), transform(20)], [], {'zero_stamp_unverified': 1}),
    # Mixed chain: static base_link -> mount_link, dynamic mount_link -> thermal_link.
    ([transform(0, parent='base_link', child='mount_link', translation=(0.1, 0.0, 0.0))],
     [transform(0, parent='mount_link', child='thermal_link', translation=(0.085, -0.04, 0.13))],
     [], {'zero_stamp_unverified': 1}),
])
def test_r31_1_zero_stamp_requires_static_provenance(rig, static_tfs, dynamic_tfs, published,
                                                     rejections):
    r = rig()
    r.calibrate()
    if static_tfs:
        r.static_tf.sendTransform(static_tfs)
    for t in dynamic_tfs:
        r.dynamic_tf.sendTransform(t)
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time()))
    r.spin_for(0.1)
    r.send(observation_msg(0))
    r.spin_for(0.3)
    assert r.stamps() == published
    assert r.node.rejections == rejections
    if published:
        ray = r.received[0].rays[0]
        assert (ray.origin.x, ray.origin.y, ray.origin.z) == pytest.approx(MOUNT)


def test_r31_3_clock_reset_discards_pending_and_dynamic_transforms(rig):
    r = rig(timeout=5.0, sim_time=True)
    r.set_clock(100)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(98))
    r.dynamic_tf.sendTransform(transform(99))
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=99)))
    r.send(observation_msg(99_800_000_000), observation_msg(100_000_000_000))  # both wait
    r.spin_for(0.1)
    assert len(r.node.pending) == 2
    r.set_clock(5)  # simulation reset: the ROS clock jumps back 95 s
    assert r.node.clock_resets == 1 and r.pending_stamps() == []
    assert r.node.rejections == {'clock_reset': 2}
    assert not r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=99))  # dynamic data discarded
    r.dynamic_tf.sendTransform(transform(99))  # stale pre-reset transform data is late...
    r.send(observation_msg(99_900_000_000))  # ...and so is a pre-reset observation
    r.spin_for(0.3)
    assert r.received == [] and r.node.rejections == {'clock_reset': 2, 'future_stamp': 1}


def test_r31_3_after_a_clock_reset_new_observations_publish_with_the_static_mount(rig):
    r = rig(sim_time=True)
    r.set_clock(100)
    r.calibrate()
    r.static_mount()
    r.send(observation_msg(100_000_000_000))
    assert r.wait_for(1)
    r.set_clock(5)
    r.send(observation_msg(5_000_000_000))
    assert r.wait_for(2)
    assert r.stamps() == [(100, 0), (5, 0)] and r.node.clock_resets == 1


def test_r31_3_small_backward_clock_jump_is_not_a_reset(rig):
    r = rig(timeout=5.0, sim_time=True)
    r.set_clock(100)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(98))
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=98)))
    r.send(observation_msg(100_000_000_000))  # waits
    r.spin_for(0.1)
    r.set_clock(99, 700_000_000)  # 0.3 s back: below the 1 s reset threshold
    assert r.node.clock_resets == 0 and r.pending_stamps() == [100_000_000_000]
    assert r.node.tf_buffer.can_transform('base_link', 'thermal_link',
                                          rclpy.time.Time(seconds=98))


def test_r31_3_delayed_observation_without_a_clock_reset(rig):
    r = rig(timeout=5.0)
    r.calibrate()
    r.dynamic_tf.sendTransform(transform(98))
    r.dynamic_tf.sendTransform(transform(102))
    assert r.spin_until(lambda: r.node.tf_buffer.can_transform(
        'base_link', 'thermal_link', rclpy.time.Time(seconds=102)))
    r.send(observation_msg(100_000_000_000))
    assert r.wait_for(1)
    r.send(observation_msg(98_500_000_000),  # 1.5 s late: beyond the 0.5 s tolerance
           observation_msg(99_600_000_000))  # 0.4 s late: within it
    assert r.wait_for(2)
    r.spin_for(0.2)
    assert r.stamps() == [(100, 0), (99, 600_000_000)]
    assert r.node.rejections == {'out_of_order': 1} and r.node.clock_resets == 0
    assert r.node.tf_buffer.can_transform(  # TF untouched by the late observation
        'base_link', 'thermal_link', rclpy.time.Time(seconds=99))


def test_topics_qos_and_no_ground_truth(rig):
    r = rig()
    subscriptions = {s.topic_name for s in r.node.subscriptions}
    publishers = {p.topic_name for p in r.node.publishers} - {'/parameter_events'}
    assert subscriptions == {'/thermal/observations', '/thermal/camera_info', '/tf', '/tf_static'}
    assert publishers == {'/thermal/rays'}
    profile = r.node.publisher.qos_profile
    assert profile.reliability == ReliabilityPolicy.RELIABLE
    assert profile.durability == DurabilityPolicy.VOLATILE and profile.depth == 5
    assert r.spin_until(lambda: r.harness.get_publishers_info_by_topic('/thermal/rays'))


@pytest.mark.parametrize('overrides, message', [
    ({'camera_frame_convention': 'rep103_body'}, 'principal_point_shift_px'),
    ({'principal_point_shift_px': -0.5}, 'camera_frame_convention'),
    ({'principal_point_shift_px': 2.0, 'camera_frame_convention': 'rep103_body'}, 'plausible'),
    ({'principal_point_shift_px': -0.5, 'camera_frame_convention': 'thermal_link'}, 'one of'),
    ({'principal_point_shift_px': -0.5, 'camera_frame_convention': 'rep103_body',
      'transform_timeout_sec': -1.0}, 'transform_timeout_sec'),
])
def test_invalid_or_missing_parameters_stop_the_node(ros, overrides, message):
    with pytest.raises(ValueError, match=message):
        ThermalRayLocalisation(parameter_overrides=[
            Parameter(name, value=value) for name, value in overrides.items()])
