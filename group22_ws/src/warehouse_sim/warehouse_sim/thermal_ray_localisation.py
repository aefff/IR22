from collections import deque
import math
import time

from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
from hazard_interfaces.msg import ThermalRayFrame, ThermalRegionRay
from hazard_perception.camera_geometry import (
    CameraGeometryError,
    FRAME_CONVENTIONS,
    intrinsics_from_calibration,
    MAX_PRINCIPAL_POINT_SHIFT_PX,
    RawCameraCalibration,
    rotation_from_quaternion,
)
from hazard_perception.thermal_rays import observation_to_rays
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.clock import Clock, ClockChange, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
import tf2_ros
from warehouse_sim.thermal_observation_ros import (
    message_to_observation,
    NS_PER_SECOND,
    ObservationConversionError,
)

# Turns /thermal/observations into camera rays in target_frame, using only
# sensor data: the observation, /thermal/camera_info and TF. No simulator
# ground truth, no range, map position or identity.
#
# Policies:
# - Calibration session: the first valid CameraInfo is frozen for the life of
#   the node; identical repeats are ignored. Any different (or invalid)
#   CameraInfo afterwards fails closed: pending observations are discarded and
#   no rays are published again until the node is restarted. Observations
#   arriving before the first valid CameraInfo are rejected, not queued.
# - TF, non-zero stamps: looked up at the observation's exact stamp; the
#   returned stamp must equal it.
# - TF, zero stamp: tf2 reads time zero as "latest", so a zero-stamped
#   observation uses only transforms received on /tf_static (a separate buffer
#   fed exclusively from that topic). If the camera can only be reached
#   through dynamic transforms, the observation is rejected.
# - Clock resets: a ROS-clock backward jump of at least
#   clock_reset_threshold_sec (or a clock-type change) discards pending
#   observations and dynamic TF data; static transforms are kept.
# - Out-of-order and future stamps: an observation more than
#   stamp_tolerance_sec older than the newest one accepted, or newer than the
#   node's clock by more than that, is rejected (a stale pre-reset message,
#   for example). It never clears TF or the queue.
# - Queue: observations waiting for TF are kept in arrival order, at most
#   max_pending_observations (oldest dropped), each until its steady-time
#   deadline. The deadline is checked before every attempt; an expired entry
#   is dropped unpublished and the next entry is tried.
# - Output: one ThermalRayFrame per successfully processed observation; an
#   empty frame for a valid observation without regions; nothing for a
#   rejected, dropped or expired one. Rejections are counted per reason and
#   logged at most every 5 s each.

RAYS_TOPIC = 'thermal/rays'
RAYS_QOS_DEPTH = 5
LOG_INTERVAL_SEC = 5.0
IDENTITY_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def calibration_from_camera_info(msg: CameraInfo) -> RawCameraCalibration:
    """
    Return the raw, unmodified calibration in a CameraInfo message.

    Rejects (CameraGeometryError) rectification, projection matrices that are
    not [K | 0], binning and partial ROIs, none of which the ray geometry
    supports. Distortion is checked later by intrinsics_from_calibration.
    """
    if [float(v) for v in msg.r] != IDENTITY_R:
        raise CameraGeometryError(f'unsupported rectification R={list(msg.r)}')
    k = [float(v) for v in msg.k]
    expected_p = k[0:3] + [0.0] + k[3:6] + [0.0] + k[6:9] + [0.0]
    if [float(v) for v in msg.p] != expected_p:
        raise CameraGeometryError(f'unsupported projection P={list(msg.p)}; expected [K | 0]')
    if msg.binning_x not in (0, 1) or msg.binning_y not in (0, 1):
        raise CameraGeometryError(f'unsupported binning {msg.binning_x} x {msg.binning_y}')
    roi = msg.roi
    if roi.do_rectify or (roi.x_offset, roi.y_offset, roi.width, roi.height) not in (
            (0, 0, 0, 0), (0, 0, msg.width, msg.height)):
        raise CameraGeometryError('unsupported region of interest (only the full image)')
    return RawCameraCalibration(
        frame_id=msg.header.frame_id, width=msg.width, height=msg.height, k=msg.k,
        distortion_model=msg.distortion_model, d=msg.d)


def rays_to_message(stamp, frame_id: str, rays) -> ThermalRayFrame:
    """Build a ThermalRayFrame from adapter output, keeping the source stamp exactly."""
    message = ThermalRayFrame()
    message.header.stamp.sec, message.header.stamp.nanosec = stamp.sec, stamp.nanosec
    message.header.frame_id = frame_id
    message.source_frame_id = rays.source_frame_id
    message.image_width, message.image_height = rays.image_width, rays.image_height
    for region_ray in rays.rays:
        item = ThermalRegionRay()
        item.region_index = region_ray.region_index
        item.centroid_x, item.centroid_y = region_ray.centroid_xy
        item.touches_image_boundary = region_ray.touches_image_boundary
        item.origin.x, item.origin.y, item.origin.z = region_ray.ray.origin
        item.direction.x, item.direction.y, item.direction.z = region_ray.ray.direction
        message.rays.append(item)
    return message


class ThermalRayLocalisation(Node):

    def __init__(self, **node_kwargs):

        super().__init__('thermal_ray_localisation', **node_kwargs)

        self.target_frame = self._parameter('target_frame', 'base_link')
        self.camera_frame = self._parameter('camera_frame', 'thermal_link')
        self.principal_point_shift_px = self._required('principal_point_shift_px',
                                                       Parameter.Type.DOUBLE)
        self.frame_convention = self._required('camera_frame_convention', Parameter.Type.STRING)
        self.transform_timeout_sec = self._parameter('transform_timeout_sec', 0.5)
        self.max_pending = self._parameter('max_pending_observations', 10)
        self.stamp_tolerance_sec = self._parameter('stamp_tolerance_sec', 0.5)
        self.clock_reset_threshold_sec = self._parameter('clock_reset_threshold_sec', 1.0)
        self._check_parameters()

        self.raw_calibration = None
        self.intrinsics = None
        self.calibration_locked = False
        self.pending = deque()  # (source stamp, observation, intrinsics, steady deadline)
        self.newest_stamp_ns = None
        self.clock_resets = 0
        self.rejections = {}
        self._last_logged = {}
        self._last_tf_error = None

        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Fed only from /tf_static (same QoS as tf2's listener): the provenance
        # check for zero-stamped observations.
        self.static_tf_buffer = tf2_ros.Buffer()
        self.create_subscription(
            TFMessage, '/tf_static', self._static_tf_callback,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE))
        self._jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None,
                          min_backward=Duration(seconds=-self.clock_reset_threshold_sec),
                          on_clock_change=True),
            post_callback=self._on_clock_jump)

        self.publisher = self.create_publisher(
            ThermalRayFrame,
            RAYS_TOPIC,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=RAYS_QOS_DEPTH,
            ),
        )
        self.create_subscription(
            ThermalFrameObservationMsg, '/thermal/observations', self.observation_callback,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10))
        # Best effort accepts both reliable and best-effort CameraInfo publishers.
        self.create_subscription(
            CameraInfo, '/thermal/camera_info', self.camera_info_callback,
            qos_profile_sensor_data)
        self.create_timer(0.05, self.drain, clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.get_logger().info(
            f'Thermal ray localisation started: {self.camera_frame} -> {self.target_frame}, '
            f'principal point shift {self.principal_point_shift_px} px, '
            f'{self.frame_convention} camera frame')

    # --- Parameters ---------------------------------------------------------------

    def _parameter(self, name, default):
        return self.declare_parameter(
            name, default, ParameterDescriptor(read_only=True)).value

    def _required(self, name, parameter_type):
        self.declare_parameter(name, parameter_type, ParameterDescriptor(read_only=True))
        try:
            return self.get_parameter(name).value
        except ParameterUninitializedException:
            raise ValueError(f'parameter {name!r} must be set explicitly; it has no '
                             'default because the node must not guess calibration') from None

    def _check_parameters(self):
        for name in ('target_frame', 'camera_frame'):
            if not getattr(self, name):
                raise ValueError(f'parameter {name!r} must not be empty')
        shift = self.principal_point_shift_px
        if not math.isfinite(shift) or abs(shift) > MAX_PRINCIPAL_POINT_SHIFT_PX:
            raise ValueError(f'principal_point_shift_px={shift} is not a plausible correction')
        if self.frame_convention not in FRAME_CONVENTIONS:
            raise ValueError(f'camera_frame_convention must be one of {FRAME_CONVENTIONS}, '
                             f'not {self.frame_convention!r}')
        if not (isinstance(self.transform_timeout_sec, float)
                and 0 < self.transform_timeout_sec < math.inf):
            raise ValueError('transform_timeout_sec must be a positive number of seconds')
        if not (isinstance(self.max_pending, int) and self.max_pending >= 1):
            raise ValueError('max_pending_observations must be a positive integer')
        for name in ('stamp_tolerance_sec', 'clock_reset_threshold_sec'):
            value = getattr(self, name)
            if not (isinstance(value, float) and 0 < value < math.inf):
                raise ValueError(f'{name} must be a positive number of seconds')

    # --- Calibration --------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        if self.calibration_locked:
            return
        try:
            raw = calibration_from_camera_info(msg)
            if raw.frame_id != self.camera_frame:
                raise CameraGeometryError(
                    f'CameraInfo frame {raw.frame_id!r} is not {self.camera_frame!r}')
            if self.raw_calibration is not None:
                if raw != self.raw_calibration:
                    raise CameraGeometryError(
                        'the calibration differs from the session calibration')
                return  # an identical repeat
            intrinsics = intrinsics_from_calibration(
                raw, principal_point_shift_px=self.principal_point_shift_px,
                frame_convention=self.frame_convention)
        except CameraGeometryError as error:
            if self.raw_calibration is None:
                self._reject('camera_info', f'CameraInfo rejected: {error}')
            else:
                self._lock_calibration(str(error))
            return
        self.raw_calibration, self.intrinsics = raw, intrinsics
        self.get_logger().info(
            f'Thermal calibration accepted and frozen for this session: '
            f'{raw.width} x {raw.height}, fx={intrinsics.fx:.4f} fy={intrinsics.fy:.4f}, '
            f'published principal point ({raw.k[2]}, {raw.k[5]}) -> '
            f'({intrinsics.cx}, {intrinsics.cy})')

    def _lock_calibration(self, reason):
        dropped = len(self.pending)
        self.pending.clear()
        self.calibration_locked = True
        self.intrinsics = None
        if dropped:
            self.rejections['calibration_changed'] = (
                self.rejections.get('calibration_changed', 0) + dropped)
        self.get_logger().error(
            f'Thermal calibration changed during the session ({reason}); {dropped} pending '
            'observation(s) discarded and ray publication stopped until the node is restarted')

    # --- Observations -----------------------------------------------------------------

    def observation_callback(self, msg: ThermalFrameObservationMsg):
        try:
            observation = message_to_observation(msg)
        except ObservationConversionError as error:
            self._reject('observation', f'Thermal observation rejected: {error}')
            return
        if self.calibration_locked:
            self._reject('calibration_locked', 'Thermal observation rejected: calibration changed '
                                               'during the session; restart the node')
            return
        if self.intrinsics is None:
            self._reject('calibration', 'Thermal observation rejected: no valid calibration yet')
            return
        stamp_ns = observation.source_stamp_ns
        tolerance_ns = round(self.stamp_tolerance_sec * NS_PER_SECOND)
        now_ns = self.get_clock().now().nanoseconds
        if stamp_ns > now_ns + tolerance_ns:
            self._reject('future_stamp', f'Thermal observation rejected: stamped {stamp_ns} ns, '
                                         f'after the current time {now_ns} ns')
            return
        if self.newest_stamp_ns is not None and stamp_ns < self.newest_stamp_ns - tolerance_ns:
            self._reject('out_of_order', f'Thermal observation rejected: stamped {stamp_ns} ns, '
                                         f'more than {self.stamp_tolerance_sec} s before '
                                         f'{self.newest_stamp_ns} ns')
            return
        self.newest_stamp_ns = max(stamp_ns, self.newest_stamp_ns or 0)
        if len(self.pending) >= self.max_pending:
            self.pending.popleft()
            self._reject('queue_full', 'Thermal observation dropped: transform queue full')
        self.pending.append((msg.header.stamp, observation, self.intrinsics,
                             time.monotonic() + self.transform_timeout_sec))
        self.drain()

    def _on_clock_jump(self, jump):
        dropped = len(self.pending)
        self.pending.clear()
        self.newest_stamp_ns = None
        self.tf_buffer.clear()  # dynamic data only; static transforms are kept
        self.clock_resets += 1
        if dropped:
            self.rejections['clock_reset'] = self.rejections.get('clock_reset', 0) + dropped
        changed = jump.clock_change in (ClockChange.ROS_TIME_ACTIVATED,
                                        ClockChange.ROS_TIME_DEACTIVATED)
        self.get_logger().warning(
            f'ROS clock reset (jump {jump.delta.nanoseconds / NS_PER_SECOND:+.3f} s'
            f'{", clock source changed" if changed else ""}): {dropped} pending '
            'observation(s) and dynamic transforms discarded')

    def _static_tf_callback(self, msg: TFMessage):
        for transform in msg.transforms:
            self.static_tf_buffer.set_transform_static(transform, 'tf_static')

    def drain(self):
        """Process waiting observations in arrival order; expired ones are dropped first."""
        while self.pending:
            stamp, observation, intrinsics, deadline = self.pending[0]
            if time.monotonic() >= deadline:
                self.pending.popleft()
                self._reject('expired', (
                    f'No {self.target_frame} <- {self.camera_frame} transform at '
                    f'{stamp.sec}.{stamp.nanosec:09d} within {self.transform_timeout_sec} s '
                    f'({type(self._last_tf_error).__name__}: {self._last_tf_error})'))
                continue
            if not self._process(stamp, observation, intrinsics):
                return
            self.pending.popleft()

    def _process(self, stamp, observation, intrinsics):
        """Publish the observation's rays; return False only if its transform may still arrive."""
        if intrinsics is not self.intrinsics:  # defensive: the calibration lock clears the queue
            self._reject('calibration_changed', 'Thermal observation dropped: calibration changed')
            return True
        if observation.source_frame_id != self.camera_frame:
            self._reject('frame', f'Thermal observation rejected: frame '
                                  f'{observation.source_frame_id!r} is not {self.camera_frame!r}')
            return True
        transform = self._camera_transform(stamp, observation.source_stamp_ns)
        if transform is None:
            return True  # rejected
        if transform is False:
            return False  # not available yet
        t = transform.transform
        try:
            rays = observation_to_rays(
                observation, intrinsics, frame_id=self.target_frame,
                translation=(t.translation.x, t.translation.y, t.translation.z),
                rotation=rotation_from_quaternion(
                    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))
        except CameraGeometryError as error:
            self._reject('geometry', f'Thermal observation rejected: {error}')
            return True
        self.publisher.publish(rays_to_message(stamp, self.target_frame, rays))
        return True

    def _camera_transform(self, stamp, stamp_ns):
        """Return the transform, False if it may still arrive, or None after rejecting."""
        lookup_errors = (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                         tf2_ros.ExtrapolationException)
        if stamp_ns == 0:
            try:
                return self.static_tf_buffer.lookup_transform(
                    self.target_frame, self.camera_frame, Time())
            except lookup_errors as error:
                self._last_tf_error = error
            if self.tf_buffer.can_transform(self.target_frame, self.camera_frame, Time()):
                self._reject('zero_stamp_unverified', (
                    'Thermal observation rejected: zero stamp, but the camera transform is not '
                    'available from /tf_static alone (dynamic or mixed chain)'))
                return None
            return False
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, Time(nanoseconds=stamp_ns))
        except lookup_errors as error:
            self._last_tf_error = error
            return False
        returned = transform.header.stamp
        if (returned.sec, returned.nanosec) != (stamp.sec, stamp.nanosec):
            self._reject('transform_time', (
                f'Thermal observation rejected: transform for {stamp.sec}.{stamp.nanosec:09d} '
                f'came back stamped {returned.sec}.{returned.nanosec:09d}'))
            return None
        return transform

    def _reject(self, reason, text):
        count = self.rejections[reason] = self.rejections.get(reason, 0) + 1
        now = time.monotonic()
        if now - self._last_logged.get(reason, -math.inf) >= LOG_INTERVAL_SEC:
            self._last_logged[reason] = now
            self.get_logger().warning(f'{text} [{reason}: {count} so far]')


def main(args=None):

    rclpy.init(args=args)
    node = ThermalRayLocalisation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a genuine failure while running, not a side effect of Ctrl-C
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
