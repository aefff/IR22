import math

from geometry_msgs.msg import PoseArray
from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

# SIMULATION ONLY: scenario management, not perception.
#
# This node tells the Gazebo heat manager which spawned fire the robot has
# seen. It does so by matching thermal regions from /thermal/observations to
# simulator ground truth (/hazards/poses, published by hazard_spawner). Its
# output must never feed operational hazard localisation or decision-making.
#
# The algorithm is the one previously run inside fire_detection
# (_detect_discovery), unchanged, except that it reads the detector's published
# regions instead of repeating connected-component extraction. Known
# limitations of that algorithm, deliberately not corrected here:
# - Frames: hazard positions are in the Gazebo 'world' frame, but the robot
#   pose comes from /odom, whose origin is the robot's spawn pose, so the two
#   are mixed without a transform.
# - Camera geometry: the lens offset (0.185 m forward, 0.04 m right) is
#   hard-coded rather than taken from TF, the principal point is (width-1)/2
#   rather than K[2], and only horizontal bearing is used (no range).
# - Suspected sign error: the region bearing atan2(u - cx, fx) is positive to
#   the right, while the hazard bearing is positive to the left.
# - Time: the latest /odom message is used; observation and odometry stamps
#   are not synchronised.

MIN_REGION_AREA_PX = 50  # same value as fire_detection.MIN_HOT_PIXELS
BEARING_TOLERANCE_RAD = 0.12


def discovered_hazard_indices(observation, camera_info, odom_pose, hazards):
    """Return the 1-based index of the nearest bearing-matched hazard for each large region."""
    q = odom_pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    # The thermal lens sits slightly forward and to the right of base_link.
    camera_x = odom_pose.position.x + 0.185 * math.cos(yaw) + 0.04 * math.sin(yaw)
    camera_y = odom_pose.position.y + 0.185 * math.sin(yaw) - 0.04 * math.cos(yaw)
    fx = camera_info.k[0] if len(camera_info.k) >= 1 else 0.0
    width = camera_info.width or observation.image_width
    if fx <= 0.0 or width <= 0:
        return []
    matches = []
    for region in observation.regions:
        if region.area_px < MIN_REGION_AREA_PX:
            continue
        # Pinhole projection; camera frame is aligned with the robot's +X.
        region_bearing = math.atan2((region.centroid_x - (width - 1) / 2.0), fx)
        candidates = []
        for index, (hx, hy) in enumerate(hazards, start=1):
            dx, dy = hx - camera_x, hy - camera_y
            bearing = math.atan2(dy, dx) - yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing - region_bearing) < BEARING_TOLERANCE_RAD:
                candidates.append((math.hypot(dx, dy), index))
        if candidates:
            matches.append(min(candidates)[1])
    return matches


class HazardDiscovery(Node):

    def __init__(self):

        super().__init__('hazard_discovery')

        self._hazards = []
        self._odom = None
        self._camera_info = None
        self._reported = set()
        self.discovery_publisher = self.create_publisher(String, '/hazards/discovered', 10)
        self.create_subscription(PoseArray, '/hazards/poses', self._hazards_callback, 10)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(
            CameraInfo, '/thermal/camera_info', self._camera_info_callback, 10)
        self.create_subscription(
            ThermalFrameObservationMsg, '/thermal/observations', self.observation_callback, 10)

        self.get_logger().info('Hazard discovery (simulation ground truth) started')

    def _hazards_callback(self, msg: PoseArray):
        self._hazards = [(pose.position.x, pose.position.y) for pose in msg.poses]

    def _odom_callback(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _camera_info_callback(self, msg: CameraInfo):
        self._camera_info = msg

    def observation_callback(self, msg: ThermalFrameObservationMsg):
        if self._odom is None or self._camera_info is None or not self._hazards:
            return
        for index in discovered_hazard_indices(
                msg, self._camera_info, self._odom, self._hazards):
            name = f'fire_hazard_{index:03d}'
            if name not in self._reported:
                self._reported.add(name)
                event = String()
                event.data = name
                self.discovery_publisher.publish(event)
                self.get_logger().warning(f'FIRE HAZARD DISCOVERED: {name}')


def main(args=None):

    rclpy.init(args=args)
    node = HazardDiscovery()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
