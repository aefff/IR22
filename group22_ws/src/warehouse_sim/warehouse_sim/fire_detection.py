import math

import cv2
from cv_bridge import CvBridge, CvBridgeError
from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalFrameObservation,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from warehouse_sim.thermal_observation_ros import (
    observation_to_message,
    ObservationConversionError,
    stamp_to_ns,
)

# Original detector parameters.
HOT_PIXEL_THRESHOLD = 180  # a pixel is hot if its intensity is strictly greater
MIN_HOT_PIXELS = 50  # warn when the whole-image hot-pixel count exceeds this

# Structured observations: relative name, so it resolves to /thermal/observations
# in the root namespace and can be remapped.
OBSERVATION_TOPIC = 'thermal/observations'
OBSERVATION_QOS_DEPTH = 5


class FireDetection(Node):

    def __init__(self):

        super().__init__('fire_detection')

        self.bridge = CvBridge()
        
        self._hazards = []
        self._odom = None
        self._camera_info = None
        self._reported = set()
        self.discovery_publisher = self.create_publisher(String, '/hazards/discovered', 10)
        self.create_subscription(PoseArray, '/hazards/poses', self._hazards_callback, 10)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(CameraInfo, '/thermal/camera_info', self._camera_info_callback, 10)

        self.observation_publisher = self.create_publisher(
            ThermalFrameObservationMsg,
            OBSERVATION_TOPIC,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=OBSERVATION_QOS_DEPTH,
            ),
        )

        self.subscription = self.create_subscription(
            Image,
            '/thermal/image_raw',
            self.image_callback,
            10,
        )

        self.get_logger().info('Fire detection node started')

    def image_callback(self, msg: Image) -> ThermalFrameObservation | None:
        # Returns the published observation, or None if nothing was published.
        # Publishing happens outside every handler below, so publisher errors
        # and unexpected exceptions propagate instead of being logged as bad frames.

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except (CvBridgeError, TypeError, ValueError) as error:
            # cv_bridge raises CvBridgeError for unsupported encodings; NumPy
            # raises TypeError/ValueError when data does not match width/step.
            # Only this one conversion call is covered by the handler.
            self.get_logger().error(
                f'Thermal frame dropped: cannot convert {msg.encoding!r} '
                f'{msg.width}x{msg.height} image to mono8 '
                f'({type(error).__name__}: {error})',
            )
            return None

        # The source stamp is validated before processing, but a bad stamp is
        # reported only after processing so that it cannot suppress the
        # legacy warning, which depends on the image alone.
        try:
            source_stamp_ns = stamp_to_ns(msg.header.stamp)
            stamp_error = None
        except (TypeError, ValueError) as error:
            source_stamp_ns = None
            stamp_error = error

        try:
            observation = self.process_thermal_image(
                image,
                source_stamp_ns,
                msg.header.frame_id or None,  # ROS uses '' for an unset frame
            )
        except ValueError as error:
            # Expected cause: an empty image. This handler would also catch a
            # ValueError raised by a defect inside processing; that frame is
            # then logged and dropped, never reported as an empty observation.
            # TypeError is not caught: mono8 conversion always gives 2-D uint8.
            self.get_logger().error(f'Thermal frame dropped: {error}')
            return None

        if stamp_error is not None:
            self.get_logger().error(
                f'Thermal frame dropped: invalid source timestamp '
                f'({type(stamp_error).__name__}: {stamp_error})',
            )
            return None

        try:
            message = observation_to_message(observation)
        except ObservationConversionError as error:
            self.get_logger().error(f'Thermal observation not published: {error}')
            return None

        self.observation_publisher.publish(message)
        self._detect_discovery(image)
        return observation

    def _hazards_callback(self, msg: PoseArray):
        self._hazards = [(pose.position.x, pose.position.y) for pose in msg.poses]

    def _odom_callback(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _camera_info_callback(self, msg: CameraInfo):
        self._camera_info = msg

    def _detect_discovery(self, image):
        """Associate a sufficiently large hot region with a hazard by bearing."""
        if self._odom is None or self._camera_info is None or not self._hazards:
            return
        binary = (image > HOT_PIXEL_THRESHOLD).astype('uint8')
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        pose = self._odom
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # The thermal lens sits slightly forward and to the right of base_link.
        camera_x = pose.position.x + 0.185 * math.cos(yaw) + 0.04 * math.sin(yaw)
        camera_y = pose.position.y + 0.185 * math.sin(yaw) - 0.04 * math.cos(yaw)
        fx = self._camera_info.k[0] if len(self._camera_info.k) >= 1 else 0.0
        width = self._camera_info.width or image.shape[1]
        if fx <= 0.0 or width <= 0:
            return
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < MIN_HOT_PIXELS:
                continue
            pixel_x = float(centroids[component][0])
            # Pinhole projection; camera frame is aligned with the robot's +X.
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
                    event = String()
                    event.data = name
                    self.discovery_publisher.publish(event)
                    self.get_logger().warning(f'FIRE HAZARD DISCOVERED: {name}')
    
    def process_thermal_image(
        self,
        image,
        source_stamp_ns: int | None,
        source_frame_id: str | None,
    ) -> ThermalFrameObservation:

        extraction = extract_thermal_regions(image, threshold=HOT_PIXEL_THRESHOLD)

        hot_pixels = extraction.hot_pixel_count

        # Minimum number of hot pixels required (whole image, not per region)
        if hot_pixels > MIN_HOT_PIXELS:

            self.get_logger().warning(
                f'FIRE HAZARD DETECTED! Hot pixels: {hot_pixels}',
            )

        return ThermalFrameObservation(
            source_stamp_ns=source_stamp_ns,
            source_frame_id=source_frame_id,
            extraction=extraction,
        )


def main(args=None):

    rclpy.init(args=args)
    node = FireDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
