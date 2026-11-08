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
        return observation

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
