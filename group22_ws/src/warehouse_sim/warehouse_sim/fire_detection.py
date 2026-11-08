from cv_bridge import CvBridge, CvBridgeError
from hazard_perception.thermal_regions import (
    extract_thermal_regions,
    ThermalFrameObservation,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Original detector parameters.
HOT_PIXEL_THRESHOLD = 180  # a pixel is hot if its intensity is strictly greater
MIN_HOT_PIXELS = 50  # warn when the whole-image hot-pixel count exceeds this


def stamp_to_ns(stamp) -> int:
    """Convert a ``builtin_interfaces/Time`` to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class FireDetection(Node):

    def __init__(self):

        super().__init__('fire_detection')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/thermal/image_raw',
            self.image_callback,
            10,
        )

        self.get_logger().info('Fire detection node started')

    def image_callback(self, msg: Image) -> ThermalFrameObservation | None:

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

        try:
            return self.process_thermal_image(
                image,
                stamp_to_ns(msg.header.stamp),
                msg.header.frame_id or None,  # ROS uses '' for an unset frame
            )
        except ValueError as error:
            # Expected causes: an empty image or a negative timestamp. This
            # handler would also catch a ValueError raised by a defect inside
            # processing; that frame is then logged and dropped, never reported
            # as an empty observation. TypeError is not caught: mono8
            # conversion always gives 2-D uint8, so it would indicate a defect.
            self.get_logger().error(f'Thermal frame dropped: {error}')
            return None

    def process_thermal_image(
        self,
        image,
        source_stamp_ns: int,
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
