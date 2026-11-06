import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class FireDetection(Node):

    def __init__(self):

        super().__init__('fire_detection')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/thermal/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Fire detection node started')


    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='mono8'
        )

        # Identify unusually bright (hot) pixels
        threshold = 180

        _, mask = cv2.threshold(
            image,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Count hot pixels
        hot_pixels = cv2.countNonZero(mask)

        # Minimum number of hot pixels required
        if hot_pixels > 50:

            self.get_logger().warning(
                f'FIRE HAZARD DETECTED! Hot pixels: {hot_pixels}'
            )


def main(args=None):

    rclpy.init(args=args)
    node = FireDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
