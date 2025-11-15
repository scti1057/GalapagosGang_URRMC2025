#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class LaneBevNode(Node):
    def __init__(self):
        super().__init__('lane_bev_node')

        # Parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('bev_width', 640)
        self.declare_parameter('bev_height', 480)
        self.declare_parameter('output_frame', 'base_footprint')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.bev_width = self.get_parameter('bev_width').get_parameter_value().integer_value
        self.bev_height = self.get_parameter('bev_height').get_parameter_value().integer_value
        self.output_frame = self.get_parameter('output_frame').get_parameter_value().string_value

        self.bridge = CvBridge()

        # Subscription
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Publishers for debug / visualization
        self.bev_pub = self.create_publisher(Image, 'lane_bev/image', 10)
        self.mask_pub = self.create_publisher(Image, 'lane_bev/mask', 10)

        # Hard-coded homography points for now (for a 640x480 input image)
        # These are EXAMPLES and will need tuning for your camera pose!
        # src: trapezoid in original image where the floor is visible
        self.src_points = np.float32([
            [180, 300],  # top-left in image
            [460, 300],  # top-right
            [40,  470],  # bottom-left
            [600, 470],  # bottom-right
        ])

        # dst: rectangle in BEV
        self.dst_points = np.float32([
            [200,   0],              # top-left in BEV
            [self.bev_width - 200, 0],  # top-right
            [200,   self.bev_height],   # bottom-left
            [self.bev_width - 200, self.bev_height],  # bottom-right
        ])

        self.M = None  # homography matrix, computed lazily

        self.get_logger().info(
            f'lane_bev_node started, subscribing to {image_topic}'
        )

    def compute_homography(self, img_shape):
        h, w = img_shape[:2]
        # If your camera resolution is different from 640x480,
        # you may want to scale src_points accordingly.
        # For now we assume 640x480 and log a warning if not.
        if (w, h) != (640, 480):
            self.get_logger().warn(
                f'Image size is {w}x{h}, but homography src_points assume 640x480. '
                'You will likely need to adjust src_points.'
            )
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        if self.M is None:
            self.compute_homography(cv_image.shape)

        # Apply BEV transform
        bev = cv2.warpPerspective(
            cv_image,
            self.M,
            (self.bev_width, self.bev_height)
        )

        # Convert to HSV for lane color filtering
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        # Simple HSV thresholds (will need tuning in your environment!)
        # Yellow
        lower_yellow = np.array([20,  80, 80], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White: high V, low saturation
        lower_white = np.array([0,   0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Morphological cleanup a bit
        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Publish BEV image
        try:
            bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
            bev_msg.header = msg.header
            bev_msg.header.frame_id = self.output_frame   # override frame
            self.bev_pub.publish(bev_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing BEV image: {e}')

        # Publish mask as mono8 image
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
            mask_msg.header = msg.header
            mask_msg.header.frame_id = self.output_frame   # override frame
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing mask image: {e}')

        # Log occasionally
        self.get_logger().debug('Published BEV and lane mask')


def main(args=None):
    rclpy.init(args=args)
    node = LaneBevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
