#!/usr/bin/env python3

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError


class RedSignDetector(Node):
    """
    Detektiert ein rotes Schild im Kamerabild.

    - Sub:  image_topic (sensor_msgs/Image, BGR)
    - Pub:  detected_topic (std_msgs/Bool)
            pose_topic     (geometry_msgs/PoseStamped) – grobe relative Pose zum Schild

    Pose-Interpretation (vereinfacht):
      - frame_id: pose_frame_id (Default: base_link)
      - position.x: default_distance
      - position.y: default_distance * tan(bearing_angle)
      - yaw: bearing_angle (Schild vor / leicht links / rechts)
    """

    def __init__(self):
        super().__init__('red_sign_detector')

        # --- Parameter einlesen ---
        self.declare_parameter('image_topic', '/camera/image_raw')

        self.declare_parameter('red1_lower', [0, 100, 100])
        self.declare_parameter('red1_upper', [10, 255, 255])
        self.declare_parameter('red2_lower', [170, 100, 100])
        self.declare_parameter('red2_upper', [180, 255, 255])

        self.declare_parameter('min_area', 500.0)
        self.declare_parameter('hfov', 1.047)  # ~60°
        self.declare_parameter('default_distance', 1.0)
        self.declare_parameter('pose_frame_id', 'base_link')

        self.declare_parameter('debug_image', False)
        self.declare_parameter('debug_image_topic', '/red_sign/debug_image')

        self.declare_parameter('detected_topic', '/red_sign_detected')
        self.declare_parameter('pose_topic', '/red_sign_pose')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        red1_lower_arr = self.get_parameter('red1_lower').get_parameter_value().integer_array_value
        red1_upper_arr = self.get_parameter('red1_upper').get_parameter_value().integer_array_value
        red2_lower_arr = self.get_parameter('red2_lower').get_parameter_value().integer_array_value
        red2_upper_arr = self.get_parameter('red2_upper').get_parameter_value().integer_array_value

        self.red1_lower = np.array(red1_lower_arr, dtype=np.uint8)
        self.red1_upper = np.array(red1_upper_arr, dtype=np.uint8)
        self.red2_lower = np.array(red2_lower_arr, dtype=np.uint8)
        self.red2_upper = np.array(red2_upper_arr, dtype=np.uint8)

        self.min_area = float(self.get_parameter('min_area').get_parameter_value().double_value)
        self.hfov = float(self.get_parameter('hfov').get_parameter_value().double_value)
        self.default_distance = float(self.get_parameter('default_distance').get_parameter_value().double_value)
        self.pose_frame_id = self.get_parameter('pose_frame_id').get_parameter_value().string_value

        self.debug_image_enabled = bool(self.get_parameter('debug_image').get_parameter_value().bool_value)
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value

        detected_topic = self.get_parameter('detected_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value

        # --- Bridge & Publisher/Sub ---
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)

        self.debug_pub = None
        if self.debug_image_enabled:
            from sensor_msgs.msg import Image as ImageMsg  # nur zur Klarheit
            self.debug_pub = self.create_publisher(ImageMsg, debug_image_topic, 1)

        self.get_logger().info(
            f"RedSignDetector gestartet. Sub: {image_topic}, "
            f"Pub: detected={detected_topic}, pose={pose_topic}, debug={self.debug_image_enabled}"
        )

    # ----------------- Bild-Callback -----------------

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Fehler: {e}")
            return

        height, width = cv_image.shape[:2]

        # BGR -> HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Rot-Masken
        mask1 = cv2.inRange(hsv, self.red1_lower, self.red1_upper)
        mask2 = cv2.inRange(hsv, self.red2_lower, self.red2_upper)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphologie zur Bereinigung
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Konturen finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.publish_no_detection()
            if self.debug_image_enabled:
                self.publish_debug_image(cv_image, None)
            return

        # Größte Kontur wählen
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))
        max_area = areas[max_idx]
        cnt = contours[max_idx]

        if max_area < self.min_area:
            # Zu klein -> kein Schild
            self.publish_no_detection()
            if self.debug_image_enabled:
                self.publish_debug_image(cv_image, None)
            return

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            self.publish_no_detection()
            if self.debug_image_enabled:
                self.publish_debug_image(cv_image, None)
            return

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Bearing-Winkel aus Bildkoordinate
        center_x = width / 2.0
        dx_pixels = cx - center_x
        # Normierter Versatz [-1,1]
        norm_x = dx_pixels / center_x
        bearing = norm_x * (self.hfov / 2.0)  # rad

        # Grobe Position im pose_frame_id
        dist = self.default_distance
        pos_x = dist * math.cos(bearing)
        pos_y = dist * math.sin(bearing)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.pose_frame_id

        pose_msg.pose.position.x = pos_x
        pose_msg.pose.position.y = pos_y
        pose_msg.pose.position.z = 0.0

        # Orientierung: Blickrichtung zum Schild
        qz = math.sin(bearing / 2.0)
        qw = math.cos(bearing / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        # Publish Detection + Pose
        detected_msg = Bool()
        detected_msg.data = True
        self.detected_pub.publish(detected_msg)
        self.pose_pub.publish(pose_msg)

        if self.debug_image_enabled:
            self.publish_debug_image(cv_image, (cx, cy), cnt)

    # ----------------- Helper -----------------

    def publish_no_detection(self):
        msg = Bool()
        msg.data = False
        self.detected_pub.publish(msg)
        # Pose lassen wir in dem Fall einfach weg

    def publish_debug_image(self, image_bgr, center, contour=None):
        debug_img = image_bgr.copy()
        if contour is not None:
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
        if center is not None:
            cx, cy = center
            cv2.circle(debug_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            dbg_msg.header.stamp = self.get_clock().now().to_msg()
            dbg_msg.header.frame_id = self.pose_frame_id
            self.debug_pub.publish(dbg_msg)
        except CvBridgeError as e:
            self.get_logger().warn(f"Debug-Bild konnte nicht konvertiert werden: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RedSignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
