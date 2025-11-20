#!/usr/bin/env python3

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from typing import Optional


class RedSignDetector(Node):
    """
    Detektiert ein rotes Schild im Kamerabild.

    - Sub:  image_topic (sensor_msgs/Image, BGR)
            scan_topic  (sensor_msgs/LaserScan) – optional für Distanz
    - Pub:  detected_topic (std_msgs/Bool)
            pose_topic     (geometry_msgs/PoseStamped) – relative Pose zum Schild

    Pose-Interpretation:
      - frame_id: pose_frame_id (Default: base_link)
      - position.x, position.y: aus Bild-Winkel + LiDAR-Distanz (falls verfügbar),
        sonst default_distance.
      - Orientierung yaw zeigt zum Schild.
    """

    def __init__(self):
        super().__init__('red_sign_detector')

        # --- Parameter einlesen ---
        self.declare_parameter('image_topic', '/camera/image_raw')

        # HSV-Schwellen für Rot
        self.declare_parameter('red1_lower', [0, 100, 100])
        self.declare_parameter('red1_upper', [10, 255, 255])
        self.declare_parameter('red2_lower', [170, 100, 100])
        self.declare_parameter('red2_upper', [180, 255, 255])

        self.declare_parameter('min_area', 500.0)
        self.declare_parameter('hfov', 160.0)  # ~160°
        self.declare_parameter('vfov', 160.0)  # ~160°
        self.declare_parameter('default_distance', 1.0)
        self.declare_parameter('pose_frame_id', 'base_link')

        self.declare_parameter('debug_image', False)
        self.declare_parameter('debug_image_topic', '/red_sign/debug_image')

        self.declare_parameter('detected_topic', '/red_sign_detected')
        self.declare_parameter('orient_topic', '/red_sign_orient')

        # Parameterwerte holen
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        red1_lower_arr = self.get_parameter('red1_lower').get_parameter_value().integer_array_value
        red1_upper_arr = self.get_parameter('red1_upper').get_parameter_value().integer_array_value
        red2_lower_arr = self.get_parameter('red2_lower').get_parameter_value().integer_array_value
        red2_upper_arr = self.get_parameter('red2_upper').get_parameter_value().integer_array_value

        self.red1_lower = np.array(red1_lower_arr, dtype=np.uint8)
        self.red1_upper = np.array(red1_upper_arr, dtype=np.uint8)
        self.red2_lower = np.array(red2_lower_arr, dtype=np.uint8)
        self.red2_upper = np.array(red2_upper_arr, dtype=np.uint8)

        self.min_area = float(
            self.get_parameter('min_area').get_parameter_value().double_value
        )
        self.hfov = float(self.get_parameter('hfov').get_parameter_value().double_value)
        self.vfov = float(self.get_parameter('vfov').get_parameter_value().double_value)
        self.default_distance = float(
            self.get_parameter('default_distance').get_parameter_value().double_value
        )
        self.pose_frame_id = (
            self.get_parameter('pose_frame_id').get_parameter_value().string_value
        )

        self.debug_image_enabled = bool(
            self.get_parameter('debug_image').get_parameter_value().bool_value
        )
        debug_image_topic = (
            self.get_parameter('debug_image_topic').get_parameter_value().string_value
        )

        detected_topic = (
            self.get_parameter('detected_topic').get_parameter_value().string_value
        )
        orient_topic = self.get_parameter('orient_topic').get_parameter_value().string_value

        # Letzte Werte für Logging
        self.last_cx_rounded = None
        self.last_cy_rounded = None
        self.last_pos_x_rounded = None
        self.last_pos_y_rounded = None

        # --- Bridge & Publisher/Sub ---
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.orient_pub = self.create_publisher(Float32MultiArray, orient_topic, 10)

        self.debug_pub = None
        if self.debug_image_enabled:
            self.debug_pub = self.create_publisher(Image, debug_image_topic, 1)

        self.pi = 3.1415

        self.get_logger().info(
            f"RedSignDetector gestartet.\n"
            f"  Sub: image={image_topic}\n"
            f"  Pub: detected={detected_topic}, orient={orient_topic}, debug={self.debug_image_enabled}"
        )

    # ----------------- Callbacks -----------------

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

        # ---- Winkel aus Pixelkoordinaten ----
        # HFOV/VFOV sind in Grad parametrisiert -> erst nach Radiant umrechnen
        hfov_rad = math.radians(self.hfov)
        vfov_rad = math.radians(self.vfov)

        # Horizontale Abweichung (Yaw):
        # Bildmitte = 0 rad, links positiv, rechts negativ (wie beim Lidar).
        center_x = width / 2.0
        dx_pixels = cx - center_x
        norm_x = dx_pixels / center_x           # [-1, 1]
        yaw_rad = -norm_x * (hfov_rad / 2.0)

        # Vertikale Abweichung (Pitch) – der Vollständigkeit halber
        center_y = height / 2.0
        dy_pixels = cy - center_y
        norm_y = dy_pixels / center_y           # [-1, 1]
        pitch_rad = norm_y * (vfov_rad / 2.0)

        # Fürs Logging in Grad
        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)

        # Publish Detection + Pose
        detected_msg = Bool()
        detected_msg.data = True
        self.detected_pub.publish(detected_msg)

        # Publish Winkel
        orientation = Float32MultiArray()
        orientation.data = [yaw_rad, pitch_rad] 
        self.orient_pub.publish(orientation)

        # self.get_logger().info(
        #     f"Orientation gesendet: yaw_rad={yaw_rad:.3f}"
        # )

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
