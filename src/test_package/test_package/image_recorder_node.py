#!/usr/bin/env python3

import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class ImageRecorderNode(Node):
    def __init__(self):
        super().__init__('image_recorder_node')

        # --- Parameter ---
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_dir', '/home/duckie6/GalapagosGang_URRMC2025/turtlebot_sign_images')
        self.declare_parameter('save_interval', 1.0)  # Sekunden

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        save_interval = self.get_parameter('save_interval').get_parameter_value().double_value

        # Verzeichnis anlegen
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info(f"Starte ImageRecorderNode")
        self.get_logger().info(f"  image_topic:  {image_topic}")
        self.get_logger().info(f"  save_dir:     {self.save_dir}")
        self.get_logger().info(f"  interval:     {save_interval} s")

        # CvBridge
        self.bridge = CvBridge()
        self.latest_img_msg = None

        # QoS (Best Effort, reicht für Kamera und ist schnell)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriber auf Kamerabild
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos
        )

        # Timer, der alle save_interval Sekunden ein Bild speichert
        self.timer = self.create_timer(save_interval, self.timer_callback)
        self.save = False

    def image_callback(self, msg: Image):
        # Nur das aktuellste Bild merken
        self.get_logger().info(f"New picture")
        self.save = True
        self.latest_img_msg = msg

    def timer_callback(self):
        if self.latest_img_msg is None:
            self.get_logger().warn("Noch kein Bild empfangen – kann nichts speichern.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Fehler: {e}")
            return

        # Dateiname mit Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.save_dir, f"frame_{timestamp}.jpg")

        if self.save:
            ok = cv2.imwrite(filename, cv_image)
            if ok:
                self.get_logger().info(f"Bild gespeichert: {filename}")
            else:
                self.get_logger().error(f"Konnte Bild NICHT speichern: {filename}")
            self.save = False


def main(args=None):
    rclpy.init(args=args)
    node = ImageRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
