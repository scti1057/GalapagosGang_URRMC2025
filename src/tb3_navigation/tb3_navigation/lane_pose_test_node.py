#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class LanePoseTestNode(Node):
    """
    Test-Node für LaneFollowingNode.

    Sie published synthetische lane_pose-Daten, damit du die Lane-Follow-Regelung
    testen kannst, ohne echte Bildverarbeitung / Costmap.

    Annahme (kompatibel zur LaneFollowingNode):
      - Topic: /lane_pose  (konfigurierbar via Parameter)
      - frame_id: "base_link"
      - pose.position.y  -> lateraler Fehler [m]
      - yaw(pose.orientation) -> Heading-Fehler [rad]
    """

    def __init__(self):
        super().__init__('lane_pose_test_node')

        # Parameter
        self.declare_parameter('lane_pose_topic', '/lane_pose')
        self.declare_parameter('publish_rate', 20.0)   # Hz
        self.declare_parameter('amp_y', 0.15)          # Amplitude lateraler Fehler [m]
        self.declare_parameter('amp_yaw', 0.3)         # Amplitude Heading-Fehler [rad]
        self.declare_parameter('freq', 0.05)           # Frequenz [Hz] (wie schnell das Schlängeln ist)
        self.declare_parameter('pattern', 'sine')      # 'sine' oder 'constant'

        lane_pose_topic = self.get_parameter('lane_pose_topic').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.amp_y = self.get_parameter('amp_y').get_parameter_value().double_value
        self.amp_yaw = self.get_parameter('amp_yaw').get_parameter_value().double_value
        self.freq = self.get_parameter('freq').get_parameter_value().double_value
        self.pattern = self.get_parameter('pattern').get_parameter_value().string_value

        # Publisher
        self.lane_pose_pub = self.create_publisher(PoseStamped, lane_pose_topic, 10)

        # Zeitbasis
        self.t0 = time.time()

        # Timer
        timer_period = 1.0 / max(publish_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"LanePoseTestNode gestartet. Publiziere auf {lane_pose_topic} mit {publish_rate} Hz. "
            f"Pattern={self.pattern}, amp_y={self.amp_y}, amp_yaw={self.amp_yaw}, freq={self.freq}"
        )

    def timer_callback(self):
        now = time.time()
        t = now - self.t0

        if self.pattern == 'sine':
            # Schlängeln: lateraler Fehler und Heading-Fehler als Sinus über die Zeit
            y_err = self.amp_y * math.sin(2.0 * math.pi * self.freq * t)
            yaw_err = self.amp_yaw * math.sin(2.0 * math.pi * self.freq * t + math.pi / 4.0)
        elif self.pattern == 'constant':
            # Konstanter Offset (z.B. 0.2m seitlich, 10° yaw)
            y_err = self.amp_y
            yaw_err = self.amp_yaw
        else:
            # Fallback: keine Abweichung
            y_err = 0.0
            yaw_err = 0.0

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # das erwartet dein Lane-Follow-Controller

        msg.pose.position.x = 0.0
        msg.pose.position.y = y_err
        msg.pose.position.z = 0.0

        # yaw_err -> Quaternion
        qz = math.sin(yaw_err / 2.0)
        qw = math.cos(yaw_err / 2.0)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.lane_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LanePoseTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
