#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool


class LaneFollowingNode(Node):
    """
    Einfache Lane-Following-Node:

    - Subscribt auf:
        * /lane_following_enabled (std_msgs/Bool): Mode an/aus
        * /lane_pose (geometry_msgs/PoseStamped): Lage der Lane relativ zum Roboter
          Annahme:
            - pose.position.y: lateraler Fehler [m]
            - yaw(pose.orientation): Heading-Fehler [rad]
    - Published:
        * /cmd_vel (oder konfigurierbares Topic): Steuerbefehl für Turtlebot

    Die eigentliche Lane-Detektion (Birdseye, Masken, etc.) wird von anderen Nodes erledigt.
    Diese Node kümmert sich nur um den Regler + Start/Stopp-Logik.
    """

    def __init__(self):
        super().__init__('lane_following_node')

        # ----------------- Parameter -----------------
        self.declare_parameter('lane_pose_topic', '/lane_pose')
        self.declare_parameter('enable_topic', '/lane_following_enabled')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('control_rate', 20.0)      # Hz
        self.declare_parameter('linear_speed', 0.15)      # m/s
        self.declare_parameter('k_y', 1.5)                # Gain für lateralen Fehler
        self.declare_parameter('k_yaw', 2.0)              # Gain für Heading-Fehler
        self.declare_parameter('max_angular', 1.5)        # max |omega| [rad/s]

        lane_pose_topic = self.get_parameter('lane_pose_topic').get_parameter_value().string_value
        enable_topic = self.get_parameter('enable_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        control_rate = self.get_parameter('control_rate').get_parameter_value().double_value

        # ----------------- State -----------------
        self.enabled = False
        self.have_lane_pose = False
        self.last_lateral_error = 0.0
        self.last_heading_error = 0.0

        # ----------------- Subscriptions -----------------
        self.enable_sub = self.create_subscription(
            Bool,
            enable_topic,
            self.enable_callback,
            10
        )

        self.lane_pose_sub = self.create_subscription(
            PoseStamped,
            lane_pose_topic,
            self.lane_pose_callback,
            10
        )

        # ----------------- Publisher -----------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        # ----------------- Timer für Regler-Loop -----------------
        timer_period = 1.0 / max(control_rate, 1.0)
        self.control_timer = self.create_timer(
            timer_period,
            self.control_loop
        )

        self.get_logger().info(
            f"LaneFollowingNode gestartet. "
            f"Sub lane_pose: {lane_pose_topic}, enable: {enable_topic}, "
            f"pub cmd_vel: {cmd_vel_topic}, rate: {control_rate} Hz"
        )

    # ------------- Callbacks -------------

    def enable_callback(self, msg: Bool):
        """Lane-Following ein-/ausschalten."""
        previous = self.enabled
        self.enabled = bool(msg.data)

        if self.enabled and not previous:
            self.get_logger().info("Lane following ENABLED.")
        elif not self.enabled and previous:
            self.get_logger().info("Lane following DISABLED. Sende Stopp-Cmd.")
            self.publish_zero_cmd()

    def lane_pose_callback(self, msg: PoseStamped):
        """
        Speichert aktuellen Lane-Fehler ab.

        Annahme:
          - position.y = lateraler Fehler [m]
          - orientation = Heading-Fehler (nur Yaw) [rad]
        """
        # Pose frame kannst du später prüfen / anpassen (map, odom, base_link)
        lateral_error = msg.pose.position.y

        # Quaternion -> yaw
        q = msg.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self.last_lateral_error = lateral_error
        self.last_heading_error = yaw
        self.have_lane_pose = True

    # ------------- Control-Loop -------------

    def control_loop(self):
        """Regler-Loop: berechnet aus Lane-Fehlern ein cmd_vel."""
        if not self.enabled:
            # Sicherstellen, dass wir nicht „aus Versehen“ fahren
            return

        if not self.have_lane_pose:
            # Keine Lane-Info -> lieber stehen bleiben
            self.get_logger().warn_throttle(1.0, "Lane following aktiv, aber noch keine lane_pose empfangen.")
            self.publish_zero_cmd()
            return

        linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        k_y = self.get_parameter('k_y').get_parameter_value().double_value
        k_yaw = self.get_parameter('k_yaw').get_parameter_value().double_value
        max_angular = self.get_parameter('max_angular').get_parameter_value().double_value

        # Fehler einlesen
        y_err = self.last_lateral_error
        yaw_err = self.last_heading_error

        # Simpler P-Regler: omega = k_y * y_err + k_yaw * yaw_err
        omega = k_y * y_err + k_yaw * yaw_err

        # Limitieren
        if omega > max_angular:
            omega = max_angular
        elif omega < -max_angular:
            omega = -max_angular

        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = omega

        self.cmd_vel_pub.publish(cmd)

    # ------------- Hilfsfunktionen -------------

    @staticmethod
    def quaternion_to_yaw(x, y, z, w) -> float:
        """Quaternion -> yaw (z-Rotation)."""
        # Standard-Formel
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_zero_cmd(self):
        """Stoppt den Roboter."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_zero_cmd()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
