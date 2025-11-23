#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray


class ReferenceLineNode(Node):
    def __init__(self):
        super().__init__('reference_line_node')

        # --- Parameter ---
        # Koordinatensystem des Strichs (z.B. base_link oder LiDAR-Link)
        self.declare_parameter('frame_id', 'base_link')

        # Länge des ersten Segments (in m) – 31 cm
        self.declare_parameter('segment1_length', 0.31)

        # Länge des zweiten Segments (in m)
        self.declare_parameter('segment2_length', 0.4)

        # Default-Winkel in Grad, falls noch keine Orientation-Message empfangen wurde
        self.declare_parameter('default_angle_deg', 0.0)

        # Marker-Topic
        self.declare_parameter('marker_topic', 'reference_line')

        # Orientation-Topic (von der anderen Node)
        self.declare_parameter('orient_topic', '/red_sign_orient')

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.L1 = self.get_parameter('segment1_length').get_parameter_value().double_value

        self.L2 = self.get_parameter('segment2_length').get_parameter_value().double_value

        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        orientation_topic = self.get_parameter('orient_topic').get_parameter_value().string_value

        # Letzter bekannter yaw-Winkel (in Rad)
        default_angle_deg = self.get_parameter('default_angle_deg').get_parameter_value().double_value
        self.current_yaw_rad = math.radians(default_angle_deg)

        # Publisher für Marker
        self.marker_pub = self.create_publisher(Marker, marker_topic, 1)

        # Subscriber für Orientation
        self.orientation_sub = self.create_subscription(
            Float32MultiArray,
            orientation_topic,
            self.orientation_callback,
            10
        )

        # Timer: Marker zyklisch neu senden (damit er in RViz bleibt)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.get_logger().info(
            f"ReferenceLineNode gestartet.\n"
            f"  Marker-Topic: {marker_topic}, \n"
            f"  Orientation-Topic: {orientation_topic}, \n"
            f"  Default-Winkel: {default_angle_deg}°, \n"
            f"  Default-Länge L2: {self.L2}°"
        )

        self.last_yaw_rad = None

    # --- Callback: Orientation aus anderem Node ---
    def orientation_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 1:
            self.current_yaw_rad = msg.data[0]  # yaw_rad
            if self.last_yaw_rad != round(self.current_yaw_rad,1):
                self.get_logger().info(
                    f"Orientation erhalten: yaw_rad={self.current_yaw_rad:.3f}"
                )
                self.last_yaw_rad = round(self.current_yaw_rad,1)
        else:
            self.get_logger().warn(
                "Orientation-Message erhalten, aber msg.data ist zu kurz!"
            )

    # --- Timer: Marker zeichnen ---
    def timer_callback(self):
        angle_rad = -self.current_yaw_rad  # immer letzter empfangener Wert

        # Punkte definieren:
        # p0: Ursprung
        p0 = (0.0, 0.0, 0.0)

        # p1: L1 in +x
        p1 = (self.L1, 0.0, 0.0)

        # p2: von p1 in Richtung angle_rad
        dx2 = self.L2 * math.cos(angle_rad)
        dy2 = self.L2 * math.sin(angle_rad)
        p2 = (p1[0] + dx2, p1[1] + dy2, 0.0)

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "reference_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Linien-Dicke
        marker.scale.x = 0.01  # 1 cm

        # Farbe grün
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0

        # Punkte eintragen
        marker.points = []
        for (x, y, z) in (p0, p1, p2):
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = z
            marker.points.append(pt)
            print(marker.points)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
