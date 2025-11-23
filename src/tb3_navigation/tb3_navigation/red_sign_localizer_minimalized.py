#!/usr/bin/env python3

import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs import do_transform_point 

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import math
import statistics

import tf2_ros
from tf2_ros import TransformException


def quaternion_to_yaw(q) -> float:
    """
    Wandelt Quaternion in Yaw (2D) um.
    q: geometry_msgs.msg.Quaternion
    """
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class RedSignLocalizer(Node):
    """
    Sammelt mehrere Linien zum roten Schild im map-Frame und schätzt deren Schnittpunkt.

    - Sub:
        /red_sign_orient : Float32MultiArray [yaw_rad, pitch_rad]

    - TF:
        map -> base_link

    - Pub:
        /red_sign_pose : PoseStamped im global_frame (z.B. map)
    """

    def __init__(self):
        super().__init__('red_sign_localizer')

        # --- Parameter ---
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('orient_topic', '/red_sign_orient')
        self.declare_parameter('goal_pose_pub', '/red_sign_goal_pose')

        # Abstand für den Startpunkt der "zweiten Linie" vor dem Roboter
        self.declare_parameter('segment1_length', 0.31)

        # Entfernung zum Zielpunkt
        self.declare_parameter('forward_goal_dist', 1.0)  # z.B. 0.7 m vor den Bot

        # Maximale Punkte für median der Zielpunkte
        self.declare_parameter('max_points', 5)

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.orient_topic = self.get_parameter('orient_topic').get_parameter_value().string_value
        goal_pose_pub = self.get_parameter('goal_pose_pub').get_parameter_value().string_value

        self.L1 = self.get_parameter('segment1_length').get_parameter_value().double_value

        self.forward_goal_dist = self.get_parameter('forward_goal_dist').get_parameter_value().double_value

        self.max_points = self.get_parameter('max_points').get_parameter_value().integer_value

        self.x_goalInGlob: List[float] = []
        self.y_goalInGlob: List[float] = []

        # TF-Buffer und -Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Letzte geschätzte Schildposition
        self.last_estimate: Optional[Tuple[float, float]] = None

        # Subscriber für Orientation
        self.orient_sub = self.create_subscription(
            Float32MultiArray,
            self.orient_topic,
            self.orientation_callback,
            10
        )
        self.last_yaw_red_sign: Optional[Float32MultiArray] = None

        # Publisher für die geschätzte Schild-Pose
        self.goal_pose_pub = self.create_publisher(PoseStamped, goal_pose_pub, 10)

        self.pi = 3.1425

        self.last_log_info = (f"RedSignLocalizer (minimalized) gestartet.\n"
                              f"  Sub: orient_topic={self.orient_topic}\n"
                              f"  Pub: goal_pose_pub={goal_pose_pub}\n"
                              f"  global_frame: {self.global_frame}\n"
                              f"  base_frame:   {self.base_frame}\n"
                              f"  forward_goal_dist: {self.forward_goal_dist}\n"
                              f"  max_points: {self.max_points}")
        self.get_logger().info(self.last_log_info)


    # ----------------- Orientation-Callback -----------------
    def orientation_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 1:
            last_log_info = "Orientation-Message ohne yaw_rad empfangen."
            if self.last_log_info != last_log_info:
                self.get_logger().warn(last_log_info)
                self.last_log_info = last_log_info
            return
        yaw_red_sign = -msg.data[0]  # yaw_rad vom RedSignDetector

        x_goalInRob = self.L1 + self.forward_goal_dist*math.cos(yaw_red_sign)
        y_goalInRob = self.forward_goal_dist*math.sin(yaw_red_sign)
        # print(f"yaw_red_sign: {yaw_red_sign} | self.forward_goal_dist: {self.forward_goal_dist} | x_goalInRob: {x_goalInRob} | y_goalInRob: {y_goalInRob}")

        point_in_robmap = PointStamped()
        point_in_robmap.header.frame_id = self.base_frame
        point_in_robmap.header.stamp = self.get_clock().now().to_msg()
        point_in_robmap.point.x = x_goalInRob
        point_in_robmap.point.y = y_goalInRob
        point_in_robmap.point.z = 0.0

        # Aktuellen Transform base_link -> map holen
        try:
            tf_rob2glob = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                Time()
            )
        except TransformException as ex:
            last_log_info = f"TF {self.global_frame} -> {self.base_frame} nicht verfügbar: {ex}"
            if self.last_log_info != last_log_info:
                self.get_logger().warn(
                    last_log_info
                )
                self.last_log_info = last_log_info
            return
        point_in_globmap = do_transform_point(point_in_robmap, tf_rob2glob)

        x_goalInGlob = point_in_globmap.point.x
        y_goalInGlob = point_in_globmap.point.y

        self.x_goalInGlob.append(x_goalInGlob)
        if len(self.x_goalInGlob) > self.max_points:
            self.x_goalInGlob.pop(0)
        elif len(self.x_goalInGlob) < 2:
            return
        self.y_goalInGlob.append(y_goalInGlob)
        if len(self.y_goalInGlob) > self.max_points:
            self.y_goalInGlob.pop(0)
        elif len(self.y_goalInGlob) < 3:
            return

        x_med_goalInGlob = statistics.median(self.x_goalInGlob)
        y_med_goalInGlob = statistics.median(self.y_goalInGlob)

        last_log_info = f"Geschätzte Schildpose: x={x_med_goalInGlob:.2f}, y={y_med_goalInGlob:.2f}"
        if self.last_log_info != last_log_info:
            self.get_logger().info(
                last_log_info
            )
            self.last_log_info = last_log_info

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x_med_goalInGlob
        pose.pose.position.y = y_med_goalInGlob
        pose.pose.position.z = 0.0

        # Orientierung erstmal egal -> Identität
        # print(f"tf_rob2glob.transform.rotation: {tf_rob2glob.transform.rotation}")
        pose.pose.orientation.w = tf_rob2glob.transform.rotation.w

        self.goal_pose_pub.publish(pose)




def main(args=None):
    rclpy.init(args=args)
    node = RedSignLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
