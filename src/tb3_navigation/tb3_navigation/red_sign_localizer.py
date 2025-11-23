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

        # Maximale Anzahl gespeicherter Linien
        self.declare_parameter('max_lines', 10)

        # Mindestbewegung, damit eine neue Linie akzeptiert wird
        self.declare_parameter('min_pose_trans', 0.05)      # 5 cm
        self.declare_parameter('min_pose_rot_deg', 5.0)     # 5°

        # Mindest-Winkelunterschied zwischen zwei Linien, damit wir ihren Schnittpunkt verwenden
        self.declare_parameter('min_angle_diff_deg', 2.0)   # 2°

        self.declare_parameter('yaw_forward_thresh_deg', 5.0)
        self.declare_parameter('forward_goal_dist', 0.7)  # z.B. 0.7 m vor den Bot

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.orient_topic = self.get_parameter('orient_topic').get_parameter_value().string_value
        goal_pose_pub = self.get_parameter('goal_pose_pub').get_parameter_value().string_value

        self.L1 = self.get_parameter('segment1_length').get_parameter_value().double_value
        self.max_lines = self.get_parameter('max_lines').get_parameter_value().integer_value

        self.min_pose_trans = self.get_parameter('min_pose_trans').get_parameter_value().double_value
        min_pose_rot_deg = self.get_parameter('min_pose_rot_deg').get_parameter_value().double_value
        self.min_pose_rot = math.radians(min_pose_rot_deg)

        min_angle_diff_deg = self.get_parameter('min_angle_diff_deg').get_parameter_value().double_value
        self.min_angle_diff_rad = math.radians(min_angle_diff_deg)
        # Für den Kreuzprodukt-Schwellenwert: |v0 x v1| = sin(Delta-Winkel)
        self.min_cross = math.sin(self.min_angle_diff_rad)

        self.yaw_forward_thresh_rad = math.radians(
            self.get_parameter('yaw_forward_thresh_deg').get_parameter_value().double_value
        )
        self.forward_goal_dist = self.get_parameter('forward_goal_dist').get_parameter_value().double_value

        # TF-Buffer und -Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Liste von Linien: [(x0, y0, vx, vy), ...]
        self.lines: List[Tuple[float, float, float, float]] = []

        # Letzte geschätzte Schildposition
        self.last_estimate: Optional[Tuple[float, float]] = None

        # Letzte Roboterpose (für Bewegungs-Filter)
        self.last_robot_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)

        self.declare_parameter('scan_topic', '/scan')
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value

        # QoS an /scan anpassen (Best Effort, sonst bekommst du die RELIABILITY-Warnung)
        scan_qos = QoSProfile(depth=10)
        scan_qos.history = HistoryPolicy.KEEP_LAST
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            scan_qos
        )
        self.last_scan: Optional[LaserScan] = None

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

        self.lidar_subtract = 0.1

        self.last_log_info = (f"RedSignLocalizer gestartet.\n"
                              f"  Sub: orient_topic={self.orient_topic}\n"
                              f"  Pub: goal_pose_pub={goal_pose_pub}\n"
                              f"  global_frame: {self.global_frame}\n"
                              f"  base_frame:   {self.base_frame}\n"
                              f"  segment1_length (L1): {self.L1} m\n"
                              f"  max_lines: {self.max_lines}\n"
                              f"  min_pose_trans: {self.min_pose_trans} m\n"
                              f"  min_pose_rot: {min_pose_rot_deg}°\n")
                            #   f"  min_angle_diff: {min_angle_diff_deg}°")
        self.get_logger().info(self.last_log_info)

    # ----------------- Orientation-Callback -----------------

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    def orientation_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 1:
            last_log_info = "Orientation-Message ohne yaw_rad empfangen."
            if self.last_log_info != last_log_info:
                self.get_logger().warn(last_log_info)
                self.last_log_info = last_log_info
            return

        yaw_red_sign = msg.data[0]  # yaw_rad vom RedSignDetector

        # Aktuellen Transform map -> base_link holen
        try:
            tf = self.tf_buffer.lookup_transform(
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

        # Roboterpose in map
        x_r = tf.transform.translation.x
        y_r = tf.transform.translation.y
        yaw_r = quaternion_to_yaw(tf.transform.rotation)

        # if abs(yaw_red_sign) < self.yaw_forward_thresh_rad:
        #     # Geradeaus-Modus: wir fahren einfach vorwärts
        #     x_s = x_r + self.forward_goal_dist * math.cos(yaw_r)
        #     y_s = y_r + self.forward_goal_dist * math.sin(yaw_r)

        #     self.last_estimate = (x_s, y_s)
        #     self.get_logger().info(
        #         f"Schild fast geradeaus -> sende einfachen Vorwärts-Goal: "
        #         f"x={x_s:.3f}, y={y_s:.3f}"
        #     )
        #     self.publish_pose(x_s, y_s)
        #     # OPTIONAL: hier return, damit wir keine Linie speichern
        #     return

        # --- Bewegungs-Filter ---
        if self.last_robot_pose is not None:
            last_x, last_y, last_yaw = self.last_robot_pose
            dx = x_r - last_x
            dy = y_r - last_y
            dtrans = math.sqrt(dx * dx + dy * dy)
            dyaw = math.atan2(math.sin(yaw_r - last_yaw), math.cos(yaw_r - last_yaw))

            if dtrans < self.min_pose_trans and abs(dyaw) < self.min_pose_rot: #abs(self.last_yaw_red_sign-yaw_red_sign) < self.min_pose_rot: #
                # Poseänderung zu klein -> keine neue Linie,
                # aber wenn wir schon eine Schätzung haben, publishen wir sie trotzdem.
                if self.last_estimate is not None:
                    x_s, y_s = self.last_estimate
                    # self.publish_pose(x_s, y_s)
                else:
                    last_log_info = "Poseänderung klein und noch keine Schild-Schätzung vorhanden."
                    if self.last_log_info != last_log_info:
                        self.get_logger().info( 
                            "Poseänderung klein und noch keine Schild-Schätzung vorhanden."
                        )
                        self.last_log_info = last_log_info
                return

        if yaw_red_sign >= 0:
            yaw_red_sign = 2*math.pi - yaw_red_sign
        else:
            yaw_red_sign = -yaw_red_sign

        # Startpunkt der zweiten Linie p1 (L1 vor dem Roboter, in Richtung yaw_r)
        x0 = x_r + self.L1 * math.cos(yaw_r)
        y0 = y_r + self.L1 * math.sin(yaw_r)

        # globaler Richtungswinkel zur Linie (Roboter-Orientierung + Messwinkel)
        theta = yaw_r + yaw_red_sign

        vx = math.cos(theta)
        vy = math.sin(theta)

        # print(f"vx: {vx}")
        # print(f"vy: {vy}")

        # Linie speichern
        self.lines.append((x0, y0, vx, vy))
        if len(self.lines) > self.max_lines:
            self.lines.pop(0)

        # letzte Roboterpose aktualisieren
        self.last_robot_pose = (x_r, y_r, yaw_r)
        self.last_yaw_red_sign = yaw_red_sign

        # self.get_logger().info(
        #     f"Neue Linie gespeichert: "
        #     f"Start=({x0:.3f}, {y0:.3f}), "
        #     f"theta={math.degrees(theta):.1f}°, "
        #     f"Anzahl Linien={len(self.lines)}"
        # )

        # Versuchen, Schnittpunkt zu schätzen
        estimate = self.compute_intersection_estimate()
        if estimate is not None:
            x_s, y_s, n_pairs = estimate
            self.last_estimate = (x_s, y_s)

            # refine mit LiDAR
            refined = self.refine_with_lidar(x_s, y_s, x_r, y_r, theta)
            if refined is not None:
                x_s, y_s = refined
                self.last_estimate = refined
                last_log_info = f"Refined Schildpose mit LiDAR: x={x_s:.3f}, y={y_s:.3f}"
                if self.last_log_info != last_log_info:
                    self.get_logger().info(
                        last_log_info
                    )
                    self.last_log_info = last_log_info
            self.publish_pose(x_s, y_s)

            last_log_info = f"Neue pose gepublished: x={x_s:.3f} | y={y_s:.3f}"
            if self.last_log_info != last_log_info:
                self.get_logger().info(
                    last_log_info
                )
                self.last_log_info = last_log_info
            self.last_log_info = last_log_info


    # ----------------- Pose publishen -----------------

    def publish_pose(self, x_s: float, y_s: float):
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x_s
        pose.pose.position.y = y_s
        pose.pose.position.z = 0.0

        # Orientierung erstmal egal -> Identität
        pose.pose.orientation.w = 1.0

        self.goal_pose_pub.publish(pose)

    # ----------------- Geometrie: Linien-Schnittpunkt -----------------

    def intersect_two_lines(self, p0, v0, p1, v1) -> Optional[Tuple[float, float]]:
        """
        Schnittpunkt zweier unendlicher Geraden in 2D.

        Gerade 0: p0 + t * v0
        Gerade 1: p1 + s * v1

        Rückgabe:
          (x, y) oder None, falls (fast) parallel.
        """
        x0, y0 = p0
        x1, y1 = p1
        vx0, vy0 = v0
        vx1, vy1 = v1

        # 2D Kreuzprodukt
        denom = vx0 * vy1 - vy0 * vx1
        if abs(denom) < self.min_cross:
            # Winkelunterschied zu klein -> Linien fast parallel -> nicht verwenden
            return None

        dx = x1 - x0
        dy = y1 - y0
        t = (dx * vy1 - dy * vx1) / denom

        xi = x0 + t * vx0
        yi = y0 + t * vy0
        return (xi, yi)

    def compute_intersection_estimate(self) -> Optional[Tuple[float, float, int]]:
        """
        Berechnet aus allen Linienpaaren einen gemittelten Schnittpunkt.

        - Ignoriert Linienpaare, die zu parallel sind.
        - Mittelt alle gültigen Schnittpunkte.
        """
        n = len(self.lines)
        if n < 2:
            return None

        points = []
        for i in range(n):
            x0, y0, vx0, vy0 = self.lines[i]
            for j in range(i + 1, n):
                x1, y1, vx1, vy1 = self.lines[j]
                p = self.intersect_two_lines(
                    (x0, y0), (vx0, vy0),
                    (x1, y1), (vx1, vy1)
                )
                if p is not None:
                    points.append(p)

        if not points:
            # self.get_logger().warn(
            #     "Keine gültigen Schnittpunkte (Linien zu parallel oder Basislinie zu klein)."
            # )
            return None

        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        x_mean = sum_x / len(points)
        y_mean = sum_y / len(points)

        return x_mean, y_mean, len(points)
    

    def refine_with_lidar(self, x_s: float, y_s: float,
                        x_r: float, y_r: float, theta: float
                        ) -> Optional[Tuple[float, float]]:
        # print(self.last_scan)
        if self.last_scan is None:
            return None

        scan = self.last_scan

        # Schild-Position relativ zum Roboter (base_link)
        # dx = x_s - x_r
        # dy = y_s - y_r
        # print(f"x_s: {x_s}")
        # print(f"y_s: {y_s}")
        # print(f"x_r: {x_r}")
        # print(f"y_r: {y_r}")

        point_in_globmap = PointStamped()
        point_in_globmap.header.frame_id = 'map'
        point_in_globmap.header.stamp = self.get_clock().now().to_msg()
        point_in_globmap.point.x = x_s
        point_in_globmap.point.y = y_s
        point_in_globmap.point.z = 0.0


        # Lookup the transform from the source frame to the target frame
        tf_glob2rob = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.global_frame,
            rclpy.time.Time() # Use the latest available transform
        )
        point_sign_in_rob = do_transform_point(point_in_globmap, tf_glob2rob)

        # Schild-Position relativ zum Roboter (base_link)
        dx = point_sign_in_rob.point.x
        dy = point_sign_in_rob.point.y

        angle_to_sign = math.atan(dy/dx)

        # print(f"point_sign_in_rob: x={dx} | y={dy} | angle_to_sign={math.degrees(angle_to_sign)}")


        # # Drehung in base_link: yaw_r ist Orientierung von base_link in map
        # # -> wir drehen den Vektor in die lokale Basis (Rotation -yaw_r)
        # cos_y = math.cos(theta)
        # sin_y = math.sin(theta)
        # x_bl = cos_y * dx - sin_y * dy
        # y_bl = sin_y * dx + cos_y * dy

        # angle_to_sign = theta #math.atan2(y_bl, x_bl)

        # check: liegt Winkel im Scan-Bereich?
        if angle_to_sign < scan.angle_min or angle_to_sign > scan.angle_max:
            self.get_logger().warn(f"Angle_to_sign außerhalb LaserScan-FOV: {angle_to_sign}")
            return None

        idx = int(round((angle_to_sign - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(idx, len(scan.ranges) - 1))

        r = scan.ranges[idx]
        if not math.isfinite(r) or r <= 0.05:
            self.get_logger().warn("LiDAR-Range ungültig oder zu klein.")
            return None
        else:
            self.get_logger().info(f"LiDAR-Range: {r}.")
        r = r - self.lidar_subtract

        # neue, präzisere Position im base_link
        x_bl_ref = r * math.cos(angle_to_sign)
        y_bl_ref = r * math.sin(angle_to_sign)
        # print(f"x_bl_ref: {x_bl_ref}")
        # print(f"y_bl_ref: {y_bl_ref}")

        tf_rob2glob = self.tf_buffer.lookup_transform(
            self.global_frame,
            self.base_frame,
            rclpy.time.Time() # Use the latest available transform
        )

        point_in_robmap = PointStamped()
        point_in_robmap.header.frame_id = 'map'
        point_in_robmap.header.stamp = self.get_clock().now().to_msg()
        point_in_robmap.point.x = x_bl_ref
        point_in_robmap.point.y = y_bl_ref
        point_in_robmap.point.z = 0.0


        # Lookup the transform from the source frame to the target frame
        tf_glob2rob = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.global_frame,
            rclpy.time.Time() # Use the latest available transform
        )
        point_sign_in_glob = do_transform_point(point_in_robmap, tf_rob2glob)

        # Schild-Position relativ zum Roboter (base_link)
        x_ref = point_sign_in_glob.point.x
        y_ref = point_sign_in_glob.point.y

        # print(f"point_sign_in_glob: x={x_ref} | y={y_ref}")

        # # zurück nach map
        # cos_y = math.cos(theta)
        # sin_y = math.sin(theta)
        # x_ref = x_r + cos_y * x_bl_ref - sin_y * y_bl_ref
        # y_ref = y_r + sin_y * x_bl_ref + cos_y * y_bl_ref

        return x_ref, y_ref



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
