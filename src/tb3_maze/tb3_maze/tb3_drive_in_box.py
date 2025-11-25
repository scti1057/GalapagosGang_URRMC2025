#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SignDoorNavigator für Turtlebot3 + Nav2 + Lidar.

Szenario:
- Bot steht VOR einem Kasten mit Öffnung (Eingang).
- Ein YOLO-Modell (später) erkennt ein Schild "Einfahrt" und triggert ein Bool-Topic.
- Dann:
  1) Mit dem Lidar vor dem Bot die Wand scannen.
  2) Größte Lücke finden, die wie ein "Türrahmen" aussieht
     (links UND rechts Wand auf ähnlicher Distanz).
  3) Kurz HINTER dieser Öffnung ein Nav2-Ziel setzen (im map-Frame).
  4) Wenn das Ziel erreicht ist, /start_exploration = True publizieren,
     sodass deine bestehende Frontier-/Box-Node übernimmt.
"""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from tf2_ros import Buffer, TransformListener


def euler_from_quaternion(quat: Tuple[float, float, float, float]):
    """
    Konvertiert Quaternion (x, y, z, w) -> Euler (roll, pitch, yaw).
    """
    x, y, z, w = quat

    # Roll (x-Achse)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-Achse)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)  # clamp
    pitch = math.asin(t2)

    # Yaw (z-Achse)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """
    Konvertiert Euler (roll, pitch, yaw) -> Quaternion (x, y, z, w).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


class DriveInBox(Node):
    def __init__(self):
        super().__init__('sign_door_navigator')

        # === Parameter ===
        # Debug: Schild-Erkennung simulieren (0.0 = aus)
        self.declare_parameter('simulate_sign_after_sec', 0.0)

        # Bereich vor dem Roboter, in dem nach Lücken gesucht wird (± Winkelbereich)
        self.declare_parameter('door_search_angle_deg', 60.0)

        # Grobe Wanddistanz (m) direkt vor dem Roboter (für Lücken-Threshold & Ziel-Offset)
        self.declare_parameter('wall_distance_estimate', 0.30)

        # Wie weit der Zielpunkt hinter die Wand gesetzt wird (m)
        self.declare_parameter('inside_offset', 0.1)

        # Faktor: ab welcher Distanz gilt ein Punkt als "Lücke" (offen)
        # r > open_threshold_factor * wall_distance_estimate -> offen
        self.declare_parameter('open_threshold_factor', 0.9)

        # Symmetrieprüfung: Winkel links/rechts der Lücke, an denen Wand sein soll
        self.declare_parameter('door_check_offset_deg', 5.0)

        # Fensterbreite um den Prüfstrahl (Bogenlänge auf Wanddistanz), z.B. ±10 cm
        self.declare_parameter('door_wall_window_m', 0.10)

        # Wände sollen z.B. zwischen 0.15 m und 0.40 m liegen
        self.declare_parameter('min_wall_dist_m', 0.15)
        self.declare_parameter('max_wall_dist_m', 0.60)

        self.declare_parameter('goal_angle_scale', 0.0)

        # Topics
        # Bool-Topic: wird auf True gesetzt, wenn das Einfahrt-Schild erkannt wurde
        self.declare_parameter('sign_topic', 'sign_detected')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('exploration_trigger_topic', 'start_exploration')

        # === Zustand ===
        self.sign_seen = False
        self.goal_sent = False
        self.exploration_started = False
        self.latest_scan: Optional[LaserScan] = None
        self.enabled = False
        self.tunnel_counter = 0

        # Sign-based overrides (only used in lane_following mode)
        #   'none'       -> normal lane following
        #   'stop'       -> do not publish x_tar (DriveNode stops via timeout)
        #   'turn_left'  -> follow x_yellow_near only for a while
        #   'turn_right' -> follow x_white_near only for a while
        self.sign_state = 'none'
        self.declare_parameter('sign_area_threshold', 8000.0)
        self.sign_area_threshold = self.get_parameter('sign_area_threshold').get_parameter_value().double_value

        # === Subscriptions ===
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        sign_topic = self.get_parameter('sign_topic').get_parameter_value().string_value

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_cb,
            qos_profile_sensor_data,
        )

        # self.sign_sub = self.create_subscription(
        #     Bool,
        #     sign_topic,
        #     self.sign_cb,
        #     10
        # )

        self.enable_sub = self.create_subscription(
            Bool,
            'mission3/drive_in_box_enable',
            self.enable_cb,
            10
        )

        # QoS: sensor-style, keep last
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # === Subscriber for YOLO traffic sign detections ===
        self.sub_signs = self.create_subscription(
            Detection2DArray,
            '/yolo/sign_detections',
            self.sign_detections_callback,
            sensor_qos
        )

        # === Publisher, um deine Frontier-/Explorer-Node zu aktivieren ===
        exploration_topic = self.get_parameter('exploration_trigger_topic').get_parameter_value().string_value
        self.exploration_pub = self.create_publisher(Bool, exploration_topic, 10)

        # === Nav2 Action Client ===
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # === TF ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # # === Optional: Schild-Erkennung simulieren ===
        # simulate_after = self.get_parameter('simulate_sign_after_sec').get_parameter_value().double_value
        # if simulate_after > 0.0:
        #     self.get_logger().info(f"Simulation: Schild wird nach {simulate_after:.1f}s als erkannt gesetzt.")
        #     self.create_timer(simulate_after, self._simulate_sign_once)

        # self.get_logger().info("SignDoorNavigator gestartet.")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def enable_cb(self, msg: Bool):
        self.enabled = msg.data
        self.get_logger().info(f"[DriveInBox] enabled={self.enabled} (von mission3_bt)")

    def scan_cb(self, msg: LaserScan):
        if not self.enabled:
            return
        self.latest_scan = msg

    # def sign_cb(self, msg: Bool):
    #     if not self.enabled:
    #         return

    #     if msg.data and self.tunnel_counter>5:
    #         self.get_logger().info("Schild erkannt (Topic) -> starte Tür-/Lücken-Suche.")
    #         self.sign_seen = True
    #         self.try_compute_and_send_goal()

    # def _simulate_sign_once(self):
    #     # Timer feuert einmal, dann nicht mehr
    #     if not self.enabled:
    #         return
    #     if not self.sign_seen:
    #         self.get_logger().info("Simuliere Schild-Erkennung (Hilfsvariable).")
    #         self.sign_seen = True
    #         self.try_compute_and_send_goal()

    def sign_detections_callback(self, msg: Detection2DArray):
        """
        React to YOLO sign detections while in lane_following mode.

        We look at classes 'left', 'right', 'stop' and take the one with the
        largest bounding-box area. If its area exceeds sign_area_threshold,
        we trigger a short-lived sign_state.
        """

        if not self.enabled:
            self.get_logger().info(f"[sign_detection_callback] self.enabled: {self.enabled}")
            return
        elif self.sign_seen:
            self.get_logger().info(f"[sign_detection_callback] self.sign_seen: {self.sign_seen}")
            return
        elif self.latest_scan is None:
            self.get_logger().info(f"[sign_detection_callback] self.latest_scan: {self.latest_scan}")
            return

        # self.get_logger().info(f"[sign_detection_callback] self.enabled: {self.enabled} | self.sign_seen {self.sign_seen}")

        best_class = None
        best_area = 0.0

        for det in msg.detections:
            if not det.results:
                continue
            class_id = det.results[0].hypothesis.class_id
            if class_id not in ('tunnel'):
                continue

            w = det.bbox.size_x
            h = det.bbox.size_y
            area = float(w * h)

            if area > best_area:
                best_area = area
                best_class = class_id

        self.get_logger().info(f"[sign_detection_callback] best_class: {best_class}")
        if best_class is None:
            return

        # Require minimum size (close sign)
        self.get_logger().info(f"[sign_detection_callback] best_area: {best_area}")
        if best_area < self.sign_area_threshold:
            return

        # (now is already defined above)
        if best_class == 'tunnel':
            self.tunnel_counter += 1

        self.get_logger().info(f"[sign_detection_callback] self.tunnel_counter: {self.tunnel_counter}")

        if self.tunnel_counter > 4 and not self.sign_seen:
            self.get_logger().info("Schild erkannt (Topic) -> starte Tür-/Lücken-Suche.")
            self.sign_seen = True
            self.try_compute_and_send_goal()


    # -------------------------------------------------------------------------
    # Hauptlogik
    # -------------------------------------------------------------------------
    def try_compute_and_send_goal(self):
        if self.goal_sent:
            return
        if self.latest_scan is None:
            self.get_logger().warn("Noch kein LaserScan empfangen, kann Lücke nicht bestimmen.")
            self.sign_seen = False
            self.tunnel_counter = 0
            return

        # 1) Lücke im Lidar finden, die wie eine Tür aussieht
        success, door_angle_rel = self.find_opening_angle(self.latest_scan)
        if not success:
            self.get_logger().warn("Keine gültige Tür-Lücke im Scan gefunden.")
            self.sign_seen = False
            self.tunnel_counter = 0
            return

        # 2) Roboterpose im map-Frame holen
        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().warn("Kann Roboterpose nicht aus TF lesen.")
            self.sign_seen = False
            self.tunnel_counter = 0
            return

        (x_r, y_r, yaw_r) = pose

        # 3) Zielposition berechnen (kurz hinter der Wand)
        wall_dist = self.get_parameter('wall_distance_estimate').get_parameter_value().double_value
        inside_offset = self.get_parameter('inside_offset').get_parameter_value().double_value
        angle_scale = self.get_parameter('goal_angle_scale').get_parameter_value().double_value

        d_goal = wall_dist + inside_offset

        # Türwinkel nur optional berücksichtigen
        theta_world = yaw_r + angle_scale * door_angle_rel

        x_goal = x_r + d_goal * math.cos(theta_world)
        y_goal = y_r + d_goal * math.sin(theta_world)

        self.get_logger().info(
            f"Setze Ziel hinter Lücke: angle_rel={math.degrees(door_angle_rel):.1f}°, "
            f"angle_scale={angle_scale:.2f}, "
            f"goal=({x_goal:.2f}, {y_goal:.2f}) in map."
        )

        # 4) Nav2 Goal senden
        self.send_nav_goal(x_goal, y_goal, theta_world)
        self.sign_seen = False
        self.tunnel_counter = 0

    # -------------------------------------------------------------------------
    # Lücken- und Tür-Suche im LaserScan
    # -------------------------------------------------------------------------
    def find_opening_angle(self, scan: LaserScan) -> Tuple[bool, float]:
        """
        Suche in einem Winkelbereich vor dem Roboter nach der größten zusammenhängenden
        "Lücke" (Messwerte > Threshold) UND prüfe, ob links & rechts davon Wand ist.
        Gibt den mittleren Winkel dieser gültigen Tür-Lücke zurück.
        """
        door_search_deg = self.get_parameter('door_search_angle_deg').get_parameter_value().double_value
        door_search_rad = math.radians(door_search_deg)

        wall_dist = self.get_parameter('wall_distance_estimate').get_parameter_value().double_value
        open_factor = self.get_parameter('open_threshold_factor').get_parameter_value().double_value
        open_threshold = wall_dist * open_factor

        angle_min = scan.angle_min
        angle_max = scan.angle_max
        angle_inc = scan.angle_increment

        # Mapping Winkel -> Index
        def angle_to_index(angle: float) -> int:
            return int(round((angle - angle_min) / angle_inc))

        # Index-Bereich [idx_min, idx_max] für [-door_search_rad, +door_search_rad]
        idx_min = max(0, angle_to_index(-door_search_rad))
        idx_max = min(len(scan.ranges) - 1, angle_to_index(+door_search_rad))

        if idx_min >= idx_max:
            self.get_logger().warn("Ungültiger Indexbereich für Lücken-Suche.")
            return False, 0.0

        ranges = scan.ranges

        # Flag-Liste: True = "offen" (vermutete Lücke)
        is_open = []
        for i in range(idx_min, idx_max + 1):
            r = ranges[i]
            # ungültige oder sehr große Werte zählen als offen
            if math.isinf(r) or math.isnan(r):
                is_open.append(True)
            else:
                is_open.append(r > open_threshold)

        # größte zusammenhängende Sequenz von True finden
        best_start = None
        best_len = 0
        cur_start = None
        cur_len = 0

        for i, flag in enumerate(is_open):
            if flag:
                if cur_start is None:
                    cur_start = i
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if cur_start is not None and cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None
                cur_len = 0

        # Ende berücksichtigen
        if cur_start is not None and cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

        if best_start is None or best_len == 0:
            self.get_logger().warn("Keine Lücke im definierten Winkelbereich gefunden.")
            return False, 0.0

        # Mittelindex der größten Lücke
        best_mid = best_start + best_len // 2
        best_idx = idx_min + best_mid
        angle_at_idx = angle_min + best_idx * angle_inc

        self.get_logger().info(
            f"Lücke gefunden: index={best_idx}, angle={math.degrees(angle_at_idx):.1f}°."
        )

        # --- Türprüfung: links und rechts muss Wand im gewünschten Distanzbereich sein ---
        check_offset_deg = self.get_parameter('door_check_offset_deg').get_parameter_value().double_value
        check_offset = math.radians(check_offset_deg)

        def clamp_index(idx: int) -> Optional[int]:
            if 0 <= idx < len(scan.ranges):
                return idx
            return None

        idx_left = clamp_index(angle_to_index(angle_at_idx - check_offset))
        idx_right = clamp_index(angle_to_index(angle_at_idx + check_offset))

        if idx_left is None or idx_right is None:
            self.get_logger().warn("Türprüfung außerhalb Scanbereich.")
            return False, 0.0

        # Fensterbreite (Bogenlänge in m auf Wanddistanz)
        wall_window_m = self.get_parameter('door_wall_window_m').get_parameter_value().double_value
        min_wall_dist = self.get_parameter('min_wall_dist_m').get_parameter_value().double_value
        max_wall_dist = self.get_parameter('max_wall_dist_m').get_parameter_value().double_value

        if wall_dist > 0.0 and wall_window_m > 0.0:
            # Bogenwinkel, der ±wall_window_m an der Wand abdeckt
            window_angle = wall_window_m / wall_dist  # rad
            window_indices = max(1, int(round(window_angle / angle_inc)))
        else:
            window_indices = 0

        def min_range_in_window(center_idx: int) -> float:
            start = max(0, center_idx - window_indices)
            end = min(len(ranges) - 1, center_idx + window_indices)
            vals = [
                ranges[i]
                for i in range(start, end + 1)
                if not math.isinf(ranges[i]) and not math.isnan(ranges[i])
            ]
            if not vals:
                return math.inf
            return min(vals)

        r_left = min_range_in_window(idx_left)
        r_right = min_range_in_window(idx_right)

        self.get_logger().info(
            f"Türprüfung (Fenster): left={r_left:.2f} m, right={r_right:.2f} m "
            f"(Wall-Range={min_wall_dist:.2f}–{max_wall_dist:.2f} m, "
            f"window_indices=±{window_indices})"
        )

        # Wand gültig, wenn:
        #  - Distanzwert endlich
        #  - innerhalb des gewünschten Bereichs (z.B. 0.15–0.40 m)
        valid_left = (
            not math.isinf(r_left)
            and (min_wall_dist <= r_left <= max_wall_dist)
        )
        valid_right = (
            not math.isinf(r_right)
            and (min_wall_dist <= r_right <= max_wall_dist)
        )

        if valid_left and valid_right:
            self.get_logger().info("Symmetrische Lücke erkannt -> Gültiger Eingang!")
            return True, angle_at_idx
        else:
            self.get_logger().warn("Gefundene Lücke ist kein Eingang (links/rechts keine passende Wand).")
            return False, 0.0

    # -------------------------------------------------------------------------
    # TF / Pose
    # -------------------------------------------------------------------------
    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', now, timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"Fehler beim TF-Lookup map->base_link: {e}")
            return None

        x = trans.transform.translation.x
        y = trans.transform.translation.y
        q = trans.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        return x, y, yaw

    # -------------------------------------------------------------------------
    # Nav2-Goal
    # -------------------------------------------------------------------------
    def send_nav_goal(self, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 Action-Server 'navigate_to_pose' nicht verfügbar.")
            return

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        goal_msg.pose = pose

        self.get_logger().info("Sende Nav2-Goal hinter Lücke...")
        self.goal_sent = True

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_cb
        )
        send_future.add_done_callback(self.nav_goal_response_cb)

    def nav_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f"Nav2 Feedback: dist_remaining={fb.distance_remaining:.2f} m"
        )

    def nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2-Goal abgelehnt.")
            return

        self.get_logger().info("Nav2-Goal akzeptiert, warte auf Ergebnis...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_cb)

    def nav_result_cb(self, future):
        result = future.result().result
        status = future.result().status

        self.get_logger().info(f"Nav2 Ergebnis-Status: {status}")
        # 4 = SUCCEEDED bei Nav2
        if status == 4 and not self.exploration_started:
            self.get_logger().info("Ziel im Kasten erreicht -> starte Frontier-Exploration.")
            self.start_exploration()

    # -------------------------------------------------------------------------
    # Übergabe an Frontier-Node
    # -------------------------------------------------------------------------
    def start_exploration(self):
        msg = Bool()
        msg.data = True
        self.exploration_pub.publish(msg)
        self.exploration_started = True
        self.get_logger().info("Exploration-Trigger gesendet (start_exploration = True).")


def main(args=None):
    rclpy.init(args=args)
    node = DriveInBox()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
