#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frontier + Exit-Explorer für ROS2 Humble + Nav2 + SLAM (Cartographer).

Szenario:
- Turtlebot3 Burger in "Kasten" (ca. 1x1 m) mit Wänden + Säulen.
- Eingang = späterer Ausgang (rechteckige Öffnung).
- Aufgabe: Ausgang finden und hinausfahren, ohne Hindernisse zu touchieren.
- Nur Lidar, kein extra Kamera.

Strategie:
1) KICKSTART: zu Beginn ein erstes Nav2-Ziel ca. 5 cm vor dem Roboter setzen,
   damit Cartographer Free-Zellen erzeugt.
2) Frontier-Exploration im Kasten, dabei:
   - Startbereich (Eingang) mit Radius `entrance_radius` meiden
     => so, als wären dort hohe Kosten (kein Frontier-Ziel).
3) Parallel Lidar auswerten:
   - Merken: bisher maximal gesehene Distanz `max_scan_range_seen` im Kasten.
   - Wenn später eine Messung deutlich größer ist (>= max_scan_range_seen + exit_range_margin
     UND >= exit_min_range), wird dies als "Exit-Öffnung" interpretiert.
   - Aus dieser Richtung wird ein Exit-Goal berechnet und mit Nav2 angefahren.
4) Zusätzlich Exit-Abbruchkriterium: Abstand zur Startpose > `exit_distance` -> Exploration beendet.
"""

import math
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data


import tf2_ros
from tf2_ros import TransformException


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extrahiere Yaw (Rotation um z) aus Quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class FrontierExitExplorer(Node):
    def __init__(self):
        super().__init__("frontier_exit_explorer")

        # ---------------- Parameter ----------------
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("entrance_radius", 0.25)          # ~ "hohe Kosten" um Start
        self.declare_parameter("exit_distance", 2.0)             # Abstand von Startpose, wann "draußen"
        self.declare_parameter("min_frontier_size", 1)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("free_max", 100)                   # <= free_max => free
        self.declare_parameter("min_motion_for_frontier", 0.0)  # m
        self.declare_parameter("project_offset", 0.2)            # m
        # Kickstart-Konfiguration (ca. 5 cm vorwärts als Nav2-Goal)
        self.declare_parameter("kickstart_distance", 0.1)
        # Exit-Erkennung über Lidar
        self.declare_parameter("exit_detection_enabled", True)
        self.declare_parameter("exit_range_margin", 0.5)         # m über bisherigem Max -> Exit
        self.declare_parameter("exit_min_range", 1.5)            # absolute Schwelle (im 1x1-Kasten max ~1 m)
        self.declare_parameter("exit_activation_distance", 0.4)  # z.B. 40 cm vom Start weg


        # Parameterwerte holen
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.entrance_radius = self.get_parameter("entrance_radius").get_parameter_value().double_value
        self.exit_distance = self.get_parameter("exit_distance").get_parameter_value().double_value
        self.min_frontier_size = self.get_parameter("min_frontier_size").get_parameter_value().integer_value
        self.global_frame = self.get_parameter("global_frame").get_parameter_value().string_value
        self.robot_base_frame = self.get_parameter("robot_base_frame").get_parameter_value().string_value
        self.free_max = self.get_parameter("free_max").get_parameter_value().integer_value
        self.min_motion_for_frontier = self.get_parameter("min_motion_for_frontier").get_parameter_value().double_value
        self.project_offset = self.get_parameter("project_offset").get_parameter_value().double_value
        self.kickstart_distance = self.get_parameter("kickstart_distance").get_parameter_value().double_value
        self.exit_detection_enabled = self.get_parameter("exit_detection_enabled").get_parameter_value().bool_value
        self.exit_range_margin = self.get_parameter("exit_range_margin").get_parameter_value().double_value
        self.exit_min_range = self.get_parameter("exit_min_range").get_parameter_value().double_value
        self.exit_activation_distance = self.get_parameter("exit_activation_distance").get_parameter_value().double_value


        # ---------------- State ----------------
        self.map: Optional[OccupancyGrid] = None
        self.map_np = None
        self.map_info = None
        self.current_pose: Optional[Tuple[float, float]] = None
        self.current_yaw: Optional[float] = None
        self.initial_pose: Optional[Tuple[float, float]] = None
        self.initial_yaw: Optional[float] = None 

        self.last_pose_for_frontier: Optional[Tuple[float, float]] = None

        # Map-Update-Tracking
        self.map_update_count: int = 0
        self.last_map_update_count: int = -1

        self.navigating: bool = False
        self.current_goal_point: Optional[Tuple[float, float]] = None
        self.visited_frontiers: List[Tuple[float, float]] = []

        self._first_map_info_logged = False

        # Pose-Quelle: Topic vs. TF
        self.pose_from_topic: bool = False

        # Kickstart-Flags
        self.kickstart_completed = False
        self.kickstart_goal_sent = False

        # Lidar / Exit-Erkennung
        self.last_scan: Optional[LaserScan] = None
        self.max_scan_range_seen: float = 0.0
        self.exit_detected: bool = False
        self.exit_goal_sent: bool = False
        self.exit_goal_point: Optional[Tuple[float, float]] = None

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- Subscriptions ----------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.pose_topic, self.pose_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )

        # ---------------- Nav2 Action Client ----------------
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ---------------- Timer ----------------
        self.timer = self.create_timer(2.0, self.timer_callback)

        self.get_logger().info("FrontierExitExplorer gestartet.")
        self.get_logger().info(
            f"pose_topic={self.pose_topic}, scan_topic={self.scan_topic}, "
            f"entrance_radius={self.entrance_radius}, exit_distance={self.exit_distance}, "
            f"min_frontier_size={self.min_frontier_size}, global_frame={self.global_frame}, "
            f"robot_base_frame={self.robot_base_frame}, free_max={self.free_max}, "
            f"min_motion_for_frontier={self.min_motion_for_frontier}, project_offset={self.project_offset}, "
            f"kickstart_distance={self.kickstart_distance}, "
            f"exit_detection_enabled={self.exit_detection_enabled}, "
            f"exit_range_margin={self.exit_range_margin}, exit_min_range={self.exit_min_range}"
        )

        self.frontier_enabled = False
        self.mission3_done_pub = self.create_publisher(Bool, "mission3/done", 10)

        self.frontier_enable_sub = self.create_subscription(
            Bool,
            "mission3/frontier_enable",
            self.frontier_enable_cb,
            10,
)

    # ===================== Callbacks =====================

    def frontier_enable_cb(self, msg: Bool):
        self.frontier_enabled = msg.data
        self.get_logger().info(f"[FrontierExitExplorer] enabled={self.frontier_enabled} (von mission3_bt)")


    def map_callback(self, msg: OccupancyGrid):
        self.map = msg
        self.map_update_count += 1


        width = msg.info.width
        height = msg.info.height
        self.map_info = msg.info
        self.map_np = np.array(msg.data, dtype=np.int16).reshape((height, width))

        if not self._first_map_info_logged:
            self._first_map_info_logged = True
            self.get_logger().info(
                f"[DEBUG] Erste Map empfangen: size=({msg.info.width}x{msg.info.height}), "
                f"res={msg.info.resolution:.3f} m, "
                f"origin=({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})"
            )

    def is_world_free(self, x: float, y: float) -> bool:
        if self.map_np is None or self.map_info is None:
            # keine Info -> sei großzügig
            return True

        res = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        mx = int((x - origin_x) / res)
        my = int((y - origin_y) / res)

        if mx < 0 or my < 0 or mx >= self.map_info.width or my >= self.map_info.height:
            # außerhalb der Karte -> eher nicht nutzen
            return False

        val = self.map_np[my, mx]
        # UNKNOWN (-1) und FREE (0..free_max) erlauben, OCCUPIED (>free_max) verbieten
        if val > self.free_max:
            return False
        return True



    def pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self.current_pose = (x, y)
        self.current_yaw = yaw
        self.pose_from_topic = True

        self.get_logger().info(
            f"[DEBUG] Pose-Update von {self.pose_topic}: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°"
        )
        if self.initial_pose is None:
            self.initial_pose = (x, y)
            self.initial_yaw = yaw 
            self.get_logger().info(
                f"Initialpose (aus {self.pose_topic}) gesetzt: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°"
            )

    def update_pose_from_tf(self) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                rclpy.time.Time()
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

            self.current_pose = (x, y)
            self.current_yaw = yaw

            self.get_logger().info(
                f"[DEBUG] Pose aus TF: {self.global_frame}->{self.robot_base_frame}, "
                f"x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°"
            )
            if self.initial_pose is None:
                self.initial_pose = (x, y)
            self.initial_yaw = yaw 
            self.get_logger().info(
                f"Initialpose (aus {self.pose_topic}) gesetzt: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°"
            )
            return True
        except TransformException as ex:
            self.get_logger().info(
                f"[DEBUG] Pose weder aus Topic noch TF verfügbar: {ex}"
            )
            return False

    def scan_callback(self, msg: LaserScan):
        if not self.frontier_enabled:
            return
        
        self.last_scan = msg
        if not self.exit_detection_enabled:
            return
        
        if self.initial_pose is not None and self.current_pose is not None:
            dist_start = self.euclidean(self.current_pose, self.initial_pose)
            if dist_start < self.exit_activation_distance:
                # Noch zu nah am Eingang -> Eingang gilt als "zu"
                # Lidar-Daten nur für SLAM, aber nicht für Exit-Erkennung nutzen
                return

        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        if not np.any(valid):
            return

        max_r = float(np.max(ranges[valid]))
        prev_max = self.max_scan_range_seen

        # Exit schon erkannt? -> nichts mehr tun
        if self.exit_detected:
            # trotzdem Max-Wert updaten, falls du das fürs Debuggen willst
            if max_r > self.max_scan_range_seen:
                self.max_scan_range_seen = max_r
            return

        # Prüfen, ob jetzt eine "deutlich größere" Distanz auftritt
        if (
            max_r >= self.exit_min_range
            and max_r >= prev_max + self.exit_range_margin
        ):
            self.get_logger().info(
                f"[DEBUG] Exit-Kandidat: neuer großer Lidar-Wert: {max_r:.2f} m "
                f"(bisher_max={prev_max:.2f} m)"
            )
            exit_goal = self.compute_exit_goal_from_scan(msg, ranges)
            if exit_goal is not None:
                self.exit_detected = True
                self.exit_goal_point = exit_goal
                self.get_logger().info(
                    f"[DEBUG] Exit-Ziel aus Lidar bestimmt: {exit_goal}"
                )

        # Danach Max-Wert aktualisieren
        if max_r > self.max_scan_range_seen:
            self.max_scan_range_seen = max_r

    # ===================== Haupt-Logik =====================

    def timer_callback(self):
        if not self.frontier_enabled:
            #self.get_logger().info("[DEBUG] Frontier-Explorer noch deaktiviert (mission3/frontier_enable==False).")
            return
        self.get_logger().info("[DEBUG] Timer-Callback")

        if self.map is None:
            self.get_logger().info("[DEBUG] Map noch nicht empfangen.")
            return

        if not self.pose_from_topic:
            if not self.update_pose_from_tf():
                self.get_logger().info("[DEBUG] Keine Pose verfügbar – skip Timer-Iteration.")
                return
        else:
            if self.current_pose is None:
                self.get_logger().info("[DEBUG] current_pose None trotz Topic – skip.")
                return

        if self.initial_pose is None:
            self.get_logger().info("[DEBUG] Initialpose noch nicht gesetzt.")
            return

        # ========== Kickstart: erstes Nav2-Goal direkt vor den Roboter ==========
        if (not self.kickstart_completed) and (self.current_yaw is not None):
            rx, ry = self.current_pose
            yaw = self.current_yaw

            gx = rx + self.kickstart_distance * math.cos(yaw)
            gy = ry + self.kickstart_distance * math.sin(yaw)
            kick_goal = (gx, gy)

            self.get_logger().info(
                f"[DEBUG] Kickstart-Nav2-Goal: {kick_goal} "
                f"(dist={self.kickstart_distance:.3f} m vor dem Roboter)"
            )

            success = self.send_goal(kick_goal)
            if success:
                self.kickstart_completed = True
                self.kickstart_goal_sent = True
                return
            else:
                # Nav2 noch nicht bereit -> beim nächsten Timer nochmal probieren
                self.get_logger().warn("[DEBUG] Kickstart-Goal konnte nicht gesendet werden (Nav2 nicht bereit).")
                return
        # ========================================================================

        # Exit-Abstand: wenn weit genug weg von Start -> fertig
        dist_start = self.euclidean(self.current_pose, self.initial_pose)
        self.get_logger().info(
            f"[DEBUG] Distanz zur Startpose: {dist_start:.2f} m (Exit-Schwelle={self.exit_distance:.2f} m)"
        )
        if dist_start > self.exit_distance:
            self.get_logger().info(
                f"Exit erreicht (Abstand {dist_start:.2f} m). Exploration beendet."
            )
            self.navigating = False
            try:
                self.timer.cancel()
            except Exception as e:
                self.get_logger().warn(f"[DEBUG] Timer cancel fehlgeschlagen: {e}")
     
            done_msg = Bool()
            done_msg.data = True
            self.mission3_done_pub.publish(done_msg)
            self.get_logger().info("[FrontierExitExplorer] mission3/done = True publiziert.")
            return

        # Wenn bereits ein Nav2-Goal läuft: warten
        if self.navigating:
            self.get_logger().info("[DEBUG] Noch am Navigieren – warte.")
            return

        # *** Exit-Goal hat Vorrang vor Frontiers ***
        if self.exit_detected and not self.exit_goal_sent and self.exit_goal_point is not None:
            self.get_logger().info("[DEBUG] Exit erkannt – sende Exit-Goal an Nav2.")
            self.send_goal(self.exit_goal_point)
            self.exit_goal_sent = True
            return

        # Frontier-Exploration nur, wenn kein Exit-Goal aktiv ist
        if self.last_map_update_count == self.map_update_count:
            self.get_logger().info(
                "[DEBUG] Map hat sich seit letztem Mal nicht geändert – skip."
            )
            return

        self.last_map_update_count = self.map_update_count

        if self.last_pose_for_frontier is not None:
            move = self.euclidean(self.current_pose, self.last_pose_for_frontier)
            self.get_logger().info(
                f"[DEBUG] Bewegung seit letzter Frontier-Suche: {move:.3f} m"
            )
            if move < self.min_motion_for_frontier:
                self.get_logger().info(
                    "[DEBUG] Zu wenig Bewegung – noch keine neue Frontier-Suche."
                )
                return

        self.last_pose_for_frontier = self.current_pose

        self.get_logger().info("[DEBUG] Berechne Frontiers...")
        frontiers = self.compute_frontiers(self.map)

        if not frontiers:
            self.get_logger().info(
                "Keine Frontiers gefunden – Karte (lokal) komplett oder noch zu leer."
            )
            return

        self.get_logger().info(
            f"[DEBUG] Roh-Frontiers (Cluster-Schwerpunkte): {len(frontiers)}"
        )

        self.get_logger().info(f"initial pose: {self.initial_pose}")
        candidates = []
        for p in frontiers:
            if self.is_near_entrance(p):
                self.get_logger().info(
                    f"[DEBUG] Frontier {p} verworfen (zu nah am Eingang, radius={self.entrance_radius})"
                )
                continue
            if self.is_in_entrance_halfspace(p):
                self.get_logger().info(
                    f"[DEBUG] Frontier {p} verworfen (liegt in verbotener Eingangshalbebene)."
                )
                continue
            if self.is_visited(p):
                self.get_logger().info(
                    f"[DEBUG] Frontier {p} verworfen (bereits besucht)."
                )
                continue
            if not self.is_world_free(p[0], p[1]):
                self.get_logger().info(
                    f"[DEBUG] Frontier {p} verworfen (liegt nicht in freier Zelle)."
                )
                continue
            if p[0] < self.initial_pose[0] or p[1] < self.initial_pose[1]:
                self.get_logger().info(
                    f"[DEBUG] Frontier {p} verworfen (liegt unter x=0 oder y=0)."
                )
                continue
            candidates.append(p)

        self.get_logger().info(
            f"[DEBUG] Kandidaten nach Filter: {len(candidates)}"
        )

        if not candidates:
            self.get_logger().info(
                "Nur Frontiers nahe Eingang oder bereits besucht – keine sinnvollen Ziele."
            )
            return

        goal_raw = max(candidates, key=lambda p: self.euclidean(self.current_pose, p))
        goal_proj = self.project_goal_inside_free_space(goal_raw, self.project_offset)

        self.get_logger().info(
            f"[DEBUG] Ziel roh: {goal_raw}, projiziert: {goal_proj}, "
            f"Distanz={self.euclidean(self.current_pose, goal_proj):.2f} m"
        )

        self.send_goal(goal_proj)

    # ===================== Frontier-Berechnung =====================

    def compute_frontiers(self, map_msg: OccupancyGrid) -> List[Tuple[float, float]]:
        width = map_msg.info.width
        height = map_msg.info.height
        res = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        data = np.array(map_msg.data, dtype=np.int16).reshape((height, width))

        UNKNOWN = -1
        FREE_MAX = self.free_max

        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]

        num_unknown = int(np.sum(data == UNKNOWN))
        num_free = int(np.sum((data >= 0) & (data <= FREE_MAX)))
        self.get_logger().info(
            f"[DEBUG] Map-Statistik: unknown={num_unknown}, free={num_free}, total={width * height}"
        )

        frontier_cells = []

        # ### NEU: Initialpose holen (für Quadranten-Filter)
        init_x = init_y = None
        if self.initial_pose is not None:
            init_x, init_y = self.initial_pose

        for y in range(height):
            for x in range(width):

                # ### NEU: Weltkoordinate dieser Zelle
                if init_x is not None:
                    wx = origin_x + (x + 0.5) * res
                    wy = origin_y + (y + 0.5) * res

                    # Nur Punkte im "positiven Quadranten" relativ zur Initialpose:
                    #   wx >= init_x  UND  wy >= init_y
                    if wx < init_x or wy < init_y:
                        # liegt "hinter" oder "links/unten" der Initialpose -> ignorieren
                        continue

                # nur UNKNOWN-Zellen als potentielle Frontier-Kandidaten
                # if data[y, x] != UNKNOWN:
                #     continue

                is_frontier = False
                for dx, dy in neighbors:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        val = data[ny, nx]
                        if 0 <= val <= FREE_MAX:
                            is_frontier = True
                            break
                if is_frontier:
                    frontier_cells.append((x, y))

        self.get_logger().info(
            f"[DEBUG] Frontier-Zellen (gefiltert im +x/+y-Bereich): {len(frontier_cells)}"
        )

        if not frontier_cells:
            return []

        frontier_set = set(frontier_cells)
        visited = set()
        clusters: List[List[Tuple[int, int]]] = []

        for cell in frontier_cells:
            if cell in visited:
                continue
            stack = [cell]
            cluster = []
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                if (cx, cy) not in frontier_set:
                    continue
                visited.add((cx, cy))
                cluster.append((cx, cy))
                for dx, dy in neighbors:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) in frontier_set and (nx, ny) not in visited:
                            stack.append((nx, ny))
            if len(cluster) >= self.min_frontier_size:
                clusters.append(cluster)

        self.get_logger().info(
            f"[DEBUG] Frontier-Cluster >= {self.min_frontier_size}: {len(clusters)}"
        )

        frontier_points_world: List[Tuple[float, float]] = []
        for cluster in clusters:
            mx = sum(c[0] for c in cluster) / len(cluster)
            my = sum(c[1] for c in cluster) / len(cluster)
            wx = origin_x + (mx + 0.5) * res
            wy = origin_y + (my + 0.5) * res
            frontier_points_world.append((wx, wy))

        self.get_logger().info(
            f"{len(frontier_points_world)} Frontier-Ziele (Cluster-Schwerpunkte, +x/+y)."
        )
        return frontier_points_world


    # ===================== Exit-Goal aus Lidar =====================

    def compute_exit_goal_from_scan(self, scan: LaserScan, ranges: np.ndarray) -> Optional[Tuple[float, float]]:
        if self.current_pose is None or self.current_yaw is None:
            self.get_logger().warn("[DEBUG] Keine Pose/Yaw vorhanden – kann Exit-Goal nicht berechnen.")
            return None

        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        if not np.any(valid):
            return None

        valid_indices = np.where(valid)[0]
        max_idx = valid_indices[int(np.argmax(ranges[valid]))]
        max_r = float(ranges[max_idx])

        angle_rel = scan.angle_min + max_idx * scan.angle_increment
        angle_global = self.current_yaw + angle_rel

        goal_dist = max_r - 0.3
        if goal_dist < 0.3:
            goal_dist = max_r * 0.8

        rx, ry = self.current_pose
        gx = rx + goal_dist * math.cos(angle_global)
        gy = ry + goal_dist * math.sin(angle_global)

        self.get_logger().info(
            f"[DEBUG] Exit-Richtung: idx={max_idx}, max_r={max_r:.2f} m, "
            f"rel_angle={math.degrees(angle_rel):.1f}°, global_angle={math.degrees(angle_global):.1f}°, "
            f"Exit-Goal=({gx:.2f}, {gy:.2f})"
        )
        return (gx, gy)

    # ===================== Nav2 / Ziele =====================

    def project_goal_inside_free_space(self, goal_point: Tuple[float, float], offset: float) -> Tuple[float, float]:
        if self.current_pose is None:
            return goal_point

        gx, gy = goal_point
        rx, ry = self.current_pose

        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        if dist < offset * 1.5:
            candidate = (gx, gy)
        else:
            scale = (dist - offset) / dist
            candidate = (rx + dx * scale, ry + dy * scale)

        # NEU: wenn der Punkt nicht frei ist, schrittweise Richtung Roboter zurück
        steps = 10
        cx, cy = candidate
        for i in range(steps):
            if self.is_world_free(cx, cy):
                if (cx, cy) != candidate:
                    self.get_logger().info(
                        f"[DEBUG] Goal leicht zurückprojiziert auf freie Zelle: ({cx:.2f}, {cy:.2f})"
                    )
                return (cx, cy)
            # 1/steps des Weges zurück Richtung Roboter
            cx = cx - dx / steps
            cy = cy - dy / steps

        # Wenn nichts frei gefunden: ursprüngliches Ziel zurückgeben (Nav2 darf entscheiden)
        self.get_logger().warn(
            f"[WARN] Konnte keine freie Zelle entlang Robot->Goal finden, benutze ursprüngliches Ziel {goal_point}"
        )
        return goal_point


    def send_goal(self, point: Tuple[float, float]) -> bool:
        x, y = point
        self.get_logger().info(f"Sende Navigationsziel: x={x:.2f}, y={y:.2f}")

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        goal.pose = pose
        self.current_goal_point = point

        self.get_logger().info("[DEBUG] Warte auf Nav2 Action-Server 'navigate_to_pose'...")
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 Action Server 'navigate_to_pose' nicht verfügbar.")
            self.navigating = False
            return False 

        self.get_logger().info("[DEBUG] Nav2-Server verfügbar, sende Goal...")
        send_future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        self.navigating = True
        return True


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Navigationsziel von Nav2 abgelehnt.")
            self.navigating = False
            return

        self.get_logger().info("Navigationsziel akzeptiert.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        status = result.status
        status_map = {
            0: "UNKNOWN",
            1: "ACCEPTED",
            2: "EXECUTING",
            3: "CANCELING",
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "ABORTED",
        }
        self.get_logger().info(
            f"Navigation beendet mit Status {status} ({status_map.get(status, 'UNDEFINED')})."
    )
        if self.current_goal_point is not None:
            self.get_logger().info(
                f"[DEBUG] Ziel {self.current_goal_point} als besucht markieren."
            )
            self.visited_frontiers.append(self.current_goal_point)
        self.navigating = False
        self.current_goal_point = None

    def feedback_callback(self, feedback_msg):
        try:
            dist_rem = feedback_msg.feedback.distance_remaining
            self.get_logger().info(
                f"[DEBUG] Nav2-Feedback: distance_remaining={dist_rem:.2f} m"
            )
        except Exception:
            pass

    # ===================== Helpers =====================

    @staticmethod
    def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def is_near_entrance(self, point: Tuple[float, float]) -> bool:
        if self.initial_pose is None:
            return False
        return self.euclidean(self.initial_pose, point) < self.entrance_radius

    def is_visited(self, point: Tuple[float, float], tol: float = 0.5) -> bool:
        return any(self.euclidean(vp, point) < tol for vp in self.visited_frontiers)
    
    def is_in_entrance_halfspace(self, point: Tuple[float, float]) -> bool:
        """
        True, wenn der Punkt 'hinter' der Startpose liegt, also in Richtung Eingang / draußen.

        Idee:
        - initial_yaw zeigt IN den Kasten.
        - Vektor dir_in = (cos(initial_yaw), sin(initial_yaw)).
        - Für Punkt p: v = p - initial_pose.
        - Wenn dot(v, dir_in) < 0  -> p liegt "hinter" der Start-Linie (Eingangsseite).
        """
        if self.initial_pose is None or self.initial_yaw is None:
            return False

        ix, iy = self.initial_pose
        px, py = point
        vx = px - ix
        vy = py - iy

        dir_in_x = math.cos(self.initial_yaw)
        dir_in_y = math.sin(self.initial_yaw)

        dot = vx * dir_in_x + vy * dir_in_y

        return dot < 0.0



def main(args=None):
    rclpy.init(args=args)
    node = FrontierExitExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
