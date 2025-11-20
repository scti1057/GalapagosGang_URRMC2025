#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from tf2_geometry_msgs import do_transform_pose
from rclpy.time import Time

from rclpy.duration import Duration
from visualization_msgs.msg import Marker


class RedSignNavClient(Node):
    """
    Node, die die Pose eines roten Schilds aus /red_sign_pose nimmt,
    in den map-Frame transformiert und als NavigateToPose-Goal an Nav2 schickt.
    Die Zielposition wird anhand neuer, besserer Beobachtungen verfeinert.
    """

    def __init__(self):
        super().__init__('red_sign_nav_client')

        # --- Parameter ---
        self.declare_parameter('enable_topic', '/red_sign_enabled')
        self.declare_parameter('pose_topic', '/red_sign_pose')
        self.declare_parameter('nav_action_name', 'navigate_to_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        # Wie viele Beobachtungen brauchen wir, bevor wir das erste Goal setzen?
        self.declare_parameter('min_observations_initial', 5)
        # Maximale Anzahl Samples, die wir für den gleitenden Mittelwert halten
        self.declare_parameter('max_samples', 20)
        # Minimum-Intervall zwischen möglichen Goal-Updates (Refinement)
        self.declare_parameter('refine_interval', 2.0)
        # Thresholds, ab wann wir das Goal tatsächlich updaten
        self.declare_parameter('pos_refine_threshold', 0.2)  # m
        self.declare_parameter('yaw_refine_threshold', 0.2)  # rad
        # Schild-Pose verwerfen, wenn länger als timeout nicht gesehen
        self.declare_parameter('sign_pose_timeout', 1.0)     # s
        # Abstand vor dem Schild, an dem der Roboter stoppen soll
        self.declare_parameter('stand_off_distance', 0.5)    # m
        # Minimale Distanz, die wir als "goal_distance" verwenden
        self.declare_parameter('min_goal_distance', 0.2)     # m

        enable_topic = self.get_parameter('enable_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.nav_action_name = self.get_parameter('nav_action_name').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.min_observations_initial = self.get_parameter(
            'min_observations_initial').get_parameter_value().integer_value
        self.max_samples = self.get_parameter('max_samples').get_parameter_value().integer_value
        self.refine_interval = self.get_parameter('refine_interval').get_parameter_value().double_value
        self.pos_refine_threshold = self.get_parameter(
            'pos_refine_threshold').get_parameter_value().double_value
        self.yaw_refine_threshold = self.get_parameter(
            'yaw_refine_threshold').get_parameter_value().double_value
        self.sign_pose_timeout = self.get_parameter('sign_pose_timeout').get_parameter_value().double_value
        self.stand_off_distance = self.get_parameter('stand_off_distance').get_parameter_value().double_value
        self.min_goal_distance = self.get_parameter('min_goal_distance').get_parameter_value().double_value

        # --- TF Buffer/Listener ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Action Client für Nav2 ---
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        # --- Zustand ---
        self.enabled: bool = False

        # Beobachtungen des Schilds im map-Frame (Liste von (x, y))
        self.sign_samples: list[tuple[float, float]] = []
        self.last_sample_time: Optional[float] = None

        # Aktuell bestgeschätzte Schildposition im map-Frame
        self.best_sign_x: Optional[float] = None
        self.best_sign_y: Optional[float] = None

        # Info über das zuletzt gesendete Nav2-Goal
        self.active_goal_handle = None
        self.last_goal_x: Optional[float] = None
        self.last_goal_y: Optional[float] = None
        self.last_goal_yaw: Optional[float] = None
        self.last_goal_time: Optional[float] = None

        # --- Subscriber ---
        self.enable_sub = self.create_subscription(
            Bool,
            enable_topic,
            self.enable_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.sign_pose_callback,
            10
        )

        # --- Marker Publisher für RViz ---
        self.sign_marker_pub = self.create_publisher(
            Marker,
            'red_sign_marker',   # Topic-Name
            10
        )
        self.goal_marker_pub = self.create_publisher(
            Marker,
            'red_sign_goal_marker',   # Topic-Name
            10
        )


        # --- Timer für Refine-Loop ---
        self.refine_timer = self.create_timer(
            self.refine_interval,
            self.refine_loop
        )

        self.get_logger().info(
            f"RedSignNavClient gestartet. enable_topic={enable_topic}, pose_topic={pose_topic}, "
            f"nav_action_name={self.nav_action_name}, map_frame={self.map_frame}, base_frame={self.base_frame}"
        )

    # ================= Callbacks =================

    def enable_callback(self, msg: Bool):
        previous = self.enabled
        self.enabled = bool(msg.data)

        if self.enabled and not previous:
            self.get_logger().info("Red sign navigation ENABLED: beginne Schild-Tracking.")
            # State zurücksetzen
            self.sign_samples.clear()
            self.last_sample_time = None
            self.best_sign_x = None
            self.best_sign_y = None
            if self.active_goal_handle is not None:
                self.cancel_active_goal()
        elif not self.enabled and previous:
            self.get_logger().info("Red sign navigation DISABLED: cancel Nav2 goal.")
            if self.active_goal_handle is not None:
                self.cancel_active_goal()

    def sign_pose_callback(self, msg: PoseStamped):
        """Schild-Pose im base_link-Frame -> in map-Frame transformieren & als Sample speichern."""
        if not self.enabled:
            return

        # Frame aus der Nachricht, sonst Fallback auf base_frame
        source_frame = msg.header.frame_id if msg.header.frame_id else self.base_frame

        try:
            # Transform von source_frame -> map: nehme aktuellen TF-Zeitpunkt (sim time)
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,     # Ziel-Frame
                source_frame,       # Quell-Frame (z.B. base_link)
                Time()              # "jetzt" in /use_sim_time
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup {self.map_frame}->{source_frame} fehlgeschlagen: {ex}"
            )
            return

        # WICHTIG: Nur die Pose transformieren, nicht PoseStamped
        pose_in_map = do_transform_pose(msg.pose, transform)  # -> geometry_msgs/Pose

        sx = pose_in_map.position.x
        sy = pose_in_map.position.y

        # Sample hinzufügen und Mittelwert bilden
        self.sign_samples.append((sx, sy))
        if len(self.sign_samples) > self.max_samples:
            self.sign_samples.pop(0)

        self.last_sample_time = self.get_clock().now().nanoseconds / 1e9

        xs = [p[0] for p in self.sign_samples]
        ys = [p[1] for p in self.sign_samples]
        self.best_sign_x = sum(xs) / len(xs)
        self.best_sign_y = sum(ys) / len(ys)

        self.publish_sign_marker()


    def publish_sign_marker(self):
        """Publiziert einen roten Marker an der aktuell geschätzten Schild-Position im map-Frame."""
        if self.best_sign_x is None or self.best_sign_y is None:
            return

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "red_sign"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = self.best_sign_x
        marker.pose.position.y = self.best_sign_y
        marker.pose.position.z = 0.5  # etwas über dem Boden

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 0  # 0 = unendlich, wird bei jedem Aufruf überschrieben

        self.sign_marker_pub.publish(marker)


    def publish_goal_marker(self, x: float, y: float, yaw: float):
        """Publiziert einen Marker an der Nav2-Goal-Position vor dem Schild."""
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "red_sign_goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05  # knapp über Boden

        # Yaw in Quaternion
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        # Pfeilgröße
        marker.scale.x = 0.4  # Länge
        marker.scale.y = 0.1  # Breite
        marker.scale.z = 0.1  # Höhe

        # z.B. grün für das Ziel
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.goal_marker_pub.publish(marker)


    # ================= Refine-Loop =================

    def refine_loop(self):
        """
        Wird periodisch aufgerufen.
        - Wenn enabled und genügend Beobachtungen vorhanden sind:
            - initiales Goal senden
            - oder bestehendes Goal verfeinern, wenn sich die Schildposition deutlich geändert hat
        """
        if not self.enabled:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        # Zu lange keine neuen Samples? Dann machen wir nichts.
        if self.last_sample_time is None or (now - self.last_sample_time) > self.sign_pose_timeout:
            self.get_logger().warn("Red sign navigation: keine aktuellen Schild-Observations.")
            return

        if self.best_sign_x is None or self.best_sign_y is None:
            return

        # Noch nicht genug Beobachtungen für das erste Goal?
        if len(self.sign_samples) < self.min_observations_initial and self.active_goal_handle is None:
            self.get_logger().info_throttle(
                5.0,
                "Sammle Schild-Observations (noch kein Goal geschickt)."
            )
            return

        # Aktuelle Roboterpose im map-Frame holen
        try:
            tf_map_base = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup map->{self.base_frame} fehlgeschlagen: {ex}")
            return

        rx = tf_map_base.transform.translation.x
        ry = tf_map_base.transform.translation.y

        # Vektor Roboter -> Schild (beste Schätzung)
        dx = self.best_sign_x - rx
        dy = self.best_sign_y - ry
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            self.get_logger().warn("Schildposition fast auf Roboterposition – ignoriere.")
            return

        yaw_to_sign = math.atan2(dy, dx)

        # Zielpunkt vor dem Schild (stand_off_distance)
        stand_off = self.stand_off_distance
        goal_distance = max(distance - stand_off, self.min_goal_distance)

        goal_x = rx + goal_distance * math.cos(yaw_to_sign)
        goal_y = ry + goal_distance * math.sin(yaw_to_sign)
        goal_yaw = yaw_to_sign

        # --- Erstes Goal vs. Refinement ---
        if self.active_goal_handle is None:
            # Erstes Goal schicken
            self.get_logger().info(
                f"Sende erstes Nav2-Goal vor dem Schild: "
                f"x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw:.2f} (dist={distance:.2f})"
            )
            self.send_new_goal(goal_x, goal_y, goal_yaw)
            self.last_goal_x = goal_x
            self.last_goal_y = goal_y
            self.last_goal_yaw = goal_yaw
            self.last_goal_time = now
        else:
            # Prüfen, ob wir verfeinern sollten
            if self.last_goal_x is None or self.last_goal_y is None or self.last_goal_yaw is None:
                return

            pos_diff = math.hypot(goal_x - self.last_goal_x, goal_y - self.last_goal_y)
            yaw_diff = abs(self.normalize_angle(goal_yaw - self.last_goal_yaw))

            # Nicht zu häufig updaten
            if self.last_goal_time is not None and (now - self.last_goal_time) < self.refine_interval:
                return

            # Wenn wir schon sehr nah am Schild sind, kein Refinement mehr
            if distance <= stand_off + 0.2:
                return

            if pos_diff > self.pos_refine_threshold or yaw_diff > self.yaw_refine_threshold:
                self.get_logger().info(
                    f"Verfeinere Nav2-Goal: Δpos={pos_diff:.2f} m, Δyaw={yaw_diff:.2f} rad"
                )
                self.cancel_active_goal()
                self.send_new_goal(goal_x, goal_y, goal_yaw)
                self.last_goal_x = goal_x
                self.last_goal_y = goal_y
                self.last_goal_yaw = goal_yaw
                self.last_goal_time = now

    # ================= Nav2-Goal-Handling =================

    def send_new_goal(self, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().error("Nav2 Action-Server navigate_to_pose nicht verfügbar.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        # Goal-Marker in RViz aktualisieren
        self.publish_goal_marker(x, y, yaw)

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2-Goal wurde abgelehnt.")
            self.active_goal_handle = None
            return

        self.get_logger().info("Nav2-Goal akzeptiert – Navigation zum roten Schild läuft.")
        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Nav2-Goal abgeschlossen. Result: {result}")
        self.active_goal_handle = None

    def feedback_callback(self, feedback_msg):
        # Optional: könnte Distanz zum Ziel etc. loggen
        pass

    def cancel_active_goal(self):
        if self.active_goal_handle is None:
            return

        self.get_logger().info("Cancel des aktuellen Nav2-Goals angefordert.")
        cancel_future = self.active_goal_handle.cancel_goal_async()

        def _cancel_done(_):
            self.get_logger().info("Nav2-Goal-Cancel abgeschlossen.")
            self.active_goal_handle = None

        cancel_future.add_done_callback(_cancel_done)

    # ================= Utilities =================

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wickelt einen Winkel in den Bereich [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = RedSignNavClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
