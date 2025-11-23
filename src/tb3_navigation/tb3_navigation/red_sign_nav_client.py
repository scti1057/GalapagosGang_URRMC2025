#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose


class RedSignNavClient(Node):
    def __init__(self):
        super().__init__('red_sign_nav_client')

        # --- Parameter ---
        # Nav2-Action-Name (bei Turtlebot3-Nav2 normalerweise: 'navigate_to_pose')
        self.declare_parameter('nav_action_name', 'navigate_to_pose')

        # Topic, auf dem der Localizer die Schild-Pose publish't
        self.declare_parameter('target_pose_topic', '/red_sign_goal_pose')

        nav_action_name = self.get_parameter('nav_action_name').get_parameter_value().string_value

        target_pose_topic = self.get_parameter('target_pose_topic').get_parameter_value().string_value

        # ActionClient für Nav2
        self._action_client = ActionClient(self, NavigateToPose, nav_action_name)

        # letzte empfangene Zielpose
        self.latest_target_pose: Optional[PoseStamped] = None

        # Flags
        self.pose_received = False   # Haben wir schon eine Pose bekommen?
        self.goal_sent = False       # Haben wir schon ein Nav2-Goal geschickt?

        # Subscriber für die Schild-Pose
        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            target_pose_topic,
            self.target_pose_callback,
            10
        )

        # Timer: läuft z.B. mit 10 Hz und steuert Bewegung + Nav2-Goal
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.last_feedback = None

        self.last_log_info = (
            f"RedSignNavClient gestartet.\n"
            f"  Action-Server:      {nav_action_name}\n"
            f"  Target-Pose-Topic:  {target_pose_topic}\n"
        )
        self.get_logger().info(self.last_log_info)

    # --- Callback: neue Zielpose vom Localizer ---
    def target_pose_callback(self, msg: PoseStamped):
        self.latest_target_pose = msg
        self.pose_received = True
        last_log_info = (
            f"Neue Schild-Pose empfangen: frame={msg.header.frame_id}, "
            f"x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
        )
        if self.last_log_info != last_log_info:
            self.get_logger().info(last_log_info)
            self.last_log_info = last_log_info

    # --- Timer: steuert Vorwärtsfahren + Nav2-Goal ---
    def timer_callback(self):
        # Wenn wir schon ein Nav2-Goal geschickt haben, machen wir hier nichts mehr
        if self.goal_sent:
            return

        # Warten, bis Nav2-Action-Server verfügbar ist
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            last_log_info = ("Nav2-Action-Server noch nicht verfügbar, warte...")
            if self.last_log_info != last_log_info:
                self.get_logger().warn(last_log_info)
                self.last_log_info = last_log_info
            return

        # Goal auf Basis der zuletzt empfangenen Pose schicken
        if self.latest_target_pose is None:
            last_log_info = ("Pose-Flag gesetzt, aber latest_target_pose ist None?")
            if self.last_log_info != last_log_info:
                self.get_logger().warn(last_log_info)
                self.last_log_info = last_log_info
            return

        self.send_goal(self.latest_target_pose)
        self.goal_sent = True


    def send_goal(self, target_pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        last_log_info = (
            f"Sende Nav2-Goal zur Schild-Pose:\n"
            f"  frame={target_pose.header.frame_id}, "
            f"x={target_pose.pose.position.x:.3f}, "
            f"y={target_pose.pose.position.y:.3f}"
        )
        if self.last_log_info != last_log_info:
            self.get_logger().warn(last_log_info)
            self.last_log_info = last_log_info

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    # --- Action-Callbacks ---

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            last_log_info = ('Nav2-Goal wurde abgelehnt!')
            if self.last_log_info != last_log_info:
                self.get_logger().error(last_log_info)
                self.last_log_info = last_log_info
            return


        last_log_info = ('Nav2-Goal akzeptiert, warte auf Ergebnis...')
        if self.last_log_info != last_log_info:
            self.get_logger().info(last_log_info)
            self.last_log_info = last_log_info
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        last_log_info = feedback_msg.feedback
        if self.last_log_info != last_log_info:
            self.get_logger().info(
                f"Nav2-Feedback: Distanz zum Ziel: {last_log_info.distance_remaining:.2f} m"
            )
            self.last_log_info = last_log_info

    def result_callback(self, future):
        result = future.result().result
        last_log_info = (f"Nav2-Result: Status={result.result}")
        if self.last_log_info != last_log_info:
            self.get_logger().info(last_log_info)
            self.last_log_info = last_log_info
        self.get_logger().info(last_log_info)
        # Hier könntest du später noch Logik einbauen (z.B. neuen Modus starten)


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
