#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_srvs.srv import Trigger


class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')

        # Parameter für Zielpose (x, y, yaw) im map-Frame
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)  # in Radiant

        # Action Client für Nav2
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Service, um die Navigation zu starten
        self._start_service = self.create_service(
            Trigger,
            'start_navigation',
            self.start_navigation_callback
        )

        self.get_logger().info('SimpleNavigator gestartet. Warte auf Service-Call /start_navigation.')

    def start_navigation_callback(self, request, response):
        """Callback für /start_navigation (std_srvs/Trigger)"""
        # Ziel aus Parametern lesen
        goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        goal_yaw = self.get_parameter('goal_yaw').get_parameter_value().double_value

        self.get_logger().info(
            f"Starte Navigation zu Ziel: x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw:.2f} rad"
        )

        # Warten, bis der Action Server von Nav2 bereit ist
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            msg = 'Action-Server navigate_to_pose nicht erreichbar!'
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        # PoseStamped im map-Frame bauen
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._create_pose_stamped(goal_x, goal_y, goal_yaw)

        # Goal senden
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )

        # Callback registrieren, um das Ergebnis zu behandeln
        send_goal_future.add_done_callback(self._goal_response_callback)

        response.success = True
        response.message = 'Navigation gestartet.'
        return response

    def _create_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # Yaw -> Quaternion (nur z-Drehung)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal wurde vom Nav2-Server abgelehnt.')
            return

        self.get_logger().info('Goal akzeptiert, warte auf Ergebnis...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        # result.result ist ein NavigateToPose_Result, du kannst hier noch mehr auswerten
        self.get_logger().info(f'Navigation beendet. Result code: {result.result}')

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Inhalt je nach Nav2-Version, meistens aktuelle Pose/Distance
        self.get_logger().debug(f'Feedback: {feedback}')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
