#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Float32MultiArray


class RedSignNavClient(Node):
    def __init__(self):
        super().__init__('red_sign_nav_client')

        # --- Parameter ---
        # Nav2-Action-Name (bei Turtlebot3-Nav2 normalerweise: 'navigate_to_pose')
        self.declare_parameter('nav_action_name', 'navigate_to_pose')

        # Frame für das Ziel (erstmal map)
        self.declare_parameter('frame_id', 'map')

        # Geometrie der "zweiten Linie" (wie bei deiner Referenzlinie)
        self.declare_parameter('segment1_length', 0.31)   # 31 cm vor dem Roboter
        self.declare_parameter('segment2_length', 0.4)    # z.B. 0.5 m weiter

        # Orientation-Topic (gleich wie beim RedSignDetector & Referenz-Line-Node)
        self.declare_parameter('orient_topic', '/red_sign_orient')

        nav_action_name = self.get_parameter('nav_action_name').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.L1 = self.get_parameter('segment1_length').get_parameter_value().double_value
        self.L2 = self.get_parameter('segment2_length').get_parameter_value().double_value

        self.current_yaw_rad = None
        self.last_feedback = None

        orientation_topic = self.get_parameter('orient_topic').get_parameter_value().string_value

        # ActionClient für Nav2
        self._action_client = ActionClient(self, NavigateToPose, nav_action_name)

        # Subscriber für Orientation (Float32MultiArray: [yaw_rad, pitch_rad])
        self.orientation_sub = self.create_subscription(
            Float32MultiArray,
            orientation_topic,
            self.orientation_callback,
            10
        )

        # Flag: nur ein Goal schicken
        self.goal_sent = False

        # Timer: wiederholt prüfen, ob Server da ist und dann EIN Goal schicken
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.get_logger().info(
            f"RedSignNavClient gestartet.\n"
            f"  Action-Server: {nav_action_name}\n"
            f"  Orientation-Topic: {orientation_topic}\n"
        )

    # --- Orientation-Callback ---
    def orientation_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 1:
            self.current_yaw_rad = msg.data[0]  # yaw_rad
            # self.get_logger().info(
            #     f"Orientation erhalten: yaw_rad={self.current_yaw_rad:.3f} rad "
            #     f"({math.degrees(self.current_yaw_rad):.1f}°)"
            # )
        else:
            self.get_logger().warn(
                "Orientation-Message erhalten, aber msg.data ist zu kurz!"
            )

    # --- Timer: versucht EIN Nav-Goal zu schicken ---
    def timer_callback(self):
        if self.goal_sent:
            return

        if not self._action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn("Nav2-Action-Server noch nicht verfügbar...")
            return

        self.send_goal()
        self.get_logger().info("Goal geschickt!")
        self.goal_sent = True

    def send_goal(self):
        angle_rad = self.current_yaw_rad

        if not angle_rad:
            return

        # Punkt auf der zweiten Linie (wie in deiner Referenzlinie):
        # p0 = (0,0)
        # p1 = (L1, 0)
        # p2 = p1 + (L2 * cos(angle), L2 * sin(angle))
        goal_x = self.L1 + self.L2 * math.cos(angle_rad)
        goal_y =           self.L2 * math.sin(angle_rad)

        yaw_rad = angle_rad

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0

        # Yaw -> Quaternion (2D)
        pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(
            f"Sende Nav2-Goal auf zweiter Linie:\n"
            f"  x={goal_x:.3f}, y={goal_y:.3f}, yaw={math.degrees(yaw_rad):.1f}°"
        )

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    # --- Action-Callbacks ---

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2-Goal wurde abgelehnt!')
            return

        self.get_logger().info('Nav2-Goal akzeptiert, warte auf Ergebnis...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        if self.last_feedback != feedback:
            self.get_logger().info(
                f"Nav2-Feedback: Distanz zum Ziel: {feedback.distance_remaining:.2f} m"
            )
            self.last_feedback = feedback

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Nav2-Result: Status={result.result}")
        # Hier könntest du später weitere Logik einbauen


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
