#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # --- Parameters ---
        # Topics for BEV lane masks
        self.declare_parameter('mask_white_topic', 'lane_bev/mask_white')
        self.declare_parameter('mask_yellow_topic', 'lane_bev/mask_yellow')

        # How far from the bottom (near robot) we scan, in pixels
        self.declare_parameter('scan_row_from_bottom_px', 40)
        self.declare_parameter('scan_band_height_px', 30)

        # Expected lane width in pixels (for fallback when only one lane is visible)
        self.declare_parameter('lane_width_px', 160)

        # PID gains (for steering)
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.2)

        # Speed limits
        self.declare_parameter('v_min', 0.05)
        self.declare_parameter('v_max', 0.20)

        # Max angular velocity (rad/s)
        self.declare_parameter('max_omega', 1.5)

        # Control loop rate (Hz)
        self.declare_parameter('control_rate_hz', 10.0)

        # Read parameters
        self.mask_white_topic = self.get_parameter('mask_white_topic').get_parameter_value().string_value
        self.mask_yellow_topic = self.get_parameter('mask_yellow_topic').get_parameter_value().string_value

        self.scan_row_from_bottom_px = int(
            self.get_parameter('scan_row_from_bottom_px').get_parameter_value().integer_value
        )
        self.scan_band_height_px = int(
            self.get_parameter('scan_band_height_px').get_parameter_value().integer_value
        )

        self.lane_width_px = int(
            self.get_parameter('lane_width_px').get_parameter_value().integer_value
        )

        self.kp = float(self.get_parameter('kp').get_parameter_value().double_value)
        self.ki = float(self.get_parameter('ki').get_parameter_value().double_value)
        self.kd = float(self.get_parameter('kd').get_parameter_value().double_value)

        self.v_min = float(self.get_parameter('v_min').get_parameter_value().double_value)
        self.v_max = float(self.get_parameter('v_max').get_parameter_value().double_value)

        self.max_omega = float(self.get_parameter('max_omega').get_parameter_value().double_value)

        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').get_parameter_value().double_value
        )

        # --- State ---
        self.bridge = CvBridge()

        self.latest_white_mask = None  # np.ndarray mono8
        self.latest_yellow_mask = None  # np.ndarray mono8

        # PID state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()

        # --- QoS for sensor data ---
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscriptions ---
        self.white_sub = self.create_subscription(
            Image,
            self.mask_white_topic,
            self.white_mask_callback,
            sensor_qos
        )

        self.yellow_sub = self.create_subscription(
            Image,
            self.mask_yellow_topic,
            self.yellow_mask_callback,
            sensor_qos
        )

        # --- Publisher for cmd_vel ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Control loop timer ---
        period = 1.0 / self.control_rate_hz if self.control_rate_hz > 0 else 0.1
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f"lane_follower_node started. Subscribing to '{self.mask_white_topic}' and "
            f"'{self.mask_yellow_topic}', publishing /cmd_vel"
        )

    # --- Callbacks for masks -------------------------------------------------

    def white_mask_callback(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.latest_white_mask = mask
        except Exception as e:
            self.get_logger().error(f'Error converting white mask: {e}')

    def yellow_mask_callback(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.latest_yellow_mask = mask
        except Exception as e:
            self.get_logger().error(f'Error converting yellow mask: {e}')

    # --- Core lane center extraction -----------------------------------------

    def compute_lane_center_pixel(self):
        """
        Returns lane center x-position in pixels (float) or None if no lane.
        Uses a horizontal band near the bottom of the BEV masks.
        """

        if self.latest_white_mask is None and self.latest_yellow_mask is None:
            return None

        # Use whichever mask we have to determine image size
        if self.latest_white_mask is not None:
            mask_ref = self.latest_white_mask
        else:
            mask_ref = self.latest_yellow_mask

        H, W = mask_ref.shape[:2]

        # Define scanning band near bottom
        row_end = H - 1 - self.scan_row_from_bottom_px
        row_end = max(0, min(H - 1, row_end))
        row_start = max(0, row_end - self.scan_band_height_px)

        # Helper: extract lane x position from a mask in that band
        def lane_column_from_mask(mask):
            if mask is None:
                return None
            band = mask[row_start:row_end + 1, :]  # [rows, cols]
            ys, xs = np.where(band > 0)
            if len(xs) == 0:
                return None
            return float(np.mean(xs))  # average column

        # Right lane (white), left lane (yellow)
        u_right = lane_column_from_mask(self.latest_white_mask)
        u_left = lane_column_from_mask(self.latest_yellow_mask)

        # Fallback logic similar in spirit to your Duckie code:
        # - both lanes: center = midpoint
        # - only right: shift left by half lane width
        # - only left:  shift right by half lane width

        if u_left is not None and u_right is not None:
            # Ensure left < right (if swapped, fix)
            if u_left > u_right:
                u_left, u_right = u_right, u_left
            center_u = 0.5 * (u_left + u_right)
            return center_u

        lane_width_px = self.lane_width_px

        if u_right is not None:
            # Only right lane seen -> center is to the left
            center_u = u_right - lane_width_px * 0.5
            return center_u

        if u_left is not None:
            # Only left lane seen -> center is to the right
            center_u = u_left + lane_width_px * 0.5
            return center_u

        # No lane seen
        return None

    # --- Control loop --------------------------------------------------------

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = 1e-3
        self.last_time = now

        center_u = self.compute_lane_center_pixel()

        twist = Twist()

        if center_u is None:
            # No lane -> stop safely and reset controller
            self.integral = 0.0
            self.last_error = 0.0
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            # self.get_logger().debug("No lane detected -> stopping")
            return

        # Use image width from the last mask
        if self.latest_white_mask is not None:
            H, W = self.latest_white_mask.shape[:2]
        else:
            H, W = self.latest_yellow_mask.shape[:2]

        image_width = float(W)
        image_center = image_width / 2.0

        # error: positive if target is left of center (-> turn left)
        error = (image_center - center_u) / image_center  # normalize approx. [-1, 1]

        # PID
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0.0 else 0.0
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Clamp omega
        if omega > self.max_omega:
            omega = self.max_omega
        elif omega < -self.max_omega:
            omega = -self.max_omega

        # Forward velocity: slow down on large steering
        error_abs = min(abs(error), 1.0)
        v = self.v_max - (self.v_max - self.v_min) * error_abs

        twist.linear.x = v
        twist.angular.z = omega

        self.cmd_pub.publish(twist)

        # Uncomment for debugging if needed
        # self.get_logger().info(
        #     f"lane_center_px={center_u:.1f}, err={error:.3f}, v={v:.3f}, omega={omega:.3f}"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        for _ in range(5):
            node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
