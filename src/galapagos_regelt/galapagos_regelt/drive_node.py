#!/usr/bin/env python3

import os
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration


from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist



class DriveNode(Node):
    """
    PID controller that keeps the TurtleBot3 in the middle of the lane.

    Subscribes:
      - x_tar (Float64): target x position in image (pixels), from control_node

    Loads from drive.yaml:
      - pid.kp, pid.ki, pid.kd
      - speed.v_min, speed.v_max
      - limits.omega_max
      - image.width

    Publishes:
      - cmd_vel (geometry_msgs/Twist)
    """

    def __init__(self):
        super().__init__('drive_node')

        # Parameters
        self.declare_parameter('config_file', 'drive.yaml')
        self.declare_parameter('max_rate_hz', 20.0)  # control loop max rate
        # Timeout: if no x_tar is received for this long, stop the robot
        self.declare_parameter('x_tar_timeout', 0.3)  # seconds
        # Whether PalettingNode is directly controlling cmd_vel
        self.pal_control_active = False


        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate_hz = self.get_parameter('max_rate_hz').get_parameter_value().double_value
        self.x_tar_timeout = self.get_parameter('x_tar_timeout').get_parameter_value().double_value


        self._min_period = 1.0 / float(max_rate_hz)
        self._last_loop_time = self.get_clock().now()

        # Load YAML config
        pkg_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(pkg_share, 'config', config_file)
        self.get_logger().info(f'Using drive config: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        pid_cfg = self.conf.get('pid', {})
        self.kp = float(pid_cfg.get('kp', 8.0))
        self.ki = float(pid_cfg.get('ki', 0.1))
        self.kd = float(pid_cfg.get('kd', 0.4))

        speed_cfg = self.conf.get('speed', {})
        self.v_min = float(speed_cfg.get('v_min', 0.05))
        self.v_max = float(speed_cfg.get('v_max', 0.20))

        limits_cfg = self.conf.get('limits', {})
        self.omega_max = float(limits_cfg.get('omega_max', 3.0))

        image_cfg = self.conf.get('image', {})
        self.image_width = float(image_cfg.get('width', 640.0))
        self.image_center = self.image_width / 2.0

        self.get_logger().info(
            f'PID: kp={self.kp}, ki={self.ki}, kd={self.kd}; '
            f'v_min={self.v_min}, v_max={self.v_max}, omega_max={self.omega_max}; '
            f'image_width={self.image_width}'
        )

        # PID state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()

        # Latest x_tar
        self.latest_x_tar = None

        # Time of last received x_tar
        self.last_x_tar_time = None

        # Debug flag (could be parameterized later)
        self.debug = True

        # QoS: sensor-style, keep only last value
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber: x_tar from control_node
        self.sub_x_tar = self.create_subscription(
            Float64,
            'x_tar',
            self.x_tar_callback,
            sensor_qos
        )

        # Publisher: cmd_vel to TurtleBot3
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)

        # Control loop timer
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('DriveNode started.')

        self.sub_pal_active = self.create_subscription(
            Bool,
            'pal_control_active',
            self.pal_control_active_callback,
            sensor_qos
        )

    # === Callbacks ===

    def pal_control_active_callback(self, msg: Bool):
        active = bool(msg.data)

        # On entering override: reset PID integrator to avoid a big kick afterwards
        if active and not self.pal_control_active:
            self.integral = 0.0
            self.last_error = 0.0

        self.pal_control_active = active


    def x_tar_callback(self, msg: Float64):
        """Update target x position and timestamp."""
        self.latest_x_tar = msg.data
        self.last_x_tar_time = self.get_clock().now()

    # === Control logic ===

    def control_loop(self):
        """Main PID control loop, rate limited by max_rate_hz."""
        now = self.get_clock().now()
        elapsed_rate = (now - self._last_loop_time).nanoseconds / 1e9
        if elapsed_rate < self._min_period:
            return
        self._last_loop_time = now

        # If PalettingNode is directly commanding cmd_vel, stay completely quiet
        if self.pal_control_active:
            return

        # If we never received x_tar or it's too old, stop the robot
        if self.last_x_tar_time is None:
            self.latest_x_tar = None
            self.publish_stop()
            return

        age = (now - self.last_x_tar_time).nanoseconds / 1e9
        if age > self.x_tar_timeout:
            self.latest_x_tar = None
            # reset PID state
            self.integral = 0.0
            self.last_error = 0.0
            self.publish_stop()
            return

        # Compute normalized steering error
        center_pixel_x = self.latest_x_tar
        if center_pixel_x is None:
            self.publish_stop()
            return

        # --- NEW: deadband in pixel space ---
        pixel_error = self.image_center - center_pixel_x

        # e.g. 3 pixels deadband
        if abs(pixel_error) < 3.0:
            pixel_error = 0.0
            # kill any integral wobble when we're basically centered
            self.integral = 0.0
            self.last_error = 0.0

        # normalized error in [-1, 1]
        error = pixel_error / self.image_center

        # Timing for PID
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1e-6
        self.last_time = now

        # PID terms
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Clamp steering
        if omega > self.omega_max:
            omega = self.omega_max
        elif omega < -self.omega_max:
            omega = -self.omega_max

        # Forward velocity: slow down on large error
        error_abs = abs(error)
        if error_abs > 1.0:
            error_abs = 1.0
        v = self.v_max - (self.v_max - self.v_min) * error_abs
        if v < 0.0:
            v = 0.0

        if self.debug:
            self.get_logger().info(f"v: {v}")
            self.get_logger().info(f"omega: {omega}")
        # Publish Twist
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = omega
        self.pub_cmd_vel.publish(twist)

        # if self.debug:
        #     self.get_logger().info(
        #         f"[PID] e={error:.3f} P={self.kp*error:.3f} "
        #         f"I={self.ki*self.integral:.3f} D={self.kd*derivative:.3f} "
        #         f"omega={omega:.3f} v={v:.3f}"
        #     )

    def publish_stop(self):
        """Publish zero Twist (stop)."""
        if self.debug:
            self.get_logger().info("STOP")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)

    # === Shutdown handling ===

    def destroy_node(self):
        """Stop the robot when shutting down."""
        self.get_logger().info('DriveNode shutting down, stopping robot.')
        for _ in range(5):
            self.publish_stop()
            time.sleep(0.1)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
