#!/usr/bin/env python3

import os
import math
import yaml
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory


class YawNode(Node):
    """
    Simple yaw controller: rotate in place until yaw_init ~= yaw_tar.

    Subscribes:
      - yaw_init (Float64): current yaw [deg], in range (-180, 180]
      - yaw_tar  (Float64): target yaw [deg], in same convention

    Publishes:
      - cmd_vel (Twist): linear.x = 0, angular.z = omega

    Behavior:
      - Only actively publishes when triggered by new yaw_init / yaw_tar.
      - If data becomes stale:
          * publish STOP for stop_publish_duration_s seconds
          * then go completely quiet until new yaw_init / yaw_tar arrive
      - While active and data is fresh:
          * Compute yaw error = shortest angle from init to target (deg, [-180,180])
          * PID on this error to generate angular.z
          * Clamp angular.z to [-omega_max, omega_max]
          * If |error| < yaw_tolerance_deg -> publish zero Twist (stop)
    """

    def __init__(self):
        super().__init__('yaw_node')

        # Parameters
        self.declare_parameter('config_file', 'yaw.yaml')
        self.declare_parameter('max_rate_hz', 20.0)

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value

        self._min_period = 1.0 / float(max_rate)
        self._last_loop_time = self.get_clock().now()

        # Load YAML config
        pkg_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(pkg_share, 'config', config_file)
        self.get_logger().info(f'Using yaw config: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        yaw_cfg = self.conf.get('yaw', {})
        self.kp = float(yaw_cfg.get('kp', 0.05))
        self.ki = float(yaw_cfg.get('ki', 0.0))
        self.kd = float(yaw_cfg.get('kd', 0.01))

        self.omega_max = float(yaw_cfg.get('omega_max', 1.5))
        self.yaw_tolerance_deg = float(yaw_cfg.get('yaw_tolerance_deg', 2.0))
        self.yaw_timeout_s = float(yaw_cfg.get('yaw_timeout_s', 0.5))
        self.stop_publish_duration_s = float(yaw_cfg.get('stop_publish_duration_s', 1.0))

        self.cmd_topic = str(yaw_cfg.get('cmd_topic', 'cmd_vel'))
        self.debug_logging = bool(yaw_cfg.get('debug_logging', True))

        if self.debug_logging:
            self.get_logger().info(
                f'Yaw PID: kp={self.kp}, ki={self.ki}, kd={self.kd}, '
                f'omega_max={self.omega_max}, tol={self.yaw_tolerance_deg} deg, '
                f'timeout={self.yaw_timeout_s}s, stop_publish_duration={self.stop_publish_duration_s}s'
            )

        # State: latest yaw values
        self.yaw_init: Optional[float] = None
        self.yaw_tar: Optional[float] = None
        self.last_yaw_init_time = None
        self.last_yaw_tar_time = None

        # Control state / PID state
        self.integral = 0.0
        self.last_error_deg = 0.0
        self.last_pid_time = self.get_clock().now()

        # Activation & staleness logic
        self.active = False           # true after new yaw_init / yaw_tar arrive
        self.stale_since = None       # when we first noticed stale data

        # QoS like other sensor topics
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_yaw_init = self.create_subscription(
            Float64,
            'yaw_init',
            self.yaw_init_callback,
            sensor_qos
        )
        self.sub_yaw_tar = self.create_subscription(
            Float64,
            'yaw_tar',
            self.yaw_tar_callback,
            sensor_qos
        )

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, self.cmd_topic, 10)

        # Control loop timer
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(f'YawNode started, publishing Twist on "{self.cmd_topic}".')

    # === Callbacks ===

    def yaw_init_callback(self, msg: Float64):
        self.yaw_init = float(msg.data)
        self.last_yaw_init_time = self.get_clock().now()
        # New data is a trigger: activate controller
        self.active = True
        self.stale_since = None

    def yaw_tar_callback(self, msg: Float64):
        self.yaw_tar = float(msg.data)
        self.last_yaw_tar_time = self.get_clock().now()
        # New data is a trigger: activate controller
        self.active = True
        self.stale_since = None

    # === Helpers ===

    @staticmethod
    def _shortest_angle_deg(target_deg: float, current_deg: float) -> float:
        """
        Compute shortest signed angle from current to target in degrees,
        result in [-180, 180].
        """
        diff = target_deg - current_deg
        # Normalize to [-180, 180]
        while diff > 180.0:
            diff -= 360.0
        while diff <= -180.0:
            diff += 360.0
        return diff

    def _data_is_stale(self, now) -> bool:
        """
        Check if yaw_init or yaw_tar are missing or older than yaw_timeout_s.
        """
        if self.yaw_init is None or self.yaw_tar is None:
            return True

        if self.last_yaw_init_time is None or self.last_yaw_tar_time is None:
            return True

        dt_init = (now - self.last_yaw_init_time).nanoseconds / 1e9
        dt_tar = (now - self.last_yaw_tar_time).nanoseconds / 1e9

        if dt_init > self.yaw_timeout_s or dt_tar > self.yaw_timeout_s:
            return True

        return False

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)

    # === Control loop ===

    def control_loop(self):
        now = self.get_clock().now()
        elapsed_rate = (now - self._last_loop_time).nanoseconds / 1e9
        if elapsed_rate < self._min_period:
            return
        self._last_loop_time = now

        # If not active, we stay quiet (no Twist publishing).
        if not self.active:
            return

        # Check staleness of data
        stale = self._data_is_stale(now)

        if stale:
            # If this is the first time we notice staleness, record the time
            if self.stale_since is None:
                self.stale_since = now
                if self.debug_logging:
                    self.get_logger().info(
                        'YawNode: data became stale, starting STOP publishing window.'
                    )

            # How long have we been stale?
            stale_duration = (now - self.stale_since).nanoseconds / 1e9

            if stale_duration <= self.stop_publish_duration_s:
                # Still in STOP window: publish zero Twist and reset PID
                self.integral = 0.0
                self.last_error_deg = 0.0
                self.last_pid_time = now
                self.publish_stop()
            else:
                # Past STOP window: go quiet and reset state
                if self.debug_logging:
                    self.get_logger().info(
                        'YawNode: stale for too long, going quiet until new yaw messages.'
                    )
                self.active = False
                self.stale_since = None
                # Reset PID state
                self.integral = 0.0
                self.last_error_deg = 0.0
                self.last_pid_time = now
            return

        # Data is fresh; clear staleness tracking
        self.stale_since = None

        # Compute error in degrees (shortest angle)
        # Treat yaw_init as "current object angle", yaw_tar as desired angle (0°).
        # Error = current - target (wrapped to [-180, 180])
        error_deg = self._shortest_angle_deg(self.yaw_init, self.yaw_tar)


        # If within tolerance, stop (but remain active as long as new data comes)
        if abs(error_deg) <= self.yaw_tolerance_deg:
            if self.debug_logging:
                self.get_logger().info(f'Yaw aligned: error={error_deg:.2f} deg -> stopping.')
            self.integral = 0.0
            self.last_error_deg = error_deg
            self.last_pid_time = now
            self.publish_stop()
            return

        # PID timing
        dt = (now - self.last_pid_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 1e-6
        self.last_pid_time = now

        # PID calculation (error in DEGREES)
        self.integral += error_deg * dt
        derivative = (error_deg - self.last_error_deg) / dt
        self.last_error_deg = error_deg

        omega = self.kp * error_deg + self.ki * self.integral + self.kd * derivative

        # Clamp omega
        if omega > self.omega_max:
            omega = self.omega_max
        elif omega < -self.omega_max:
            omega = -self.omega_max

        # Twist: rotate in place only
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = omega
        self.pub_cmd_vel.publish(twist)

        if self.debug_logging:
            self.get_logger().info(
                f'[YawPID] e={error_deg:.2f} deg, omega={omega:.3f} rad/s'
            )

    # === Shutdown ===

    def destroy_node(self):
        self.get_logger().info('YawNode shutting down, stopping robot.')
        for _ in range(5):
            self.publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YawNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
