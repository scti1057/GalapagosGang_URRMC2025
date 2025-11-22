#!/usr/bin/env python3

import os
import math
import yaml
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64, Bool, Float64MultiArray
from ament_index_python.packages import get_package_share_directory


class ParcourNode(Node):
    """
    High-level parcours planner / sequencer.

    Subscribes:
      - r_lidar          (Float64MultiArray)  : [deg1, dist1, deg2, dist2, ...]
      - x_red            (Float64)           : center of red sign in pixels
      - red_sign_big     (Bool)              : True while we are in parcours mode
      - lane/x_white_far (Float64)
      - lane/x_white_near(Float64)
      - lane/x_yellow_far(Float64)
      - lane/x_yellow_near(Float64)

    Publishes:
      - yaw_init_par (Float64): initial yaw reference for alignment (future)
      - yaw_tar_par  (Float64): desired yaw target (future)
      - x_par        (Float64): planned x target in image coordinates

    State machine (skeleton for now):
      - IDLE     : waiting for red_sign_big == True
      - PLANNING : compute a provisional x_par and yaw targets
      - ALIGNING : controller rotates until yaw_init_par ~ yaw_tar_par (future)
      - (later): DRIVING, REPLANNING, etc.

    For now:
      - When red_sign_big becomes True:
          * enter PLANNING
          * set x_par to image center
          * set yaw_init_par and yaw_tar_par to 0.0 (placeholder)
          * transition to ALIGNING
      - While red_sign_big is False:
          * stay in IDLE and do nothing.
    """

    def __init__(self):
        super().__init__('parcour_node')

        # === Parameters ===
        self.declare_parameter('config_file', 'parcour.yaml')
        self.declare_parameter('max_rate_hz', 10.0)

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value

        self._min_period = 1.0 / float(max_rate)
        self._last_step_time = self.get_clock().now()

        # Load YAML config
        pkg_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(pkg_share, 'config', config_file)
        self.get_logger().info(f'Using parcour config: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        par_cfg = self.conf.get('parcour', {})
        self.image_width_px = float(par_cfg.get('image_width_px', 640.0))
        self.image_center_x = self.image_width_px / 2.0

        self.replan_interval_s = float(par_cfg.get('replan_interval_s', 1.0))
        self.min_clear_distance_m = float(par_cfg.get('min_clear_distance_m', 0.3))
        self.forward_step_distance_m = float(par_cfg.get('forward_step_distance_m', 0.3))
        self.yaw_align_tolerance_deg = float(par_cfg.get('yaw_align_tolerance_deg', 3.0))
        self.yaw_max_turn_deg = float(par_cfg.get('yaw_max_turn_deg', 90.0))

        self.debug_logging = bool(par_cfg.get('debug_logging', True))
        self.use_lidar_for_planning = bool(par_cfg.get('use_lidar_for_planning', True))
        self.use_red_for_planning = bool(par_cfg.get('use_red_for_planning', True))

        if self.debug_logging:
            self.get_logger().info(
                f'Parcour params: image_width={self.image_width_px}, '
                f'replan_interval={self.replan_interval_s}s, '
                f'min_clear_distance={self.min_clear_distance_m}m, '
                f'forward_step={self.forward_step_distance_m}m, '
                f'yaw_align_tol={self.yaw_align_tolerance_deg}deg'
            )

        # === Internal state ===

        # latest sensor values
        self.lidar_objects: List[Dict[str, float]] = []  # from r_lidar
        self.x_red = None

        self.red_sign_big = False
        self.red_sign_big_prev = False

        self.x_white_far = None
        self.x_white_near = None
        self.x_yellow_far = None
        self.x_yellow_near = None

        # parcours state machine
        self.state = 'IDLE'  # 'IDLE', 'PLANNING', 'ALIGNING', later 'DRIVING', ...
        self.last_plan_time = self.get_clock().now()

        # current planned outputs
        self.current_yaw_init = 0.0
        self.current_yaw_tar = 0.0
        self.current_x_par = None

        # === ROS wiring ===

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_r_lidar = self.create_subscription(
            Float64MultiArray,
            'r_lidar',
            self.r_lidar_callback,
            sensor_qos
        )

        self.sub_x_red = self.create_subscription(
            Float64,
            'x_red',
            self.x_red_callback,
            sensor_qos
        )

        self.sub_red_sign_big = self.create_subscription(
            Bool,
            'red_sign_big',
            self.red_sign_big_callback,
            sensor_qos
        )

        self.sub_x_white_far = self.create_subscription(
            Float64,
            'lane/x_white_far',
            self.x_white_far_callback,
            sensor_qos
        )
        self.sub_x_white_near = self.create_subscription(
            Float64,
            'lane/x_white_near',
            self.x_white_near_callback,
            sensor_qos
        )
        self.sub_x_yellow_far = self.create_subscription(
            Float64,
            'lane/x_yellow_far',
            self.x_yellow_far_callback,
            sensor_qos
        )
        self.sub_x_yellow_near = self.create_subscription(
            Float64,
            'lane/x_yellow_near',
            self.x_yellow_near_callback,
            sensor_qos
        )

        # Publishers
        self.pub_yaw_init_par = self.create_publisher(Float64, 'yaw_init_par', 10)
        self.pub_yaw_tar_par = self.create_publisher(Float64, 'yaw_tar_par', 10)
        self.pub_x_par = self.create_publisher(Float64, 'x_par', 10)

        # Main timer
        self.timer = self.create_timer(0.05, self.step)  # 20Hz, but rate-limited

        self.get_logger().info('ParcourNode started (state=IDLE).')

    # === Callbacks for inputs ===

    def r_lidar_callback(self, msg: Float64MultiArray):
        """Parse r_lidar into a list of objects [angle_deg, distance]."""
        data = list(msg.data)
        objects = []
        for i in range(0, len(data), 2):
            if i + 1 >= len(data):
                break
            ang = data[i]
            dist = data[i + 1]
            if math.isnan(ang) or math.isnan(dist):
                continue
            objects.append({'angle_deg': float(ang), 'distance': float(dist)})
        self.lidar_objects = objects

    def x_red_callback(self, msg: Float64):
        self.x_red = msg.data

    def red_sign_big_callback(self, msg: Bool):
        self.red_sign_big = bool(msg.data)

    def x_white_far_callback(self, msg: Float64):
        self.x_white_far = msg.data

    def x_white_near_callback(self, msg: Float64):
        self.x_white_near = msg.data

    def x_yellow_far_callback(self, msg: Float64):
        self.x_yellow_far = msg.data

    def x_yellow_near_callback(self, msg: Float64):
        self.x_yellow_near = msg.data

    # === Main state machine step ===

    def step(self):
        """Main loop for parcours state machine."""
        now = self.get_clock().now()
        elapsed = (now - self._last_step_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_step_time = now

        # Detect rising / falling edge of red_sign_big
        if self.red_sign_big and not self.red_sign_big_prev:
            # rising edge: start parcours
            self._on_red_sign_activated(now)
        elif not self.red_sign_big and self.red_sign_big_prev:
            # falling edge: stop parcours
            self._on_red_sign_deactivated(now)

        self.red_sign_big_prev = self.red_sign_big

        # If not in parcours mode, stay idle
        if not self.red_sign_big:
            return

        # State machine
        if self.state == 'IDLE':
            # In case red_sign_big was already true before we started
            self.state = 'PLANNING'
            self.last_plan_time = now
            if self.debug_logging:
                self.get_logger().info('Parcour: entering PLANNING (from IDLE).')

        if self.state == 'PLANNING':
            self._do_planning(now)

        elif self.state == 'ALIGNING':
            self._do_aligning(now)

        # (Later we’ll add DRIVING, REPLAN, etc.)

    # === State handlers ===

    def _on_red_sign_activated(self, now):
        """Called when red_sign_big goes from False -> True."""
        self.state = 'PLANNING'
        self.last_plan_time = now
        if self.debug_logging:
            self.get_logger().info('Parcour: red_sign_big TRUE, starting PLANNING.')

    def _on_red_sign_deactivated(self, now):
        """Called when red_sign_big goes from True -> False."""
        if self.debug_logging:
            self.get_logger().info('Parcour: red_sign_big FALSE, stopping parcours, back to IDLE.')
        self.state = 'IDLE'
        # Optionally reset outputs or leave last ones

    def _do_planning(self, now):
        """
        Planning step: for now, simply plan to go straight (x_par = image center),
        and set a dummy yaw target (0.0). In the future we’ll use:
          - r_lidar to avoid obstacles
          - x_red / lane lines to choose path
        """
        # Simple initial plan: keep x_par at image center
        x_par = self.image_center_x

        # Placeholder for yaw planning:
        # For example, 0.0 rad or deg relative heading — we’ll define this later
        yaw_init = 0.0
        yaw_tar = 0.0

        self.current_x_par = x_par
        self.current_yaw_init = yaw_init
        self.current_yaw_tar = yaw_tar

        # Publish current plan
        self.pub_x_par.publish(Float64(data=float(x_par)))
        self.pub_yaw_init_par.publish(Float64(data=float(yaw_init)))
        self.pub_yaw_tar_par.publish(Float64(data=float(yaw_tar)))

        if self.debug_logging:
            self.get_logger().info(
                f'Parcour PLANNING: x_par={x_par:.1f}, yaw_init={yaw_init:.1f}, yaw_tar={yaw_tar:.1f}'
            )

        # For now, immediately transition to ALIGNING
        self.state = 'ALIGNING'
        if self.debug_logging:
            self.get_logger().info('Parcour: switching to ALIGNING.')

    def _do_aligning(self, now):
        """
        Alignment step: controller should rotate until yaw reaches yaw_tar_par.
        For now, we just keep publishing the same yaw_init_par / yaw_tar_par and x_par.
        Later we’ll:
          - read actual yaw from another node
          - detect when alignment finished
          - transition to DRIVING, then back to PLANNING, etc.
        """
        if self.current_x_par is None:
            # No plan -> go back to planning
            self.state = 'PLANNING'
            if self.debug_logging:
                self.get_logger().info('Parcour ALIGNING: no x_par, going back to PLANNING.')
            return

        # Re-publish the current targets
        self.pub_x_par.publish(Float64(data=float(self.current_x_par)))
        self.pub_yaw_init_par.publish(Float64(data=float(self.current_yaw_init)))
        self.pub_yaw_tar_par.publish(Float64(data=float(self.current_yaw_tar)))

        # In future: check yaw feedback and transition to DRIVING when aligned

    # === Cleanup ===

    def destroy_node(self):
        self.get_logger().info('ParcourNode shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ParcourNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
