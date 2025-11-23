#!/usr/bin/env python3

import os
import math
import yaml
from typing import List, Dict, Optional

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64, Bool, Float64MultiArray
from ament_index_python.packages import get_package_share_directory


class ParcourNode(Node):
    """
    High-level parcours planner / sequencer + debug visualizer.

    Subscribes:
      - r_lidar           (Float64MultiArray): [deg1, dist1, deg2, dist2, ...]
      - x_red             (Float64)         : center of red sign in pixels
      - red_sign_big      (Bool)            : TRUE while we're in parcours mode
      - lane/x_white_far  (Float64)
      - lane/x_white_near (Float64)
      - lane/x_yellow_far (Float64)
      - lane/x_yellow_near(Float64)
      - camera_topic      (CompressedImage, for debug view)

    Publishes:
      - yaw_init_par (Float64): initial yaw reference for alignment (future)
      - yaw_tar_par  (Float64): desired yaw target (future)
      - x_par        (Float64): planned x target in image coordinates

    State machine (for now, still simple):
      - IDLE     : waiting for red_sign_big == True
      - PLANNING : decide x_par and yaw targets
      - ALIGNING : controller rotates until yaw_init_par ~ yaw_tar_par (future)
      - (later): DRIVING, REPLANNING, etc.

    Planning logic (current version):
      - If red_sign_big is FALSE:
          -> stay in IDLE, no x_par planning.
      - When red_sign_big becomes TRUE (rising edge):
          -> enter PLANNING.
      - In PLANNING:
          -> if x_red is known: x_par aims at red sign (x_par = x_red)
             else: x_par approx lane center based on white/yellow lines.
          -> yaw_init_par, yaw_tar_par are placeholder (0.0) for now.
          -> publish x_par, yaw_init_par, yaw_tar_par.
          -> go to ALIGNING.
      - In ALIGNING:
          -> keep publishing the same x_par / yaw targets.
          -> later we'll integrate yaw feedback + driving steps.

    Debug visualization:
      - If debug_visualization := true and camera_topic is valid:
          -> show image with:
              * x_red
              * x_white_far, x_white_near
              * x_yellow_far, x_yellow_near
              * x_par (planned)
              * lidar objects projected into image using
                lidar_angle_min/max and mapping_type from YAML.
    """

    def __init__(self):
        super().__init__('parcour_node')

        # Debug image state
        self.bridge: Optional[CvBridge] = None
        self.debug_image = None
        self._debug_window = 'parcour_debug'

        # === Parameters ===
        self.declare_parameter('config_file', 'parcour.yaml')
        self.declare_parameter('max_rate_hz', 10.0)
        self.declare_parameter('debug_visualization', False)
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value
        self.debug_visualization = self.get_parameter('debug_visualization').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

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

        vis_cfg = par_cfg.get('visualization', {})
        self.viz_angle_min_deg = float(vis_cfg.get('lidar_angle_min_deg', -90.0))
        self.viz_angle_max_deg = float(vis_cfg.get('lidar_angle_max_deg', 90.0))
        self.viz_mapping_type = str(vis_cfg.get('lidar_mapping_type', 'linear')).lower()
        self.viz_mapping_exp_k = float(vis_cfg.get('lidar_mapping_exp_k', 1.5))

        self.viz_dist_min_m = float(vis_cfg.get('lidar_dist_min_m', 0.2))
        self.viz_dist_max_m = float(vis_cfg.get('lidar_dist_max_m', 1.0))
        self.viz_dot_radius_min = int(vis_cfg.get('lidar_dot_radius_min_px', 3))
        self.viz_dot_radius_max = int(vis_cfg.get('lidar_dot_radius_max_px', 12))
        self.viz_y_line_offset_px = int(vis_cfg.get('y_line_offset_px', 50))

        if self.debug_logging:
            self.get_logger().info(
                f'Parcour params: image_width={self.image_width_px}, '
                f'replan_interval={self.replan_interval_s}s, '
                f'min_clear_distance={self.min_clear_distance_m}m, '
                f'forward_step={self.forward_step_distance_m}m, '
                f'yaw_align_tol={self.yaw_align_tolerance_deg}deg, '
                f'lidar_vis_angle=[{self.viz_angle_min_deg}, {self.viz_angle_max_deg}]deg, '
                f'lidar_mapping_type={self.viz_mapping_type}'
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

        # Debug camera subscription
        if self.debug_visualization:
            self.bridge = CvBridge()
            self.sub_debug_cam = self.create_subscription(
                CompressedImage,
                self.camera_topic,
                self.debug_image_callback,
                sensor_qos
            )
            self.get_logger().info(
                f'Parcour debug visualization enabled, subscribing to camera: {self.camera_topic}'
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

    def debug_image_callback(self, msg: CompressedImage):
        """Store latest camera image for debug visualization."""
        if not msg.data:
            return
        if self.bridge is None:
            return
        try:
            self.debug_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'Parcour debug: failed to convert compressed image: {e}')

    # === Main state machine step ===

    def step(self):
        """Main loop for parcours state machine + debug drawing."""
        now = self.get_clock().now()
        elapsed = (now - self._last_step_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            # Even if we skip state logic, we can still update debug view occasionally
            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
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
            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
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

        # After state logic, draw debug overlay if enabled
        if self.debug_visualization and self.debug_image is not None:
            self._draw_debug_overlay()

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
        # Optionally reset outputs (we can leave last values as-is for now)

    def _do_planning(self, now):
        """
        Planning step:
          - Primary goal: point towards red sign (if visible).
          - If red sign is not visible: approximate lane center using lane lines.
          - For now, yaw_init and yaw_tar are placeholders (0.0).
          - Later we integrate:
              * lidar_objects to avoid obstacles
              * lane constraints to avoid crossing white/yellow
              * step-based forward motion.
        """
        # 1) Base target on red sign if available
        if self.use_red_for_planning and self.x_red is not None:
            x_par = self.x_red
        else:
            # 2) Otherwise approximate lane center (Duckietown-style heuristic)
            left_x = self.x_white_near  # treat white_near as left
            right_x = self.x_yellow_near  # treat yellow_near as right
            x_par = self._compute_lane_center(left_x, right_x)

            # Fallback to image center if no lane info
            if x_par is None:
                x_par = self.image_center_x

        # Placeholders for yaw planning (we'll define conventions later)
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

    # === Helpers ===

    def _compute_lane_center(self, left_x: Optional[float], right_x: Optional[float]) -> Optional[float]:
        """
        Roughly mimic the Duckietown lane-center heuristic used in control_node:
          if left_x and right_x valid and left_x > right_x:
              center = (left_x + right_x)/2 - 30
          elif right only:
              center = right_x + 170
          elif left only:
              center = left_x - 230
          else:
              None
        """
        if left_x is not None and right_x is not None and left_x > right_x:
            return (left_x + right_x) / 2.0 - 30.0
        elif right_x is not None:
            return right_x + 170.0
        elif left_x is not None:
            return left_x - 230.0
        else:
            return None

    def _angle_to_image_x(self, angle_deg: float, image_width: int) -> Optional[int]:
        """
        Map lidar angle (deg) into image x pixel using configuration.

        We assume:
          - angle_deg in [viz_angle_min_deg, viz_angle_max_deg]
          - 0 deg = center of image
          - positive angles = LEFT of the robot
          - image x increases to the RIGHT

        So we:
          1) normalize angle to [0, 1] between viz_angle_min_deg and viz_angle_max_deg
          2) optionally warp it (exp mode)
          3) invert it: positive angle => smaller x (left)
        """
        # Outside visualization range -> ignore
        if angle_deg < self.viz_angle_min_deg or angle_deg > self.viz_angle_max_deg:
            return None

        span = self.viz_angle_max_deg - self.viz_angle_min_deg
        if span <= 0.0:
            return None

        # normalize angle to [0,1]
        norm = (angle_deg - self.viz_angle_min_deg) / span

        # Optional non-linear mapping
        if self.viz_mapping_type == 'exp':
            # map norm from [0,1] to [-1,1]
            a = 2.0 * norm - 1.0
            k = self.viz_mapping_exp_k
            # signed power warp
            a_warp = math.copysign(abs(a) ** k, a)
            # back to [0,1]
            norm = (a_warp + 1.0) / 2.0

        # Clamp
        norm = max(0.0, min(1.0, norm))

        # Invert orientation: positive angles => left side (small x)
        norm_inv = 1.0 - norm

        x = int(round(norm_inv * (image_width - 1)))
        return x


    def _distance_to_radius(self, dist_m: float) -> int:
        """
        Map distance to a dot radius: closer -> larger, farther -> smaller.
        Uses viz_dist_min_m, viz_dist_max_m and dot_radius_min/max.
        """
        d_min = self.viz_dist_min_m
        d_max = self.viz_dist_max_m
        r_min = self.viz_dot_radius_min
        r_max = self.viz_dot_radius_max

        if dist_m <= 0.0:
            return r_max

        # Clamp distance
        d = max(d_min, min(d_max, dist_m))
        # 0 at d_min, 1 at d_max
        t = (d - d_min) / (d_max - d_min) if d_max > d_min else 1.0
        radius = int(round(r_max - t * (r_max - r_min)))
        return max(r_min, min(r_max, radius))

    def _draw_debug_overlay(self):
        """Draw x_red, lane points, x_par and lidar objects onto the debug image."""
        if self.debug_image is None:
            return

        img = self.debug_image.copy()
        h, w = img.shape[:2]
        y_line = h - self.viz_y_line_offset_px
        if y_line < 0:
            y_line = int(h * 0.9)

        # Helper to draw a labeled point
        def draw_point(x_val: Optional[float], label: str, color, dy: int = 0):
            if x_val is None:
                return
            x_int = int(round(x_val))
            x_int = max(0, min(w - 1, x_int))
            cv2.circle(img, (x_int, y_line), 5, color, -1)
            cv2.putText(
                img, label,
                (x_int - 20, y_line - 10 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1
            )

        # Draw center of image
        center_x_int = int(round(self.image_center_x))
        cv2.circle(img, (center_x_int, y_line), 4, (255, 255, 255), -1)
        cv2.putText(
            img, 'center',
            (center_x_int - 30, y_line - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )

        # Draw lane-related x positions
        draw_point(self.x_white_far, 'white_far', (200, 200, 200), dy=-20)
        draw_point(self.x_white_near, 'white_near', (255, 255, 255), dy=0)
        draw_point(self.x_yellow_far, 'yellow_far', (0, 200, 200), dy=-20)
        draw_point(self.x_yellow_near, 'yellow_near', (0, 255, 255), dy=0)

        # Draw x_red (red sign center)
        draw_point(self.x_red, 'x_red', (0, 0, 255), dy=20)

        # Draw x_par (planned target)
        if self.current_x_par is not None:
            draw_point(self.current_x_par, 'x_par', (255, 0, 255), dy=40)

        # Draw lidar objects projected into the image
        for obj in self.lidar_objects:
            ang = obj['angle_deg']
            dist = obj['distance']
            x_pix = self._angle_to_image_x(ang, w)
            if x_pix is None:
                continue
            radius = self._distance_to_radius(dist)
            cv2.circle(img, (x_pix, y_line), radius, (0, 255, 0), -1)
            # Optional: draw distance text above
            cv2.putText(
                img, f'{dist:.2f}m',
                (x_pix - 15, y_line - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1
            )

        cv2.imshow(self._debug_window, img)
        cv2.waitKey(1)

    # === Cleanup ===

    def destroy_node(self):
        self.get_logger().info('ParcourNode shutting down.')
        if self.debug_visualization:
            cv2.destroyAllWindows()
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
