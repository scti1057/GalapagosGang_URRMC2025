#!/usr/bin/env python3

import os
import math
import yaml
from typing import List, Dict, Optional
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Float64, Bool, Float64MultiArray
from ament_index_python.packages import get_package_share_directory


class ParcourNode(Node):
    """
    Parcours planner with 3 states + pauses:

      States:
        - IDLE:
            * Waiting for conditions
        - ALIGN (state 1):
            * Align closest object to 0 deg using yaw_init_par / yaw_tar_par
        - PAUSE:
            * 2s pause between states, no commands
        - DRIVE (state 2):
            * Publish x_par (closest object projection in image)
        - ALIGN_90 (state 3):
            * Align closest object to 90 deg
        - DONE:
            * Do nothing, just stay here

      Transitions (happy path):

        IDLE
          └─> ALIGN      (if |angle| > deadzone and dist > stop_dist)
             └─> PAUSE (2s)
                └─> DRIVE
                   └─> PAUSE (2s) when object is close (dist <= stop_dist)
                      └─> ALIGN_90
                         └─> PAUSE (2s) when aligned to 90°
                            └─> DONE

      Gating:
        - We only run states if:
            * x_red is not None (we see red sign)
            * red_sign_big is False
            * we have at least 1 LiDAR object
        - Otherwise we reset to IDLE.
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

        # Deadzone for closest object angle (state 1 + 3)
        self.align_deadzone_deg = float(par_cfg.get('align_object_angle_deadzone_deg', 5.0))

        # Distance threshold for stopping DRIVE and switching to next state
        self.drive_stop_distance_m = float(par_cfg.get('drive_stop_distance_m', 0.4))

        # Pause duration (between states)
        self.pause_duration_s = float(par_cfg.get('pause_duration_s', 2.0))

        # Target angle for state 3 (object at +90° on the left)
        self.align90_target_deg = float(par_cfg.get('align90_target_deg', 90.0))

        # Step 0: align to red in the image
        self.align_red_tolerance_px = float(par_cfg.get('align_red_tolerance_px', 20.0))
    
        # Max pseudo-angle (deg) we use when mapping x_red offset to yaw_init
        self.red_align_max_deg = float(par_cfg.get('red_align_max_deg', 90.0))

        # Corridor ahead towards the red sign
        # corridor_width_m is the **total** width (robot width + margin)
        self.corridor_width_m = float(par_cfg.get('corridor_width_m', 0.25))
        self.corridor_max_distance_m = float(par_cfg.get('corridor_max_distance_m', 1.0))

        # Step 4 circle parameters (around the pillar)
        self.circle_linear_x = float(par_cfg.get('circle_linear_x', 0.08))          # m/s
        self.circle_angular_z = float(par_cfg.get('circle_angular_z', 0.6))         # rad/s (CCW / left)
        self.circle_center_tolerance_px = float(par_cfg.get('circle_center_tolerance_px', 20.0))

        # Abort conditions for circle-left / circle-right
        self.circle_min_distance_m = float(par_cfg.get('circle_min_distance_m', 0.25))  # too close to object
        self.circle_line_freshness_s = float(par_cfg.get('circle_line_freshness_s', 0.5))  # "fresh" line detection window



        self.debug_logging = bool(par_cfg.get('debug_logging', True))
        self.use_lidar_for_planning = bool(par_cfg.get('use_lidar_for_planning', True))
        self.use_red_for_planning = bool(par_cfg.get('use_red_for_planning', True))

        vis_cfg = par_cfg.get('visualization', {})
        self.viz_angle_min_deg = float(vis_cfg.get('lidar_angle_min_deg', -50.0))
        self.viz_angle_max_deg = float(vis_cfg.get('lidar_angle_max_deg', 54.0))
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
                f'align_deadzone={self.align_deadzone_deg}deg, '
                f'drive_stop_distance={self.drive_stop_distance_m}m, '
                f'pause_duration={self.pause_duration_s}s, '
                f'align90_target_deg={self.align90_target_deg}deg, '
                f'lidar_vis_angle=[{self.viz_angle_min_deg}, {self.viz_angle_max_deg}]deg, '
                f'lidar_mapping_type={self.viz_mapping_type}'
            )

        # === Internal state ===

        # latest sensor values
        self.lidar_objects: List[Dict[str, float]] = []  # from r_lidar
        self.x_red = None

        self.red_sign_big = False

        self.x_white_far = None
        self.x_white_near = None
        self.x_yellow_far = None
        self.x_yellow_near = None  # <-- add this

        # Timestamps to know if lane detections are "fresh" during circle states
        self.x_white_near_stamp = None
        self.x_yellow_near_stamp = None



        # state machine: 'IDLE', 'ALIGN', 'PAUSE', 'DRIVE', 'ALIGN_90', 'CIRCLE_LEFT', 'DONE'
        self.state = 'IDLE'


        # pause state info
        self.pause_until: Optional[Duration] = None
        self.pause_next_state: Optional[str] = None

        # current planned outputs (for debug)
        self.current_yaw_init = None
        self.current_yaw_tar = None
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
        # Publisher for direct cmd_vel during CIRCLE_LEFT (state 4)
        self.pub_cmd_vel_circle = self.create_publisher(Twist, 'cmd_vel', 10)


        # Main timer
        self.timer = self.create_timer(0.05, self.step)  # 20Hz, but rate-limited

        self.get_logger().info('ParcourNode started (state machine: IDLE -> ALIGN -> DRIVE -> ALIGN_90 -> DONE).')

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
        self.x_white_near_stamp = self.get_clock().now()

    def x_yellow_far_callback(self, msg: Float64):
        self.x_yellow_far = msg.data

    def x_yellow_near_callback(self, msg: Float64):
        self.x_yellow_near = msg.data
        self.x_yellow_near_stamp = self.get_clock().now()

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

    # === Pause helper ===

    def _enter_pause(self, next_state: str, now):
        """Enter a PAUSE state that lasts pause_duration_s, then switches to next_state."""
        self.state = 'PAUSE'
        self.pause_next_state = next_state
        self.pause_until = now + Duration(seconds=self.pause_duration_s)
        if self.debug_logging:
            self.get_logger().info(
                f'Parcour: entering PAUSE ({self.pause_duration_s:.1f}s) before {next_state}.'
            )

    # === Main step ===

    def step(self):
        """Main loop: ALIGN (0°) -> PAUSE -> DRIVE -> PAUSE -> ALIGN_90 -> PAUSE -> DONE."""
        now = self.get_clock().now()
        elapsed = (now - self._last_step_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
            return
        self._last_step_time = now

        yaw_init = None
        yaw_tar = None
        x_par = None

        # === DONE state: final success or blocked; just show debug ===
        if self.state == 'DONE':
            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
            return

        # If the red sign is "big", parcours is finished
        if self.red_sign_big:
            if self.debug_logging and self.state != 'DONE':
                self.get_logger().info('Parcour: red_sign_big TRUE -> DONE (stop).')

            # stop any direct cmd_vel we may be sending
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            self.pub_cmd_vel_circle.publish(stop_twist)

            self.state = 'DONE'
            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
            return

        # Optional: nearest LiDAR object (may be None if we currently have no objects)
        closest = None
        angle_deg = None
        dist_m = None
        if self.lidar_objects:
            closest = min(self.lidar_objects, key=lambda o: o['distance'])
            angle_deg = closest['angle_deg']
            dist_m = closest['distance']


        # --- Gating by x_red, red_sign_big and LiDAR availability ---

        # We only run parcours if:
        #   - x_red is known (we see the red sign in the image)
        #   - red_sign_big is FALSE (we are not yet done with parcour)
        #   - we have at least one LiDAR object
        if (self.x_red is None) or self.red_sign_big or (len(self.lidar_objects) == 0):
            if self.state != 'IDLE' and self.debug_logging:
                reasons = []
                if self.x_red is None:
                    reasons.append('no x_red')
                if self.red_sign_big:
                    reasons.append('red_sign_big TRUE')
                if len(self.lidar_objects) == 0:
                    reasons.append('no LiDAR objects')
                reason_str = ', '.join(reasons) if reasons else 'gating condition'
                self.get_logger().info(
                    f'Parcour: {reason_str} -> state=IDLE.'
                )

            self.state = 'IDLE'
            self.pause_until = None
            self.pause_next_state = None
            self.current_yaw_init = None
            self.current_yaw_tar = None
            self.current_x_par = None

            if self.debug_visualization and self.debug_image is not None:
                self._draw_debug_overlay()
            return

        # We have x_red, red_sign_big == False, and at least one object
        closest = min(self.lidar_objects, key=lambda o: o['distance'])
        angle_deg = closest['angle_deg']
        dist_m = closest['distance']

        # === PAUSE state ===
        if self.state == 'PAUSE':
            if self.pause_until is not None and now < self.pause_until:
                # Still pausing, no commands
                if self.debug_visualization and self.debug_image is not None:
                    self._draw_debug_overlay()
                return
            # Pause finished
            next_state = self.pause_next_state or 'IDLE'
            if self.debug_logging:
                self.get_logger().info(f'Parcour: PAUSE done -> {next_state}.')
            self.state = next_state
            self.pause_until = None
            self.pause_next_state = None
            # Continue with new state logic in this same cycle

        # === IDLE: Step 0 -> align to red & corridor check ===
        if self.state == 'IDLE':
            # If we don't see the red sign yet, do nothing.
            if self.x_red is None:
                if self.debug_logging:
                    self.get_logger().info('Parcour: IDLE but no x_red yet, waiting.')
                if self.debug_visualization and self.debug_image is not None:
                    self._draw_debug_overlay()
                return

            # Align red sign to image center using yaw_node, but yaw_node expects degrees.
            # We "fake" an angle based on how far x_red is from the center in pixels.
            err_px = self.image_center_x - self.x_red

            if abs(err_px) > self.align_red_tolerance_px:
                # Map pixel error to a pseudo-angle in [-red_align_max_deg, +red_align_max_deg]
                
                yaw_init = err_px * 0.3


                # yaw_tar is always 0 deg (we want to end at "0")
                yaw_tar = 0.0

                self.pub_yaw_init_par.publish(Float64(data=float(yaw_init)))
                self.pub_yaw_tar_par.publish(Float64(data=float(yaw_tar)))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Parcour ALIGN_RED: x_red={self.x_red:.1f}, center={self.image_center_x:.1f}, '
                        f'err_px={err_px:.1f} -> yaw_init={yaw_init:.1f} deg, yaw_tar={yaw_tar:.1f} deg'
                    )
            else:
                # Red roughly centered -> check corridor ahead
                corridor_clear = True
                if closest is not None:
                    corridor_clear = self._is_corridor_clear()

                if corridor_clear:
                    self.state = 'GO_RED'
                    if self.debug_logging:
                        self.get_logger().info(
                            'Parcour: ALIGN_RED done, corridor clear -> GO_RED.'
                        )
                else:
                    if closest is None:
                        if self.debug_logging:
                            self.get_logger().info(
                                'Parcour: corridor blocked but no closest object? -> back to IDLE.'
                            )
                    else:
                        self.state = 'ALIGN'
                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour: ALIGN_RED done but corridor blocked '
                                f'-> ALIGN on closest object (angle={angle_deg:.1f} deg).'
                            )


        # === GO_RED: drive straight towards the red sign ===
        if self.state == 'GO_RED':
            if self.x_red is None:
                # Lost the red sign -> back to IDLE (re-acquire)
                if self.debug_logging:
                    self.get_logger().info('Parcour GO_RED: lost x_red, back to IDLE.')
                self.state = 'IDLE'
            else:
                x_par = float(self.x_red)
                self.pub_x_par.publish(Float64(data=x_par))

                if self.debug_logging:
                    self.get_logger().info(
                        f'Parcour GO_RED: x_par={x_par:.1f}'
                    )

        # === ALIGN (state 1, target 0° for object) ===
        if self.state == 'ALIGN':
            if closest is None:
                if self.debug_logging:
                    self.get_logger().info('Parcour ALIGN: no LiDAR objects, back to IDLE.')
                self.state = 'IDLE'
            else:
                if abs(angle_deg) > self.align_deadzone_deg:
                    # Still need to align: publish yaw commands, no x_par
                    yaw_init = angle_deg
                    yaw_tar = 0.0

                    self.pub_yaw_init_par.publish(Float64(data=float(yaw_init)))
                    self.pub_yaw_tar_par.publish(Float64(data=float(yaw_tar)))

                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour ALIGN: closest angle={angle_deg:.1f} deg, '
                            f'yaw_init_par={yaw_init:.1f}, yaw_tar_par={yaw_tar:.1f}'
                        )
                else:
                    # Alignment complete -> enter PAUSE before DRIVE
                    self._enter_pause('DRIVE', now)


        # === DRIVE (state 2) ===
        if self.state == 'DRIVE':
            if closest is None:
                if self.debug_logging:
                    self.get_logger().info('Parcour DRIVE: no LiDAR objects, back to IDLE.')
                self.state = 'IDLE'
            else:
                # If object becomes too close, stop driving and go to PAUSE -> ALIGN_90
                if dist_m <= self.drive_stop_distance_m:
                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour DRIVE: object too close (dist={dist_m:.2f} m) '
                            f'-> PAUSE then ALIGN_90.'
                        )
                    self._enter_pause('ALIGN_90', now)
                else:
                    # Drive toward the closest object: publish x_par = mapped x-position
                    x_pix = self._angle_to_image_x(angle_deg, int(self.image_width_px))
                    if x_pix is not None:
                        x_par = float(x_pix)
                        self.pub_x_par.publish(Float64(data=x_par))

                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour DRIVE: angle={angle_deg:.1f} deg, dist={dist_m:.2f} m, '
                                f'x_par={x_par:.1f}'
                            )
                    else:
                        # Angle outside mapping range -> back to IDLE
                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour DRIVE: angle={angle_deg:.1f} deg outside mapping range, '
                                f'-> state=IDLE.'
                            )
                        self.state = 'IDLE'


        # === ALIGN_90 (state 3, target +90° on the left) ===
        if self.state == 'ALIGN_90':
            if closest is None:
                if self.debug_logging:
                    self.get_logger().info('Parcour ALIGN_90: no LiDAR objects, back to IDLE.')
                self.state = 'IDLE'
            else:
                # Measure error vs. target (e.g. 90°)
                err_angle = angle_deg - self.align90_target_deg
                if abs(err_angle) > self.align_deadzone_deg:
                    # Still need to align: publish yaw commands with target=90°
                    yaw_init = angle_deg
                    yaw_tar = self.align90_target_deg

                    self.pub_yaw_init_par.publish(Float64(data=float(yaw_init)))
                    self.pub_yaw_tar_par.publish(Float64(data=float(yaw_tar)))

                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour ALIGN_90: angle={angle_deg:.1f} deg -> target={yaw_tar:.1f} deg'
                        )
                else:
                    # Alignment to 90° complete -> PAUSE then CIRCLE_LEFT
                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour: ALIGN_90 complete (angle={angle_deg:.1f} deg) '
                            f'-> PAUSE -> CIRCLE_LEFT.'
                        )
                    self._enter_pause('CIRCLE_LEFT', now)


        # === CIRCLE_LEFT (state 4): circle CCW around object ===
        if self.state == 'CIRCLE_LEFT':
            # Abort if object too close or fresh white line detected
            obstacle_too_close = (
                dist_m is not None and
                self.circle_min_distance_m > 0.0 and
                dist_m <= self.circle_min_distance_m
            )

            fresh_white = False
            if self.x_white_near is not None and self.x_white_near_stamp is not None:
                dt_white = (now - self.x_white_near_stamp).nanoseconds / 1e9
                if dt_white <= self.circle_line_freshness_s:
                    fresh_white = True

            if obstacle_too_close or fresh_white:
                # Abort circle-left and prepare for right side approach
                if self.debug_logging:
                    reason = []
                    if obstacle_too_close:
                        reason.append(f'dist={dist_m:.2f}m')
                    if fresh_white:
                        reason.append('x_white_near')
                    self.get_logger().info(
                        'Parcour CIRCLE_LEFT: abort due to ' + ' & '.join(reason)
                        + ' -> PAUSE then ALIGN_MINUS90.'
                    )

                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel_circle.publish(stop_twist)

                self._enter_pause('ALIGN_MINUS90', now)
            else:
                # Normal success condition: red sign centered again
                if self.x_red is not None:
                    err_px = self.x_red - self.image_center_x

                    if abs(err_px) <= self.circle_center_tolerance_px:
                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour CIRCLE_LEFT: x_red centered '
                                f'(err_px={err_px:.1f}) -> IDLE (restart loop).'
                            )

                        stop_twist = Twist()
                        stop_twist.linear.x = 0.0
                        stop_twist.angular.z = 0.0
                        self.pub_cmd_vel_circle.publish(stop_twist)

                        # Go back to step 0 (ALIGN_RED + corridor check)
                        self.state = 'IDLE'
                    else:
                        # Keep circling CCW
                        twist = Twist()
                        twist.linear.x = self.circle_linear_x
                        twist.angular.z = self.circle_angular_z   # CCW
                        self.pub_cmd_vel_circle.publish(twist)

                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour CIRCLE_LEFT: x_red_err={err_px:.1f} px, '
                                f'v={twist.linear.x:.3f}, omega={twist.angular.z:.3f}'
                            )
                else:
                    # No red detected -> keep circling, hoping to see it later
                    twist = Twist()
                    twist.linear.x = self.circle_linear_x
                    twist.angular.z = self.circle_angular_z
                    self.pub_cmd_vel_circle.publish(twist)

                    if self.debug_logging:
                        self.get_logger().info('Parcour CIRCLE_LEFT: no x_red, keep circling.')

        # === ALIGN_MINUS90: align object to -90° (right) before CIRCLE_RIGHT ===
        if self.state == 'ALIGN_MINUS90':
            if closest is None:
                if self.debug_logging:
                    self.get_logger().info('Parcour ALIGN_MINUS90: no LiDAR objects, back to IDLE.')
                self.state = 'IDLE'
            else:
                target_deg = -90.0
                err_angle = angle_deg - target_deg
                if abs(err_angle) > self.align_deadzone_deg:
                    yaw_init = angle_deg
                    yaw_tar = target_deg

                    self.pub_yaw_init_par.publish(Float64(data=float(yaw_init)))
                    self.pub_yaw_tar_par.publish(Float64(data=float(yaw_tar)))

                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour ALIGN_MINUS90: angle={angle_deg:.1f} deg -> target={yaw_tar:.1f} deg'
                        )
                else:
                    if self.debug_logging:
                        self.get_logger().info(
                            f'Parcour: ALIGN_MINUS90 complete (angle={angle_deg:.1f} deg) '
                            f'-> PAUSE -> CIRCLE_RIGHT.'
                        )
                    self._enter_pause('CIRCLE_RIGHT', now)

        # === CIRCLE_RIGHT: mirror of CIRCLE_LEFT but clockwise ===
        if self.state == 'CIRCLE_RIGHT':
            obstacle_too_close = (
                dist_m is not None and
                self.circle_min_distance_m > 0.0 and
                dist_m <= self.circle_min_distance_m
            )

            fresh_yellow = False
            if self.x_yellow_near is not None and self.x_yellow_near_stamp is not None:
                dt_yellow = (now - self.x_yellow_near_stamp).nanoseconds / 1e9
                if dt_yellow <= self.circle_line_freshness_s:
                    fresh_yellow = True

            if obstacle_too_close or fresh_yellow:
                # Both directions problematic -> stop and "wait"
                if self.debug_logging:
                    reason = []
                    if obstacle_too_close:
                        reason.append(f'dist={dist_m:.2f}m')
                    if fresh_yellow:
                        reason.append('x_yellow_near')
                    self.get_logger().info(
                        'Parcour CIRCLE_RIGHT: abort due to ' + ' & '.join(reason)
                        + ' -> DONE (waiting).'
                    )

                stop_twist = Twist()
                stop_twist.linear.x = 0.0
                stop_twist.angular.z = 0.0
                self.pub_cmd_vel_circle.publish(stop_twist)

                self.state = 'DONE'
            else:
                # Success condition: red sign centered again
                if self.x_red is not None:
                    err_px = self.x_red - self.image_center_x

                    if abs(err_px) <= self.circle_center_tolerance_px:
                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour CIRCLE_RIGHT: x_red centered '
                                f'(err_px={err_px:.1f}) -> IDLE (restart loop).'
                            )

                        stop_twist = Twist()
                        stop_twist.linear.x = 0.0
                        stop_twist.angular.z = 0.0
                        self.pub_cmd_vel_circle.publish(stop_twist)

                        self.state = 'IDLE'
                    else:
                        # Keep circling clockwise
                        twist = Twist()
                        twist.linear.x = self.circle_linear_x
                        twist.angular.z = -self.circle_angular_z   # CW
                        self.pub_cmd_vel_circle.publish(twist)

                        if self.debug_logging:
                            self.get_logger().info(
                                f'Parcour CIRCLE_RIGHT: x_red_err={err_px:.1f} px, '
                                f'v={twist.linear.x:.3f}, omega={twist.angular.z:.3f}'
                            )
                else:
                    # No red detected -> keep circling CW
                    twist = Twist()
                    twist.linear.x = self.circle_linear_x
                    twist.angular.z = -self.circle_angular_z
                    self.pub_cmd_vel_circle.publish(twist)

                    if self.debug_logging:
                        self.get_logger().info('Parcour CIRCLE_RIGHT: no x_red, keep circling.')


        # Save for debug overlay
        self.current_yaw_init = yaw_init
        self.current_yaw_tar = yaw_tar
        self.current_x_par = x_par

        # Debug overlay
        if self.debug_visualization and self.debug_image is not None:
            self._draw_debug_overlay()

    # === Helpers ===

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
    
    def _is_corridor_clear(self) -> bool:
        """
        Check if there is any LiDAR object inside a forward corridor towards the red sign.

        Corridor is defined by:
          - |lateral_offset| <= corridor_width_m / 2
          - 0 < distance <= corridor_max_distance_m
        """
        if not self.lidar_objects:
            # No objects -> assume clear
            return True

        half_w = self.corridor_width_m / 2.0
        d_max = self.corridor_max_distance_m

        for obj in self.lidar_objects:
            d = obj['distance']
            ang_deg = obj['angle_deg']
            if d <= 0.0 or d > d_max:
                continue
            theta = math.radians(ang_deg)
            lateral = d * math.sin(theta)

            if abs(lateral) <= half_w:
                # Object lies in the corridor
                return False

        return True


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

        # Draw x_par (planned target, if set)
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
            cv2.putText(
                img, f'{dist:.2f}m',
                (x_pix - 15, y_line - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1
            )

        # Show current state
        cv2.putText(
            img, f'state={self.state}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
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
