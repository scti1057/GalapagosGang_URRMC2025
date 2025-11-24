#!/usr/bin/env python3

import os
import yaml

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Float64, Bool
from vision_msgs.msg import Detection2DArray
from ament_index_python.packages import get_package_share_directory



class ControlNode(Node):
    """
    Main brain with two modes:

    Modes:
      - "lane_following":
          * Uses lane/x_white_near, lane/x_yellow_near and finish_line
          * Publishes x_tar for lane following with finish-line behavior
      - "parcour":
          * Bridges parcour_node -> yaw_node & drive_node:
              - yaw_init_par, yaw_tar_par -> yaw_init, yaw_tar
              - x_par -> x_tar

    Subscribes (always):
      - lane/x_white_near   (Float64)  -> left_x  (for lane_following mode)
      - lane/x_yellow_near  (Float64)  -> right_x
      - finish_line         (Bool)
      - yaw_init_par        (Float64)  -> from parcour_node
      - yaw_tar_par         (Float64)  -> from parcour_node
      - x_par               (Float64)  -> from parcour_node
      - camera_topic        (CompressedImage) for debug

    Publishes:
      - x_tar      (Float64): target x in pixels -> drive_node
      - yaw_init   (Float64): -> yaw_node
      - yaw_tar    (Float64): -> yaw_node

    In parcour mode:
      - Any new yaw_init_par causes yaw_init to be published
      - Any new yaw_tar_par causes yaw_tar to be published
      - Any new x_par causes x_tar to be published
    """

    def __init__(self):
        super().__init__('control_node')

        # Debug image stuff (always define attributes)
        self.bridge = None
        self.debug_image = None
        self._debug_window = 'control_debug'

        # Hardcode mode for now
        #self.mode = 'lane_following'
        #self.mode = 'parcour'
        self.mode = 'paletting'

        # === Parameters ===
        self.declare_parameter('max_rate_hz', 20.0)
        self.declare_parameter('image_width_px', 640.0)

        # Debug visualization params
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')

        # Extra config file (YAML in share/galapagos_regelt/config)
        self.declare_parameter('config_file', 'control.yaml')

        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value
        self._min_period = 1.0 / float(max_rate)
        self._last_control_time = self.get_clock().now()

        # Base image geometry from param (can be overridden by YAML)
        self.image_width_px = self.get_parameter('image_width_px').get_parameter_value().double_value
        self.image_center_x = self.image_width_px / 2.0

        self.debug_visualization = self.get_parameter('debug_visualization').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # --- Load control.yaml (optional) ---
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        try:
            pkg_share = get_package_share_directory('galapagos_regelt')
            self._config_path = os.path.join(pkg_share, 'config', config_file)
            self.get_logger().info(f'Using control config: {self._config_path}')
            with open(self._config_path, 'r') as f:
                self.conf = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f'Could not load control config "{config_file}", using defaults only: {e}')
            self.conf = {}

        ctrl_cfg = self.conf.get('control', {})

        # Optional override of image width
        if 'image_width_px' in ctrl_cfg:
            self.image_width_px = float(ctrl_cfg['image_width_px'])
            self.image_center_x = self.image_width_px / 2.0

        # Finish-line timing from YAML (defaults match old code)
        finish_cfg = ctrl_cfg.get('finish_line', {})
        self.center_before_stop_duration = float(finish_cfg.get('center_before_stop_s', 1.0))
        self.stop_duration = float(finish_cfg.get('stop_duration_s', 3.0))
        self.yellow_only_duration = float(finish_cfg.get('yellow_only_duration_s', 2.0))

        # Sign behavior tuning
        sign_cfg = ctrl_cfg.get('signs', {})
        self.sign_area_threshold = float(sign_cfg.get('area_threshold', 8000.0))
        self.sign_stop_duration = float(sign_cfg.get('stop_duration_s', 3.0))
        self.sign_turn_duration = float(sign_cfg.get('turn_duration_s', 4.0))
        # Paletting override (lane-following gets paused briefly after a pal command)
        pal_cfg = ctrl_cfg.get('paletting', {})
        # How long after the last paletting command we still consider paletting "active"
        self.pal_cmd_timeout_s = float(pal_cfg.get('pal_cmd_timeout_s', 0.5))


        # After finishing a sign action, ignore that sign for a short time
        self.sign_cooldown_duration = float(sign_cfg.get('cooldown_after_action_s', 5.0))

        # Cooldown timestamps per sign class ('left', 'right', 'stop')
        # values are rclpy.time.Time or None
        self.sign_cooldown_until = {
            'left': None,
            'right': None,
            'stop': None,
        }


        if self.debug_visualization:
            self.bridge = CvBridge()

        # Latest lane inputs
        self.latest_x_white_near = None   # left_x
        self.latest_x_yellow_near = None  # right_x

        # Sign-based overrides (only used in lane_following mode)
        #   'none'       -> normal lane following
        #   'stop'       -> do not publish x_tar (DriveNode stops via timeout)
        #   'turn_left'  -> follow x_yellow_near only for a while
        #   'turn_right' -> follow x_white_near only for a while
        self.sign_state = 'none'
        self.sign_state_end = None  # rclpy.time.Time
        # Paletting override: last time we got a command from paletting_node
        self.last_pal_cmd_time = None  # rclpy.time.Time


        # === Finish-line state machine (used only in lane_following mode) ===
        # phases: 'normal', 'center_before_stop', 'stop', 'center_after_stop', 'yellow_only'
        self.finish_phase = 'normal'

        self.center_before_stop_end = None  # rclpy.time.Time
        self.stop_end = None                # rclpy.time.Time
        self.yellow_only_end = None         # rclpy.time.Time

        self.seen_yellow_after_stop = False


        # QoS: sensor-style, keep last
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Debug camera subscription (optional)
        if self.debug_visualization:
            self.sub_debug_cam = self.create_subscription(
                CompressedImage,
                self.camera_topic,
                self.debug_image_callback,
                sensor_qos
            )
            self.get_logger().info(
                f'Debug visualization enabled, subscribing to camera: {self.camera_topic}'
            )

        # === Subscribers for lane-following ===
        self.sub_white_near = self.create_subscription(
            Float64,
            'lane/x_white_near',
            self.white_near_callback,
            sensor_qos
        )

        self.sub_yellow_near = self.create_subscription(
            Float64,
            'lane/x_yellow_near',
            self.yellow_near_callback,
            sensor_qos
        )

        self.sub_finish_line = self.create_subscription(
            Bool,
            'finish_line',
            self.finish_line_callback,
            sensor_qos
        )

        # === Subscriber for YOLO traffic sign detections ===
        self.sub_signs = self.create_subscription(
            Detection2DArray,
            '/yolo/sign_detections',
            self.sign_detections_callback,
            sensor_qos
        )


        # === Subscribers for parcour bridging ===
        self.sub_yaw_init_par = self.create_subscription(
            Float64,
            'yaw_init_par',
            self.yaw_init_par_callback,
            sensor_qos
        )

        self.sub_yaw_tar_par = self.create_subscription(
            Float64,
            'yaw_tar_par',
            self.yaw_tar_par_callback,
            sensor_qos
        )

        self.sub_x_par = self.create_subscription(
            Float64,
            'x_par',
            self.x_par_callback,
            sensor_qos
        )

        # === Subscribers for paletting bridging ===
        self.sub_yaw_init_pal = self.create_subscription(
            Float64,
            'yaw_init_pal',
            self.yaw_init_pal_callback,
            sensor_qos
        )

        self.sub_yaw_tar_pal = self.create_subscription(
            Float64,
            'yaw_tar_pal',
            self.yaw_tar_pal_callback,
            sensor_qos
        )

        self.sub_x_pal = self.create_subscription(
            Float64,
            'x_pal',
            self.x_pal_callback,
            sensor_qos
        )

        

        # === Publishers ===
        # To drive_node
        self.pub_x_tar = self.create_publisher(Float64, 'x_tar', 10)
        # To yaw_node
        self.pub_yaw_init = self.create_publisher(Float64, 'yaw_init', 10)
        self.pub_yaw_tar = self.create_publisher(Float64, 'yaw_tar', 10)

        # === Control loop ===
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(
            f'ControlNode started (mode={self.mode}, image_width_px={self.image_width_px}).'
        )

    # === Callbacks: lane-following inputs ===

    def white_near_callback(self, msg: Float64):
        self.latest_x_white_near = msg.data

    def yellow_near_callback(self, msg: Float64):
        self.latest_x_yellow_near = msg.data

        # If we're in "center_after_stop" and see yellow again, move to yellow-only phase
        if self.finish_phase == 'center_after_stop' and not self.seen_yellow_after_stop:
            self.seen_yellow_after_stop = True
            now = self.get_clock().now()
            self.finish_phase = 'yellow_only'
            self.yellow_only_end = now + Duration(seconds=self.yellow_only_duration)
            self.get_logger().info(
                f'Yellow lane seen again, switching to yellow-only for {self.yellow_only_duration}s.'
            )

    def finish_line_callback(self, msg: Bool):
        if not msg.data:
            return

        # Only trigger when we're in normal mode
        if self.finish_phase != 'normal':
            self.get_logger().debug('Finish line detected but sequence already in progress, ignoring.')
            return

        now = self.get_clock().now()
        self.finish_phase = 'center_before_stop'
        self.center_before_stop_end = now + Duration(seconds=self.center_before_stop_duration)
        self.stop_end = self.center_before_stop_end + Duration(seconds=self.stop_duration)

        self.yellow_only_end = None
        self.seen_yellow_after_stop = False

        self.get_logger().info(
            f'Finish line detected: center {self.center_before_stop_duration}s, '
            f'stop {self.stop_duration}s, center until yellow, then yellow-only '
            f'{self.yellow_only_duration}s.'
        )

    def _compute_default_lane_x(self, left_x, right_x):
        """
        Duckietown-style heuristic:
          - If both lines are visible and correctly ordered: use the middle
          - Otherwise extrapolate from a single line
        """
        x_tar = None
        if left_x is not None and right_x is not None and left_x > right_x:
            x_tar = (left_x + right_x) / 2.0
        elif right_x is not None:
            x_tar = right_x + 238.0
        elif left_x is not None:
            x_tar = left_x - 244.0
        return x_tar


    def sign_detections_callback(self, msg: Detection2DArray):
        """
        React to YOLO sign detections while in lane_following mode.

        We look at classes 'left', 'right', 'stop' and take the one with the
        largest bounding-box area. If its area exceeds sign_area_threshold,
        we trigger a short-lived sign_state.
        """
        # Only relevant in lane_following mode
        if self.mode != 'lane_following':
            return

        # Already executing a sign behavior -> wait until it finishes
        if self.sign_state != 'none':
            return

        now = self.get_clock().now()

        best_class = None
        best_area = 0.0

        for det in msg.detections:
            if not det.results:
                continue
            class_id = det.results[0].hypothesis.class_id
            if class_id not in ('left', 'right', 'stop'):
                continue

            # Check cooldown: ignore this class if still in ignore window
            cooldown_until = self.sign_cooldown_until.get(class_id)
            if cooldown_until is not None and now < cooldown_until:
                # Still cooling down for this sign -> ignore this detection
                continue

            w = det.bbox.size_x
            h = det.bbox.size_y
            area = float(w * h)

            if area > best_area:
                best_area = area
                best_class = class_id

        if best_class is None:
            return

        # Require minimum size (close sign)
        if best_area < self.sign_area_threshold:
            return

        # (now is already defined above)
        if best_class == 'stop':
            self.sign_state = 'stop'
            self.sign_state_end = now + Duration(seconds=self.sign_stop_duration)
            self.get_logger().info(
                f'ControlNode: STOP sign detected (area={best_area:.0f}) '
                f'-> stopping for {self.sign_stop_duration:.1f}s.'
            )
        elif best_class == 'left':
            self.sign_state = 'turn_left'
            self.sign_state_end = now + Duration(seconds=self.sign_turn_duration)
            self.get_logger().info(
                f'ControlNode: LEFT sign detected (area={best_area:.0f}) '
                f'-> follow yellow_near for {self.sign_turn_duration:.1f}s.'
            )
        elif best_class == 'right':
            self.sign_state = 'turn_right'
            self.sign_state_end = now + Duration(seconds=self.sign_turn_duration)
            self.get_logger().info(
                f'ControlNode: RIGHT sign detected (area={best_area:.0f}) '
                f'-> follow white_near for {self.sign_turn_duration:.1f}s.'
            )


    def debug_image_callback(self, msg: CompressedImage):
        """Store latest camera image for debug visualization."""
        if not msg.data:
            return
        if self.bridge is None:
            return
        try:
            self.debug_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to convert debug image: {e}')

    # === Callbacks: parcour bridging ===

    def yaw_init_par_callback(self, msg: Float64):
        """
        Bridge yaw_init_par -> yaw_init when in parcour mode.
        """
        if self.mode != 'parcour':
            return
        self.pub_yaw_init.publish(Float64(data=float(msg.data)))

    def yaw_tar_par_callback(self, msg: Float64):
        """
        Bridge yaw_tar_par -> yaw_tar when in parcour mode.
        """
        if self.mode != 'parcour':
            return
        self.pub_yaw_tar.publish(Float64(data=float(msg.data)))

    def x_par_callback(self, msg: Float64):
        """
        Bridge x_par -> x_tar when in parcour mode.
        """
        if self.mode != 'parcour':
            return
        self.pub_x_tar.publish(Float64(data=float(msg.data)))

    def yaw_init_pal_callback(self, msg: Float64):
        """
        Bridge yaw_init_pal -> yaw_init when in paletting mode.
        Also marks paletting override as active for a short time.
        """
        if self.mode != 'paletting':
            return
        self.last_pal_cmd_time = self.get_clock().now()
        self.pub_yaw_init.publish(Float64(data=float(msg.data)))

    def yaw_tar_pal_callback(self, msg: Float64):
        """
        Bridge yaw_tar_pal -> yaw_tar when in paletting mode.
        Also marks paletting override as active for a short time.
        """
        if self.mode != 'paletting':
            return
        self.last_pal_cmd_time = self.get_clock().now()
        self.pub_yaw_tar.publish(Float64(data=float(msg.data)))

    def x_pal_callback(self, msg: Float64):
        """
        Bridge x_pal -> x_tar when in paletting mode.
        Also marks paletting override as active for a short time.
        """
        if self.mode != 'paletting':
            return
        self.last_pal_cmd_time = self.get_clock().now()
        self.pub_x_tar.publish(Float64(data=float(msg.data)))



    # === Control loop ===

    def control_loop(self):
        now = self.get_clock().now()
        elapsed = (now - self._last_control_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_control_time = now

        # In parcour mode, we only bridge in the callbacks, no lane logic here
        if self.mode == 'parcour':
            return

        # Is paletting override currently active? (recent paletting command)
        pal_active = False
        if self.mode == 'paletting' and self.last_pal_cmd_time is not None:
            pal_elapsed = (now - self.last_pal_cmd_time).nanoseconds / 1e9
            if pal_elapsed < self.pal_cmd_timeout_s:
                pal_active = True


        # --- Expire sign state if its timer ran out ---
        if self.sign_state != 'none' and self.sign_state_end is not None:
            if now >= self.sign_state_end:
                finished_state = self.sign_state

                # Start cooldown for the corresponding sign class
                sign_key = None
                if finished_state == 'stop':
                    sign_key = 'stop'
                elif finished_state == 'turn_left':
                    sign_key = 'left'
                elif finished_state == 'turn_right':
                    sign_key = 'right'

                if sign_key is not None and self.sign_cooldown_duration > 0.0:
                    self.sign_cooldown_until[sign_key] = now + Duration(
                        seconds=self.sign_cooldown_duration
                    )
                    self.get_logger().info(
                        f'ControlNode: sign state {finished_state} finished, '
                        f'ignoring "{sign_key}" signs for '
                        f'{self.sign_cooldown_duration:.1f}s.'
                    )
                else:
                    self.get_logger().info(
                        f'ControlNode: sign state {finished_state} finished, '
                        f'no cooldown configured.'
                    )

                self.sign_state = 'none'
                self.sign_state_end = None


        phase = self.finish_phase
        x_tar = None

        # --- Finish-line phases ---

        if phase == 'center_before_stop':
            if self.center_before_stop_end is not None and now < self.center_before_stop_end:
                # Keep driving straight
                x_tar = self.image_center_x
            else:
                # Move into stop phase
                self.finish_phase = 'stop'
                return  # start stop phase next cycle

        elif phase == 'stop':
            if self.stop_end is not None and now < self.stop_end:
                # Do not publish x_tar during stop window
                x_tar = None
            else:
                # After stop: drive straight until we see yellow
                self.finish_phase = 'center_after_stop'
                phase = self.finish_phase

        # After this, we might be in center_after_stop, yellow_only, or normal

        if self.finish_phase == 'center_after_stop':
            if not self.seen_yellow_after_stop:
                # Still haven't re-seen yellow: drive straight
                x_tar = self.image_center_x
                # (once yellow is seen, yellow_near_callback will switch to yellow_only)
            else:
                # yellow_near_callback has already switched to yellow_only
                phase = self.finish_phase

        if self.finish_phase == 'yellow_only':
            # Use only yellow for configured duration
            if self.yellow_only_end is not None and now < self.yellow_only_end:
                right_x = self.latest_x_yellow_near
                if right_x is not None:
                    x_tar = right_x + 170.0
                else:
                    # Fallback: if yellow vanished briefly, keep straight
                    x_tar = self.image_center_x
            else:
                # End of yellow-only phase, back to normal lane-following
                self.finish_phase = 'normal'
                self.seen_yellow_after_stop = False
                phase = self.finish_phase

        # --- Normal lane-following if not in special phases ---
        if self.finish_phase == 'normal':
            left_x = self.latest_x_white_near
            right_x = self.latest_x_yellow_near

            if self.sign_state == 'turn_left':
                # Only follow yellow for a few seconds
                if right_x is not None:
                    x_tar = right_x + 238.0
                else:
                    x_tar = self._compute_default_lane_x(left_x, right_x)
            elif self.sign_state == 'turn_right':
                # Only follow white for a few seconds
                if left_x is not None:
                    x_tar = left_x - 244.0
                else:
                    x_tar = self._compute_default_lane_x(left_x, right_x)
            else:
                # No sign override: normal heuristic
                x_tar = self._compute_default_lane_x(left_x, right_x)

        # --- STOP sign override: never publish x_tar while active ---
        if self.sign_state == 'stop':
            x_tar = None

        # --- Paletting override: while pal_active, don't send lane-following x_tar ---
        if self.mode == 'paletting' and pal_active:
            x_tar = None

        # --- Publish x_tar if we have a target ---
        if x_tar is not None:
            self.pub_x_tar.publish(Float64(data=float(x_tar)))


        # --- Debug visualization: draw center, x_tar, and state text ---
        if self.debug_visualization and self.debug_image is not None:
            dbg = self.debug_image.copy()
            h, w = dbg.shape[:2]
            y_line = h - 50

            # Lane middle (image center)
            center_x_int = int(self.image_center_x)
            cv2.circle(dbg, (center_x_int, y_line), 6, (0, 0, 255), -1)
            cv2.putText(
                dbg, 'center',
                (center_x_int - 30, y_line - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1
            )

            # x_tar dot (if available)
            if x_tar is not None:
                x_tar_int = int(x_tar)
                cv2.circle(dbg, (x_tar_int, y_line), 6, (255, 0, 0), -1)
                cv2.putText(
                    dbg, 'x_tar',
                    (x_tar_int - 20, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1
                )

            # Lane-following state text for debug window
            lane_state = 'lane following'
            if self.mode == 'paletting' and pal_active:
                lane_state = 'paletting active'
            elif self.sign_state == 'stop' or self.finish_phase == 'stop':
                lane_state = 'stopping'
            elif self.sign_state == 'turn_left':
                lane_state = 'turning left'
            elif self.sign_state == 'turn_right':
                lane_state = 'turning right'


            cv2.putText(
                dbg, lane_state,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2
            )

            cv2.imshow(self._debug_window, dbg)
            cv2.waitKey(1)


    def destroy_node(self):
        if self.debug_visualization:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
