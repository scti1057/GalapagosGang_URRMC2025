#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Float64, Bool


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
        # self.mode = 'lane_following'
        self.mode = 'parcour'

        # === Parameters ===
        self.declare_parameter('max_rate_hz', 20.0)
        self.declare_parameter('image_width_px', 640.0)

        # Debug visualization params
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')

        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value
        self._min_period = 1.0 / float(max_rate)
        self._last_control_time = self.get_clock().now()

        self.image_width_px = self.get_parameter('image_width_px').get_parameter_value().double_value
        self.image_center_x = self.image_width_px / 2.0

        self.debug_visualization = self.get_parameter('debug_visualization').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        if self.debug_visualization:
            self.bridge = CvBridge()

        # Latest lane inputs
        self.latest_x_white_near = None   # left_x
        self.latest_x_yellow_near = None  # right_x

        # === Finish-line state machine (used only in lane_following mode) ===
        # phases: 'normal', 'center_before_stop', 'stop', 'center_after_stop', 'yellow_only'
        self.finish_phase = 'normal'

        self.center_before_stop_duration = 1.0   # seconds (center)
        self.stop_duration = 3.0                 # seconds (no x_tar)
        self.yellow_only_duration = 2.0          # seconds

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

    # === Control loop ===

    def control_loop(self):
        now = self.get_clock().now()
        elapsed = (now - self._last_control_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_control_time = now

        # In parcour mode, we only bridge in the callbacks, no lane logic
        if self.mode != 'lane_following':
            return

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
                return
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
            # Use only yellow for 2 seconds
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

            # Duckietown-style heuristic
            if left_x is not None and right_x is not None and left_x > right_x:
                x_tar = (left_x + right_x) / 2.0
            elif right_x is not None:
                x_tar = right_x + 238
            elif left_x is not None:
                x_tar = left_x - 244
            # else: x_tar stays None

        # --- Publish if we have a target ---
        if x_tar is not None:
            self.pub_x_tar.publish(Float64(data=float(x_tar)))

            # Debug visualization: draw center and x_tar on camera image
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

                # x_tar dot
                x_tar_int = int(x_tar)
                cv2.circle(dbg, (x_tar_int, y_line), 6, (255, 0, 0), -1)
                cv2.putText(
                    dbg, 'x_tar',
                    (x_tar_int - 20, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1
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
