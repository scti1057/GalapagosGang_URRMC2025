#!/usr/bin/env python3

import os
import cv2
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32


class WhiteYellowCalibrationNode(Node):
    def __init__(self):
        super().__init__('white_yellow_calibration_node')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter(
            'config_path',
            '/home/duckie5/GalapagosGang_URRMC2025/src/tb3_lane_fusion/tb3_lane_fusion/config/lane_bev_params.yaml'
        )

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self._config_path = self.get_parameter('config_path').get_parameter_value().string_value

        self._bridge = CvBridge()
        self._window = "calibration"
        self.image = None
        self.played = False

        # Color modes and HSV keys (trackbars)
        self.mode_map = {0: 'white', 1: 'gelb', 2: 'red'}
        self.current_mode = 'white'
        self._hsv_keys = ['hl', 'hh', 'sl', 'sh', 'vl', 'vh']

        # Map color mode -> (lower_param_name, upper_param_name) in lane_bev_params.yaml
        self._color_to_param = {
            'white': ('white_hsv_lower', 'white_hsv_upper'),
            'gelb':  ('yellow_hsv_lower', 'yellow_hsv_upper'),
            'red':   ('red_hsv_lower', 'red_hsv_upper'),
        }

        # --- Load full YAML config ---
        self.full_conf = {}
        try:
            with open(self._config_path, 'r') as f:
                self.full_conf = yaml.safe_load(f) or {}
            self.get_logger().info(f'Loaded HSV config from: {self._config_path}')
        except FileNotFoundError:
            self.get_logger().warn(
                f'Config file not found at {self._config_path}. '
                'Starting with default values (all zeros).'
            )
            self.full_conf = {}

        # Ensure lane_bev_node / ros__parameters exist
        self.node_key = 'lane_bev_node'
        if self.node_key not in self.full_conf:
            self.full_conf[self.node_key] = {}
        if 'ros__parameters' not in self.full_conf[self.node_key]:
            self.full_conf[self.node_key]['ros__parameters'] = {}

        self.params = self.full_conf[self.node_key]['ros__parameters']

        # Ensure all lower/upper arrays exist
        for mode, (lower_name, upper_name) in self._color_to_param.items():
            self.params.setdefault(lower_name, [0, 0, 0])
            self.params.setdefault(upper_name, [0, 0, 0])

        # --- Publisher for optional sound feedback ---
        self.sound_pub = self.create_publisher(Int32, "/play_sound_trigger", 1)

        # --- Subscribe to camera topic ---
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # --- OpenCV window / trackbars ---
        cv2.namedWindow(self._window)
        self.init_trackbars()

        self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")
        self.get_logger().info(f"Using config file: {self._config_path}")

        # --- Timer for GUI update (10 Hz) ---
        self.timer = self.create_timer(0.1, self.timer_callback)

    # ------------- Helpers to map YAML <-> trackbars -------------

    def _get_mode_hsv_from_yaml(self, mode):
        """Return dict hl/hh/sl/sh/vl/vh for given mode based on YAML arrays."""
        lower_name, upper_name = self._color_to_param[mode]
        lower = self.params.get(lower_name, [0, 0, 0])
        upper = self.params.get(upper_name, [0, 0, 0])

        # Ensure length 3
        lower = (lower + [0, 0, 0])[:3]
        upper = (upper + [0, 0, 0])[:3]

        return {
            'hl': int(lower[0]),
            'sl': int(lower[1]),
            'vl': int(lower[2]),
            'hh': int(upper[0]),
            'sh': int(upper[1]),
            'vh': int(upper[2]),
        }

    def _write_mode_hsv_to_yaml(self, mode, vals):
        """Write current trackbar vals back into YAML arrays."""
        lower_name, upper_name = self._color_to_param[mode]
        lower = [int(vals['hl']), int(vals['sl']), int(vals['vl'])]
        upper = [int(vals['hh']), int(vals['sh']), int(vals['vh'])]
        self.params[lower_name] = lower
        self.params[upper_name] = upper

    def _save_current_mode_to_yaml(self):
        """Save function (triggered by keyboard 's')."""
        vals = self.get_trackbar_values()
        self._write_mode_hsv_to_yaml(self.current_mode, vals)
        try:
            with open(self._config_path, 'w') as f:
                yaml.dump(self.full_conf, f)
            self.get_logger().info(
                f"[SAVE] HSV values for '{self.current_mode}' written to {self._config_path}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to write config file: {e}")

    # ---------------- OpenCV trackbars ----------------

    def init_trackbars(self):
        def nothing(x):
            pass

        # Initialize from current_mode's values
        vals = self._get_mode_hsv_from_yaml(self.current_mode)

        for name in self._hsv_keys:
            cv2.createTrackbar(name, self._window, vals[name], 255, nothing)

        # Trackbar to switch between color modes: 0=white, 1=gelb, 2=red
        cv2.createTrackbar("mode", self._window, 0, 2, self.switch_mode)

    def switch_mode(self, val):
        # Change current color mode and update trackbars
        self.current_mode = self.mode_map.get(val, 'white')
        self.get_logger().info(f"Switched mode to: {self.current_mode}")
        self.update_trackbars()

    def update_trackbars(self):
        # Set trackbars to current HSV values for selected mode
        vals = self._get_mode_hsv_from_yaml(self.current_mode)
        for name in self._hsv_keys:
            cv2.setTrackbarPos(name, self._window, vals[name])

    def get_trackbar_values(self):
        # Read current HSV values from trackbars
        vals = {}
        for name in self._hsv_keys:
            vals[name] = cv2.getTrackbarPos(name, self._window)
        return vals

    # ---------------- ROS callbacks ----------------

    def image_callback(self, msg: Image):
        try:
            self.image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    # ---------------- Main GUI / processing loop ----------------

    def timer_callback(self):
        if self.image is None:
            return

        # Play sound once at startup
        if not self.played:
            self.get_logger().info("Publishing to /play_sound_trigger with value 1")
            msg = Int32()
            msg.data = 1
            self.sound_pub.publish(msg)
            self.played = True

        # Get HSV values from trackbars and apply mask
        vals = self.get_trackbar_values()
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower = np.array([vals['hl'], vals['sl'], vals['vl']], dtype=np.uint8)
        upper = np.array([vals['hh'], vals['sh'], vals['vh']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Visualize mask: ALWAYS GREEN
        output = self.image.copy()
        output[mask > 0] = (0, 255, 0)

        # Overlay help text
        cv2.putText(
            output,
            "mode: white=0, gelb=1, red=2 | SAVE: press 's' | ESC to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "mode: white=0, gelb=1, red=2 | SAVE: press 's' | ESC to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self._window, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Keyboard save
            self.get_logger().info("[KEYBOARD] 's' pressed → saving current mode")
            self._save_current_mode_to_yaml()

        elif key == 27:  # ESC
            self.get_logger().info("ESC pressed, shutting down node.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = WhiteYellowCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
