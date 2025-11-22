#!/usr/bin/env python3

import os
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory


class RedSignDetectNode(Node):
    """
    Detect red signs using HSV thresholding with two red hue ranges.

    - Subscribes:
        * camera_topic (CompressedImage), default: /camera/image_raw/compressed

    - Loads from red_sign_detect.yaml:
        * red1, red2: HSV ranges for red (two intervals combined)
        * detect:
            - min_pixels: minimum largest-blob area to consider a sign present
            - big_pixels: threshold to set red_sign_big = True
            - min_blob_pixels: ignore blobs smaller than this
            - kernel_size: morphology kernel size
            - big_state_publish_repeats: how many times to publish on state change

    - Publishes:
        * x_red (Float64): x coordinate (pixels) of sign center if detected
        * red_sign_big (Bool): True/False, but only burst-published N times when changing

    - Debug mode:
        * draw x_red (if present) and the number of red pixels on the image
        * displayed in an OpenCV window "red_sign_debug"
    """

    def __init__(self):
        super().__init__('red_sign_detect_node')

        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')
        self.declare_parameter('config_file', 'red_sign_detect.yaml')
        self.declare_parameter('max_rate_hz', 10.0)
        self.declare_parameter('debug_visualization', False)

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        max_rate = self.get_parameter('max_rate_hz').get_parameter_value().double_value
        self.debug_visualization = self.get_parameter('debug_visualization').get_parameter_value().bool_value

        self._min_period = 1.0 / float(max_rate)
        self._last_processed_time = self.get_clock().now()

        # Load configuration from share/galapagos_regelt/config
        package_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(package_share, 'config', config_file)
        self.get_logger().info(f'Using red sign config: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Two HSV ranges for red
        self.red1 = self.conf['red1']
        self.red2 = self.conf['red2']

        detect_cfg = self.conf.get('detect', {})
        self.min_pixels = int(detect_cfg.get('min_pixels', 300))
        self.big_pixels = int(detect_cfg.get('big_pixels', 1500))
        self.min_blob_pixels = int(detect_cfg.get('min_blob_pixels', 50))
        kernel_size = int(detect_cfg.get('kernel_size', 3))
        self.big_state_publish_repeats = int(detect_cfg.get('big_state_publish_repeats', 10))

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        self.get_logger().info(
            f'Red detect params: min_pixels={self.min_pixels}, big_pixels={self.big_pixels}, '
            f'min_blob_pixels={self.min_blob_pixels}, kernel_size={kernel_size}, '
            f'big_state_publish_repeats={self.big_state_publish_repeats}'
        )

        # State
        self.bridge = CvBridge()
        self.image = None
        self._debug_window = 'red_sign_debug'

        self.last_big_state = False
        self.big_publish_remaining = 0

        # Publishers
        self.pub_x_red = self.create_publisher(Float64, 'x_red', 10)
        self.pub_red_sign_big = self.create_publisher(Bool, 'red_sign_big', 10)

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.subscription = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            sensor_qos
        )

        self.get_logger().info(f'Subscribing to camera topic: {camera_topic}')

    # === Callbacks ===

    def image_callback(self, msg: CompressedImage):
        """Convert compressed image to OpenCV image and trigger processing."""
        if not msg.data:
            self.get_logger().debug('Received empty compressed image frame, skipping.')
            return

        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to convert compressed image: {e}')
            return

        self.process_image()

    # === Main processing ===

    def process_image(self):
        if self.image is None:
            return

        # Rate limiting
        now = self.get_clock().now()
        elapsed = (now - self._last_processed_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_processed_time = now

        image = self.image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Build combined red mask from two HSV ranges
        r1 = self.red1
        r2 = self.red2

        mask1 = cv2.inRange(
            hsv,
            (r1['hl'], r1['sl'], r1['vl']),
            (r1['hh'], r1['sh'], r1['vh'])
        )
        mask2 = cv2.inRange(
            hsv,
            (r2['hl'], r2['sl'], r2['vl']),
            (r2['hh'], r2['sh'], r2['vh'])
        )

        binary = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

        # Connected components to find largest red blob
        sign_area, sign_cx = self._find_largest_blob(binary)

        # Decide presence and "big" state (force Python bool)
        sign_present = bool(sign_area >= self.min_pixels)
        big_state = bool(sign_area >= self.big_pixels)


        # Publish x_red if we have a sign
        if sign_present and sign_cx is not None:
            self.pub_x_red.publish(Float64(data=float(sign_cx)))

        # Handle red_sign_big state changes (burst publish on change)
        if big_state != self.last_big_state:
            self.last_big_state = bool(big_state)
            self.big_publish_remaining = self.big_state_publish_repeats

        if self.big_publish_remaining > 0:
            self.pub_red_sign_big.publish(Bool(data=bool(self.last_big_state)))
            self.big_publish_remaining -= 1

        # Debug visualization
        if self.debug_visualization:
            dbg = image.copy()
            h, w = dbg.shape[:2]
            y_line = h - 50

            # Draw x_red if present
            if sign_present and sign_cx is not None:
                x_int = int(sign_cx)
                cv2.circle(dbg, (x_int, y_line), 6, (0, 0, 255), -1)
                cv2.putText(
                    dbg, 'x_red',
                    (x_int - 20, y_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1
                )

            # Show area of largest red blob
            cv2.putText(
                dbg, f'red_pixels={int(sign_area)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            # Optionally show whether it's "big"
            state_text = 'BIG' if big_state else 'small/none'
            cv2.putText(
                dbg, f'state={state_text}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )

            cv2.imshow(self._debug_window, dbg)
            cv2.waitKey(1)

    def _find_largest_blob(self, binary: np.ndarray):
        """
        Find the largest connected white blob in the binary image.

        Returns:
          - area (int): number of pixels in the largest blob (0 if none)
          - cx (float or None): x centroid of the largest blob in image coordinates
        """
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        best_label = None
        best_area = 0

        # label 0 is background
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])  # cast to Python int
            if area < self.min_blob_pixels:
                continue
            if area > best_area:
                best_area = area
                best_label = label

        if best_label is None:
            return 0, None

        cx = float(centroids[best_label][0])
        return int(best_area), cx


    # === Lifecycle ===

    def destroy_node(self):
        if self.debug_visualization:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedSignDetectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
