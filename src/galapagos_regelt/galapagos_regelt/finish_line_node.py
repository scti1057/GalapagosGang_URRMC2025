#!/usr/bin/env python3

import os
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory


class FinishLineNode(Node):
    """
    Detects a black/white checkered finish line in a given polygonal ROI.

    Approach:
      - Subscribe to /camera/image_raw/compressed (configurable)
      - Load polygon + detection thresholds from YAML
      - Convert ROI to grayscale, threshold (Otsu) to binary
      - For each column inside the polygon, count black/white transitions
      - If the pattern is "busy enough" -> publish Bool(True) on 'finish_line'
    """

    def __init__(self):
        super().__init__('finish_line_node')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')
        self.declare_parameter('config_file', 'finish_line_detect.yaml')

        # Rate limiting parameter (like rospy.Rate)
        self.declare_parameter('max_rate_hz', 10.0)
        self._min_period = 1.0 / float(
            self.get_parameter('max_rate_hz').get_parameter_value().double_value
        )
        self._last_processed_time = self.get_clock().now()

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        # --- Load configuration from share/galapagos_regelt/config ---
        package_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(package_share, 'config', config_file)
        self.get_logger().info(f'Using config file: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Polygon ROI config
        roi_cfg = self.conf['finish_line_roi']
        self.finish_polygon = self._create_polygon(roi_cfg)

        # White HSV range from config
        self.white = self.conf['white']

        # Detection thresholds
        det_cfg = self.conf.get('finish_line', {})
        self.min_mean_transitions = float(det_cfg.get('min_mean_transitions', 6.0))
        self.min_active_rows = int(det_cfg.get('min_active_rows', 20))
        self.min_blob_pixels = int(det_cfg.get('min_blob_pixels', 50))

        self.bridge = CvBridge()
        self.image = None
        self._window = 'finish_line_detect'

        # Publisher: Bool topic that says if a finish line is seen in this frame
        self.pub_finish_line = self.create_publisher(Bool, 'finish_line', 10)

        # QoS tuned for sensor data (same idea as in lane_detect_node)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber to compressed camera topic
        self.subscription = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            sensor_qos
        )

        self.get_logger().info(f'Subscribing to camera topic: {camera_topic}')

    # === Utilities ===

    def _create_polygon(self, cfg):
        """Create an OpenCV polygon array from config dict."""
        return np.array([[
            [cfg['top_left_x'], cfg['top_left_y']],
            [cfg['top_right_x'], cfg['top_right_y']],
            [cfg['bottom_right_x'], cfg['bottom_right_y']],
            [cfg['bottom_left_x'], cfg['bottom_left_y']],
        ]], dtype=np.int32)

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

        # Process immediately, but the processing is rate-limited inside
        self.process_image()

    # === Main processing ===

    def process_image(self):
        """Detect finish line pattern and publish Bool."""
        if self.image is None:
            return

        # Rate limiting (max_rate_hz)
        now = self.get_clock().now()
        elapsed = (now - self._last_processed_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_processed_time = now

        image = self.image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask from polygon (single-channel mask)
        mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_full, self.finish_polygon, 255)

        # Work on the bounding rectangle of the polygon for efficiency
        x, y, w, h = cv2.boundingRect(self.finish_polygon[0])
        hsv_roi = hsv[y:y + h, x:x + w]
        mask_roi = mask_full[y:y + h, x:x + w]

        # Threshold using the configured white HSV range
        wh = self.white
        mask_white = cv2.inRange(
            hsv_roi,
            (wh['hl'], wh['sl'], wh['vl']),
            (wh['hh'], wh['sh'], wh['vh'])
        )

        # Keep only pixels inside the polygon
        binary = cv2.bitwise_and(mask_white, mask_roi)

        # Optional morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Remove small connected white blobs
        binary_clean = self._remove_small_components(binary, self.min_blob_pixels)

        detected = self._detect_pattern(binary_clean, mask_roi)



        # Publish result
        self.pub_finish_line.publish(Bool(data=detected))

        # --- Visualization ---
        # Draw polygon
        cv2.polylines(image, self.finish_polygon, isClosed=True, color=(255, 255, 255), thickness=2)

        # Draw detection text
        text = 'FINISH LINE' if detected else 'no finish line'
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(
            image, text,
            (x, y - 10 if y - 10 > 20 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2
        )

        # Show for debugging
        cv2.imshow(self._window, binary_clean)
        #cv2.imshow(self._window, image)
        cv2.waitKey(1)

    def _remove_small_components(self, binary: np.ndarray, min_pixels: int) -> np.ndarray:
        """
        Remove small connected white components (blobs) with area < min_pixels.
        binary: uint8 image with values {0, 255}
        """
        # Ensure proper type for connectedComponentsWithStats
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Get labels and stats for each connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Start from a copy or a clean mask; here we clean in-place
        cleaned = binary.copy()

        # label 0 is background, skip it
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_pixels:
                # remove this small blob
                cleaned[labels == label] = 0

        return cleaned


    def _detect_pattern(self, binary: np.ndarray, mask_roi: np.ndarray) -> bool:
        """
        Count black/white transitions per row inside the polygon.
        A horizontal checkered / striped finish line produces many transitions.

        Returns True if pattern exceeds the configured thresholds.
        """
        h, w = binary.shape

        transition_counts = []
        for row in range(h):
            # Only consider columns where mask is inside the polygon
            valid_cols = mask_roi[row, :] > 0
            row_values = binary[row, :][valid_cols]

            # Skip rows that don't really intersect the polygon
            if row_values.size < 2:
                continue

            transitions = np.count_nonzero(row_values[1:] != row_values[:-1])
            transition_counts.append(transitions)

        if not transition_counts:
            return False

        mean_transitions = float(np.mean(transition_counts))
        active_rows = len(transition_counts)

        print(f"mean_transitions: ", mean_transitions)
        print(f"active_rows: ", active_rows)

        self.get_logger().debug(
            f'mean_transitions={mean_transitions:.2f}, active_rows={active_rows}'
        )

        if active_rows < self.min_active_rows:
            return False

        return mean_transitions >= self.min_mean_transitions


    # === Lifecycle ===

    def destroy_node(self):
        """Close OpenCV window properly."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FinishLineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
