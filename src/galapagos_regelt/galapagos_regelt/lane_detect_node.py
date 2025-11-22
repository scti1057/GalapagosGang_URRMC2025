#!/usr/bin/env python3

import os
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory


class LaneDetectNode(Node):
    """
    ROS2 version of the Duckietown lane detector for TurtleBot3.

    - Subscribes: /camera/image_raw/compressed (by default, configurable via param)
    - Loads HSV + polygon configs from YAML:
        * white / yellow thresholds
        * lane_image_near, lane_image_far (two ROI polygons)
    - Computes:
        * x_white_near, x_white_far
        * x_yellow_near, x_yellow_far
    - Publishes them as Float64 topics and draws:
        * the polygons
        * a point + text for each x_*
        * the image center
    """

    def __init__(self):
        super().__init__('lane_detect_node')

        # --- Parameters (can be overridden with ROS params) ---
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')
        self.declare_parameter('config_file', 'lane_detect.yaml')
        
        # mimic rospy.Rate(10) but configurable
        self.declare_parameter('max_rate_hz', 10.0)
        self._min_period = 1.0 / float(
            self.get_parameter('max_rate_hz').get_parameter_value().double_value
        )
        self._last_processed_time = self.get_clock().now()

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        # --- Load config file from installed share/galapagos_regelt/config ---
        package_share = get_package_share_directory('galapagos_regelt')
        self._config_path = os.path.join(package_share, 'config', config_file)
        self.get_logger().info(f'Using config file: {self._config_path}')

        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.bridge = CvBridge()
        self.image = None
        self._window = 'lane_detect'

        # --- Pre-compute polygons from config ---
        self.polygon_near = self._create_polygon(self.conf['lane_image_near'])
        self.polygon_far = self._create_polygon(self.conf['lane_image_far'])

        # --- Publishers for the four x-values ---
        self.pub_x_white_near = self.create_publisher(Float64, 'lane/x_white_near', 10)
        self.pub_x_white_far = self.create_publisher(Float64, 'lane/x_white_far', 10)
        self.pub_x_yellow_near = self.create_publisher(Float64, 'lane/x_yellow_near', 10)
        self.pub_x_yellow_far = self.create_publisher(Float64, 'lane/x_yellow_far', 10)

        # QoS tuned for sensor data: no retries, keep only last frame
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscriber to camera compressed image (sensor QoS) ---
        self.subscription = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.image_callback,
            sensor_qos
        )

        self.get_logger().info(f'Subscribing to camera topic: {camera_topic}')

    # === Utils ===

    def _create_polygon(self, cfg):
        """
        Create an OpenCV polygon array from config dict with
        top_left_x, top_left_y, top_right_x, ... etc.
        """
        return np.array([[
            [cfg['top_left_x'], cfg['top_left_y']],
            [cfg['top_right_x'], cfg['top_right_y']],
            [cfg['bottom_right_x'], cfg['bottom_right_y']],
            [cfg['bottom_left_x'], cfg['bottom_left_y']],
        ]], dtype=np.int32)

    # === Callbacks ===

    def image_callback(self, msg: CompressedImage):
        """Convert compressed image to OpenCV image and trigger processing."""
        # Some drivers may publish a few empty frames when starting up
        if not msg.data:
            self.get_logger().debug('Received empty compressed image frame, skipping.')
            return

        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'Failed to convert compressed image: {e}')
            return

        # Process immediately, but the processing itself is rate-limited
        self.process_image()



    # === Core lane detection ===

    def _compute_lane_x_from_polygon(self, polygon, mask, image, mode='left', contour_color=(0, 255, 0)):
        """
        Given:
          - polygon (ROI)
          - binary mask (white OR yellow)
          - image (for drawing)
          - mode: 'left'  -> choose smallest cx (white lane, left side)
                  'right' -> choose largest cx (yellow lane, right side)
        Returns:
          - x coordinate (int) or None
        """
        min_area = 10

        # Apply ROI polygon to the mask
        mask_poly = np.zeros_like(mask)
        cv2.fillPoly(mask_poly, polygon, 255)
        masked = cv2.bitwise_and(mask, mask_poly)

        # Edge detection
        edges = cv2.Canny(cv2.GaussianBlur(masked, (5, 5), 0), 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        selected_x = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])

            if selected_x is None:
                selected_x = cx
            elif mode == 'left' and cx < selected_x:
                selected_x = cx
            elif mode == 'right' and cx > selected_x:
                selected_x = cx

            # Draw contour for debugging
            cv2.drawContours(image, [cnt], -1, contour_color, 2)

        return selected_x

    def process_image(self):
        """Main processing loop, called periodically by the timer."""
        if self.image is None:
            return

        # Enforce max_rate_hz (like rospy.Rate in ROS1)
        now = self.get_clock().now()
        elapsed = (now - self._last_processed_time).nanoseconds / 1e9
        if elapsed < self._min_period:
            return
        self._last_processed_time = now

        image = self.image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV ranges from config
        wh = self.conf['white']
        yl = self.conf['yellow']

        # Binary masks for white and yellow
        mask_white = cv2.inRange(
            hsv,
            (wh['hl'], wh['sl'], wh['vl']),
            (wh['hh'], wh['sh'], wh['vh'])
        )
        mask_yellow = cv2.inRange(
            hsv,
            (yl['hl'], yl['sl'], yl['vl']),
            (yl['hh'], yl['sh'], yl['vh'])
        )

        # Morphological cleanup (same idea as Duckietown code)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # --- Compute x-values for each color and polygon ---

        # white lane: "left side" -> mode='left'
        x_white_near = self._compute_lane_x_from_polygon(
            self.polygon_near, mask_white, image,
            mode='left', contour_color=(0, 255, 0)
        )
        x_white_far = self._compute_lane_x_from_polygon(
            self.polygon_far, mask_white, image,
            mode='left', contour_color=(0, 200, 0)
        )

        # yellow lane: "right side" -> mode='right'
        x_yellow_near = self._compute_lane_x_from_polygon(
            self.polygon_near, mask_yellow, image,
            mode='right', contour_color=(0, 255, 255)
        )
        x_yellow_far = self._compute_lane_x_from_polygon(
            self.polygon_far, mask_yellow, image,
            mode='right', contour_color=(0, 200, 200)
        )

        # --- Publish the values if available ---

        if x_white_near is not None:
            self.pub_x_white_near.publish(Float64(data=float(x_white_near)))
        if x_white_far is not None:
            self.pub_x_white_far.publish(Float64(data=float(x_white_far)))
        if x_yellow_near is not None:
            self.pub_x_yellow_near.publish(Float64(data=float(x_yellow_near)))
        if x_yellow_far is not None:
            self.pub_x_yellow_far.publish(Float64(data=float(x_yellow_far)))

        # --- Visualization (points + text + polygons + center) ---

        h, w = image.shape[:2]
        base_y = h - 50  # same idea as Duckietown example
        dy = 20          # vertical offset between labels

        def draw_point_with_label(x, y, label, color):
            x_int = int(x)
            cv2.circle(image, (x_int, y), 6, color, -1)
            cv2.putText(
                image, label,
                (x_int - 40, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1
            )

        # near row (closer polygon)
        if x_white_near is not None:
            draw_point_with_label(x_white_near, base_y, 'x_white_near', (255, 0, 255))
        if x_yellow_near is not None:
            draw_point_with_label(x_yellow_near, base_y + dy, 'x_yellow_near', (0, 128, 255))

        # far row (further polygon)
        if x_white_far is not None:
            draw_point_with_label(x_white_far, base_y - dy, 'x_white_far', (255, 0, 200))
        if x_yellow_far is not None:
            draw_point_with_label(x_yellow_far, base_y - 2 * dy, 'x_yellow_far', (0, 200, 255))

        # Draw reference center of the image
        center_x = int(w / 2)
        center_y = base_y
        cv2.circle(image, (center_x, center_y), 6, (0, 0, 255), -1)
        cv2.putText(
            image, "Center",
            (center_x - 25, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 1
        )

        # Draw detection polygons (near + far)
        cv2.polylines(image, self.polygon_near, isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.polylines(image, self.polygon_far, isClosed=True, color=(200, 200, 200), thickness=2)

        # Show debug window
        cv2.imshow(self._window, image)
        cv2.waitKey(1)

    # === Node lifecycle ===

    def destroy_node(self):
        """Override destroy_node to close OpenCV window too."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
