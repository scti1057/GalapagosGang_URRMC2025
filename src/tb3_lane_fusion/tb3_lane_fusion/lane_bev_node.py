#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import numpy as np


class LaneBevNode(Node):
    def __init__(self):
        super().__init__('lane_bev_node')

        # --- Parameters ---
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('bev_width', 920)
        self.declare_parameter('bev_height', 480)
        self.declare_parameter('output_frame', 'base_footprint')

        # Undistortion tuning
        # 'auto'    -> use CameraInfo.distortion_model (if == 'fisheye' then fisheye, else pinhole)
        # 'fisheye' -> force fisheye model
        # 'pinhole' -> force pinhole/plumb_bob model
        self.declare_parameter('undistort_model', 'auto')
        self.declare_parameter('fisheye_balance', 0.0)  # 0.0 .. 1.0

        # Read parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.bev_width = self.get_parameter('bev_width').get_parameter_value().integer_value
        self.bev_height = self.get_parameter('bev_height').get_parameter_value().integer_value
        self.output_frame = self.get_parameter('output_frame').get_parameter_value().string_value

        self.bridge = CvBridge()

        # --- Subscriptions / publishers ---
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.bev_pub = self.create_publisher(Image, 'lane_bev/image', 10)
        self.mask_pub = self.create_publisher(Image, 'lane_bev/mask', 10)
        self.mask_white_pub = self.create_publisher(Image, 'lane_bev/mask_white', 10)
        self.mask_yellow_pub = self.create_publisher(Image, 'lane_bev/mask_yellow', 10)
        self.calib_pub = self.create_publisher(Image, 'lane_bev/image_calib', 10)

        # --- Camera calibration / undistortion ---
        self.camera_info_received = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.distortion_model = None
        self.undistort_map1 = None
        self.undistort_map2 = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',   # adjust if your topic is different
            self.camera_info_callback,
            1
        )

        # --- Homography (old V-shape style) ---
        # src: trapezoid in original (UNDISTORTED) image where the floor is visible
        # -> you can tune these in lane_bev/image_calib
        self.src_points = np.float32([
            [195, 220],  # top-left in image
            [435, 220],  # top-right
            [40,  470],  # bottom-left
            [600, 470],  # bottom-right
        ])

        # dst: rectangle in BEV with side margins => V-like shape in lane_bev/image
        self.dst_points = np.float32([
            [200,                  0],                   # top-left in BEV
            [self.bev_width - 200, 0],                   # top-right
            [200,                  self.bev_height - 1], # bottom-left
            [self.bev_width - 200, self.bev_height - 1], # bottom-right
        ])

        self.M = None  # homography matrix, computed lazily

        self.get_logger().info(
            f'lane_bev_node started, subscribing to {image_topic}, '
            f'BEV size {self.bev_width}x{self.bev_height}'
        )

    # ----------------- camera_info / undistortion -----------------

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.get_logger().info('Received camera_info, initializing undistortion maps')

        self.distortion_model = msg.distortion_model
        K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float32)
        image_size = (msg.width, msg.height)

        self.camera_matrix = K
        self.dist_coeffs = D

        # Read tuning parameters
        undistort_model = self.get_parameter('undistort_model').get_parameter_value().string_value
        fisheye_balance = float(self.get_parameter('fisheye_balance').get_parameter_value().double_value)

        # Decide whether to use fisheye model
        # 'auto' = use CameraInfo.distortion_model if it says 'fisheye'
        use_fisheye = False
        if undistort_model == 'fisheye':
            use_fisheye = True
        elif undistort_model == 'pinhole':
            use_fisheye = False
        else:
            # auto
            use_fisheye = (self.distortion_model == 'fisheye')

        if use_fisheye:
            self.get_logger().info(f'Using OpenCV fisheye undistortion (balance={fisheye_balance})')

            # Clamp balance to [0,1]
            fisheye_balance = max(0.0, min(1.0, fisheye_balance))

            # New camera matrix with given balance
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.camera_matrix, self.dist_coeffs, image_size, np.eye(3),
                balance=fisheye_balance
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs,
                np.eye(3), K_new, image_size, cv2.CV_16SC2
            )
            self.undistort_map1, self.undistort_map2 = map1, map2

        else:
            # Pinhole / plumb_bob etc.
            self.get_logger().info('Using pinhole undistortion (initUndistortRectifyMap)')
            self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, self.camera_matrix,
                image_size, cv2.CV_16SC2
            )

        self.camera_info_received = True
        self.get_logger().info(f'Undistortion maps initialized using model string: "{self.distortion_model}"')

    # ----------------- homography -----------------

    def compute_homography(self, img_shape):
        h, w = img_shape[:2]
        if (w, h) != (640, 480):
            self.get_logger().warn(
                f'Image size is {w}x{h}, but homography src_points assume 640x480. '
                'You will likely need to adjust src_points.'
            )
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    # ----------------- main image callback -----------------

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # --- undistort fisheye/pinhole image ---
        if self.camera_info_received and self.undistort_map1 is not None:
            cv_image = cv2.remap(
                cv_image,
                self.undistort_map1,
                self.undistort_map2,
                interpolation=cv2.INTER_LINEAR
            )
        # --- end undistort ---

        # --- Debug: draw BEV trapezoid on original (undistorted) image ---
        debug_img = cv_image.copy()

        src_pts_int = self.src_points.astype(np.int32).reshape(-1, 1, 2)

        # Draw polygon edges (red)
        cv2.polylines(debug_img, [src_pts_int], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw corner points with small circles and labels
        labels = ['P0', 'P1', 'P2', 'P3']
        for i, (x, y) in enumerate(self.src_points):
            x_i, y_i = int(x), int(y)
            cv2.circle(debug_img, (x_i, y_i), 5, (0, 255, 0), -1)  # green dot
            cv2.putText(
                debug_img, labels[i],
                (x_i + 5, y_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        # Publish calibration image
        try:
            calib_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            calib_msg.header = msg.header
            # keep camera frame_id here – this is the *camera* frame
            self.calib_pub.publish(calib_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing calibration image: {e}')

        # --- Compute homography if needed ---
        if self.M is None:
            self.compute_homography(cv_image.shape)

        # --- Apply BEV transform ---
        bev = cv2.warpPerspective(
            cv_image,
            self.M,
            (self.bev_width, self.bev_height)
        )
        # bev now has the old "V-like" road band: black on left/right due to dst_points margins

        # --- HSV lane color filtering ---
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        # Yellow lane
        lower_yellow = np.array([20,  80,  80], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White lane: high V, low saturation
        lower_white = np.array([0,   0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # --- Publish BEV image ---
        try:
            bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
            bev_msg.header = msg.header
            bev_msg.header.frame_id = self.output_frame   # override frame
            self.bev_pub.publish(bev_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing BEV image: {e}')

        # --- Publish combined mask ---
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
            mask_msg.header = msg.header
            mask_msg.header.frame_id = self.output_frame
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing mask image: {e}')

        # --- Publish white and yellow masks separately ---
        try:
            white_msg = self.bridge.cv2_to_imgmsg(mask_white, encoding='mono8')
            white_msg.header = msg.header
            white_msg.header.frame_id = self.output_frame
            self.mask_white_pub.publish(white_msg)

            yellow_msg = self.bridge.cv2_to_imgmsg(mask_yellow, encoding='mono8')
            yellow_msg.header = msg.header
            yellow_msg.header.frame_id = self.output_frame
            self.mask_yellow_pub.publish(yellow_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing white/yellow masks: {e}')

        self.get_logger().debug('Published BEV and lane masks')


def main(args=None):
    rclpy.init(args=args)
    node = LaneBevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
