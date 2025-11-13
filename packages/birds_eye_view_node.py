#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class BirdsEyeViewNode(DTROS):
    def __init__(self, node_name):
        super(BirdsEyeViewNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()

        # locate configuration file next to this module
        self._config_path = os.path.join(os.path.dirname(__file__), 'birds_eye_view.yaml')

        # === Load YAML configuration ===
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Required parameters from YAML
        self.src_points = np.float32(self.conf["src_points_px"])  # expect 4 points: [bl, br, tl, tr]
        self.dst_width = int(self.conf["dst_size_px"][0])
        self.dst_height = int(self.conf["dst_size_px"][1])

        # Zielpunkte automatisch erzeugen
        self.dst_points = np.float32([
            [0, self.dst_height],                # unten links
            [self.dst_width, self.dst_height],   # unten rechts
            [0, 0],                              # oben links
            [self.dst_width, 0]                  # oben rechts
        ])

        # Transformationsmatrix vorberechnen
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # Optional camera calibration for undistortion
        # If YAML contains 'camera_matrix' and 'dist_coeffs' we will undistort frames before warping.
        self.do_undistort = False
        self.undistort_map = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.use_fisheye = False
        if "camera_matrix" in self.conf and "dist_coeffs" in self.conf:
            try:
                K = np.array(self.conf["camera_matrix"], dtype=np.float64)
                D = np.array(self.conf["dist_coeffs"], dtype=np.float64)
                # Accept either flat list (9 elements) or nested 3x3
                if K.size == 9:
                    K = K.reshape((3, 3))
                # Dist coeffs may be 4 (fisheye) or 5+ (classic)
                self.camera_matrix = K
                self.dist_coeffs = D
                # optional flag in YAML to indicate fisheye model
                self.use_fisheye = bool(self.conf.get("fisheye", False))
                self.do_undistort = True
            except Exception:
                # keep undistort disabled if parsing fails
                self.do_undistort = False

        # If undistort is requested, we'll create mapping lazily when first frame arrives (need frame size)

        self.image = None

        # Subscriber
        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

        self._window_original = "camera"
        self._window_bev = "birds-eye-view"

    def image_callback(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def get_birds_eye_view(self, image):
        return cv2.warpPerspective(image, self.M, (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR)

    def _prepare_undistort_map(self, image_shape):
        # image_shape: (height, width)
        h, w = image_shape
        if self.use_fisheye:
            try:
                K = self.camera_matrix
                D = self.dist_coeffs
                newK = K.copy()
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
                self.undistort_map = (map1, map2)
                # undistort the source points so the homography matches undistorted images
                try:
                    pts = self.src_points.reshape(-1, 1, 2).astype(np.float64)
                    und_pts = cv2.fisheye.undistortPoints(pts, K, D, P=newK)
                    und_pts = und_pts.reshape(-1, 2)
                    self.M = cv2.getPerspectiveTransform(np.float32(und_pts), self.dst_points)
                except Exception:
                    pass
            except Exception:
                self.undistort_map = None
        else:
            try:
                K = self.camera_matrix
                D = self.dist_coeffs
                newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newCameraMatrix, (w, h), cv2.CV_16SC2)
                self.undistort_map = (map1, map2)
                # undistort the source points and update homography
                try:
                    pts = self.src_points.reshape(-1, 1, 2).astype(np.float64)
                    und_pts = cv2.undistortPoints(pts, K, D, P=newCameraMatrix)
                    und_pts = und_pts.reshape(-1, 2)
                    self.M = cv2.getPerspectiveTransform(np.float32(und_pts), self.dst_points)
                except Exception:
                    pass
            except Exception:
                self.undistort_map = None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            image = self.image.copy()

            # draw the source quad on a copy of the original for debugging
            vis = image.copy()
            try:
                pts = self.src_points.astype(np.int32)
                cv2.polylines(vis, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                for p in pts:
                    cv2.circle(vis, tuple(p), 4, (0, 255, 0), -1)
            except Exception:
                pass

            # If undistortion is requested, lazily prepare maps and undistort the image
            if self.do_undistort:
                if self.undistort_map is None:
                    self._prepare_undistort_map((image.shape[0], image.shape[1]))
                if self.undistort_map:
                    map1, map2 = self.undistort_map
                    image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

            bev = self.get_birds_eye_view(image)

            # annotate whether undistort is active
            label = f"Undistort: {'ON' if (self.do_undistort and self.undistort_map is not None) else 'OFF'}"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(self._window_original, vis)
            cv2.imshow(self._window_bev, bev)
            cv2.waitKey(1)

            rate.sleep()


if __name__ == '__main__':
    node = BirdsEyeViewNode(node_name='birds_eye_view_node')
    node.run()
