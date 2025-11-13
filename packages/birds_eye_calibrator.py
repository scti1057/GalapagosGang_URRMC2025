#!/usr/bin/env python3
"""
Interactive Birds-Eye View and Distortion Calibrator

Features:
- Subscribe to compressed camera topic and show live image
- Trackbars to adjust the four source points (8 sliders)
- Trackbars for destination size (width, height)
- Trackbars to tweak radial/tangential distortion coefficients (k1,k2,p1,p2,k3)
- Toggle to choose fisheye model (0/1)
- Preview undistorted frame and resulting BEV
- Press 's' to save configuration to packages/birds_eye_view.yaml
- Press 'q' to quit

Usage:
  export VEHICLE_NAME=<your_vehicle>
  python3 packages/birds_eye_calibrator.py

Notes:
- Distortion sliders map integer ranges to small floating values; sliders are scaled by 1e-3.
"""

import os
import cv2
import yaml
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class BirdsEyeCalibrator:
    def __init__(self, yaml_path=None):
        rospy.init_node('birds_eye_calibrator', anonymous=True)

        self._vehicle_name = os.environ.get('VEHICLE_NAME', None)
        if not self._vehicle_name:
            raise RuntimeError('Please set VEHICLE_NAME environment variable')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()

        # yaml path next to module by default
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'birds_eye_view.yaml')
        self.yaml_path = yaml_path

        # load existing config if any
        self.conf = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, 'r') as f:
                    self.conf = yaml.safe_load(f) or {}
            except Exception:
                self.conf = {}

        # placeholder image
        self.image = None

        # default image size guess until we receive a frame
        self.img_h = 480
        self.img_w = 640

        # default src points from YAML or fallback
        default_src = self.conf.get('src_points_px', [[0, 432], [622, 432], [144, 283], [473, 283]])
        self.src = np.array(default_src, dtype=np.int32)

        # dst size from YAML or fallback
        dst = self.conf.get('dst_size_px', [400, 600])
        self.dst_w = int(dst[0])
        self.dst_h = int(dst[1])

        # distortion coefficients initial values (scaled ints -> float = val/1000)
        camK = self.conf.get('camera_matrix', None)
        D = self.conf.get('dist_coeffs', None) or [0, 0, 0, 0, 0]
        # Use first 5 values
        self.k1 = float(D[0])
        self.k2 = float(D[1]) if len(D) > 1 else 0.0
        self.p1 = float(D[2]) if len(D) > 2 else 0.0
        self.p2 = float(D[3]) if len(D) > 3 else 0.0
        self.k3 = float(D[4]) if len(D) > 4 else 0.0

        self.fisheye = bool(self.conf.get('fisheye', False))

        # UI
        self.win_orig = 'calib_original'        # undistorted preview + overlays
        self.win_raw = 'calib_raw'              # raw original preview + overlays (new)
        self.win_bev = 'calib_bev'
        cv2.namedWindow(self.win_orig, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_raw, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_bev, cv2.WINDOW_NORMAL)

        # create trackbars (will be updated once image size is known)
        self._create_trackbars()

        # subscribe
        rospy.Subscriber(self._camera_topic, CompressedImage, self._image_cb, queue_size=1)

    def _create_trackbars(self):
        # src point trackbars: x,y for 4 points
        # to avoid creating trackbars with invalid ranges, use a default max that we update later
        max_w = max(1280, self.img_w)
        max_h = max(720, self.img_h)
        for i in range(4):
            cv2.createTrackbar(f'p{i}x', self.win_orig, int(self.src[i, 0]), max_w, lambda v, idx=i: self._on_trackbar_point(idx, 0, v))
            cv2.createTrackbar(f'p{i}y', self.win_orig, int(self.src[i, 1]), max_h, lambda v, idx=i: self._on_trackbar_point(idx, 1, v))

        # dst size
        cv2.createTrackbar('dst_w', self.win_orig, self.dst_w, 2000, lambda v: self._on_dst_size('w', v))
        cv2.createTrackbar('dst_h', self.win_orig, self.dst_h, 2000, lambda v: self._on_dst_size('h', v))

        # distortion sliders: support negative values.
        # We map trackbar 0..2000 -> -1.0..+1.0 (value_center = 1000). That gives a wide tuning range.
        def init_slider(name, val):
            intv = int(round(val * 1000))
            # position in 0..2000
            pos = intv + 1000
            cv2.createTrackbar(name, self.win_orig, pos, 2000, lambda v, n=name: self._on_dist_trackbar(n, v))

        init_slider('k1', self.k1)
        init_slider('k2', self.k2)
        init_slider('p1', self.p1)
        init_slider('p2', self.p2)
        init_slider('k3', self.k3)
        # fisheye toggle: 0 or 1
        cv2.createTrackbar('fisheye', self.win_orig, int(self.fisheye), 1, lambda v: self._on_fisheye(v))

        # mask toggle: when on, raw window will display only polygon (rest black)
        self.mask_enabled = False
        cv2.createTrackbar('mask', self.win_orig, 0, 1, lambda v: self._on_mask(v))

    def _on_trackbar_point(self, idx, coord, value):
        # coord 0->x, 1->y
        self.src[idx, coord] = int(value)

    def _on_dst_size(self, which, value):
        if which == 'w' and value > 0:
            self.dst_w = int(value)
        if which == 'h' and value > 0:
            self.dst_h = int(value)

    def _on_dist_trackbar(self, name, value):
        # value in 0..2000 -> map to -1000..1000 by v-1000 then /1000 => -1.0..1.0
        v = (int(value) - 1000) / 1000.0
        if name == 'k1':
            self.k1 = v
        elif name == 'k2':
            self.k2 = v
        elif name == 'p1':
            self.p1 = v
        elif name == 'p2':
            self.p2 = v
        elif name == 'k3':
            self.k3 = v

    def _on_fisheye(self, val):
        self.fisheye = bool(val)

    def _on_mask(self, val):
        self.mask_enabled = bool(val)

    def _image_cb(self, msg):
        try:
            img = self._bridge.compressed_imgmsg_to_cv2(msg)
            # make sure image is BGR
            if img is not None:
                self.image = img
                self.img_h, self.img_w = img.shape[:2]
                # update trackbar ranges if necessary
                # (trackbars can't change max after creation, so we ignore for now)
        except Exception:
            pass

    def _get_undistorted(self, img):
        # Build camera matrix from image center and focal length estimate (if camera_matrix not provided)
        # We'll make a simple K with fx=fy=0.8*width as a heuristic if config doesn't include camera_matrix
        camK = self.conf.get('camera_matrix', None)
        if camK is not None:
            K = np.array(camK, dtype=np.float64)
            if K.size == 9:
                K = K.reshape((3, 3))
        else:
            f = 0.8 * max(self.img_w, self.img_h)
            K = np.array([[f, 0.0, self.img_w / 2.0], [0.0, f, self.img_h / 2.0], [0.0, 0.0, 1.0]])

        D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

        if self.fisheye:
            try:
                # fisheye expects 4 coeffs, but we'll use first 4
                Df = D[:4]
                newK = K.copy()
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, Df, np.eye(3), newK, (self.img_w, self.img_h), cv2.CV_16SC2)
                und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                return und
            except Exception:
                return img
        else:
            try:
                newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (self.img_w, self.img_h), 1)
                map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (self.img_w, self.img_h), cv2.CV_16SC2)
                und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                return und
            except Exception:
                return img

    def save_yaml(self):
        out = {
            'src_points_px': self.src.tolist(),
            'dst_size_px': [int(self.dst_w), int(self.dst_h)],
            'camera_matrix': self.conf.get('camera_matrix', None),
            'dist_coeffs': [self.k1, self.k2, self.p1, self.p2, self.k3],
            'fisheye': bool(self.fisheye)
        }
        # If camera_matrix is None, omit it (user might want just distortion values)
        if out['camera_matrix'] is None:
            out.pop('camera_matrix')
        with open(self.yaml_path, 'w') as f:
            yaml.safe_dump(out, f)
        print(f"Saved configuration to {self.yaml_path}")

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            img = self.image.copy()

            # undistort preview
            und = self._get_undistorted(img)

            # compute homography from src (current trackbars) to dst
            src_pts = np.float32(self.src)
            dst_pts = np.float32([[0, self.dst_h], [self.dst_w, self.dst_h], [0, 0], [self.dst_w, 0]])
            try:
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                bev = cv2.warpPerspective(und, M, (self.dst_w, self.dst_h), flags=cv2.INTER_LINEAR)
            except Exception:
                bev = np.zeros((max(1, self.dst_h), max(1, self.dst_w), 3), dtype=np.uint8)

            # draw source quad on undistorted original for visual feedback
            vis = und.copy()
            try:
                pts = self.src.astype(np.int32)
                cv2.polylines(vis, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                for p in pts:
                    cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
            except Exception:
                pass

            # raw original with overlay (shows the actual image where the sliders point to)
            raw_vis = img.copy()
            try:
                rpts = self.src.astype(np.int32)
                cv2.polylines(raw_vis, [rpts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                for p in rpts:
                    cv2.circle(raw_vis, tuple(p), 4, (0, 0, 255), -1)
            except Exception:
                pass

            # if mask is enabled, mask outside of source polygon (raw image)
            if self.mask_enabled:
                try:
                    mask = np.zeros_like(raw_vis[:, :, 0], dtype=np.uint8)
                    cv2.fillPoly(mask, [rpts.reshape((-1, 1, 2))], 255)
                    raw_masked = cv2.bitwise_and(raw_vis, raw_vis, mask=mask)
                except Exception:
                    raw_masked = raw_vis
            else:
                raw_masked = raw_vis

            # info text
            info = f'dst={self.dst_w}x{self.dst_h} fisheye={int(self.fisheye)} k1={self.k1:.4f} k2={self.k2:.4f} p1={self.p1:.4f} p2={self.p2:.4f} k3={self.k3:.4f}'
            cv2.putText(vis, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imshow(self.win_orig, vis)
            cv2.imshow(self.win_raw, raw_masked)
            cv2.imshow(self.win_bev, bev)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                self.save_yaml()

            rate.sleep()


if __name__ == '__main__':
    try:
        cal = BirdsEyeCalibrator()
        cal.run()
    except Exception as e:
        print('Error starting calibrator:', e)
    finally:
        cv2.destroyAllWindows()
