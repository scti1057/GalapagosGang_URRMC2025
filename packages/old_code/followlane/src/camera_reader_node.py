#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
import time
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64, Bool, Float64MultiArray


class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "camera-reader"
        self._config_path = 'packages/followlane/config/detect_lane.yaml'

        # Load configuration from YAML file
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.debug = self.conf.get('show_debug', False)
        self.target_x_buffer = []  # buffer to smooth target x values
        self.image = None

        # === Publishers ===
        self.pub_lane = rospy.Publisher(f"/{self._vehicle_name}/detect/lane", Float64, queue_size=1)
        self.pub_left_x = rospy.Publisher(f"/{self._vehicle_name}/detect/lane/left_x", Float64, queue_size=1)
        self.pub_right_x = rospy.Publisher(f"/{self._vehicle_name}/detect/lane/right_x", Float64, queue_size=1)
        self.pub_redline = rospy.Publisher(f"/{self._vehicle_name}/stop_line_detected", Bool, queue_size=1)

        # === Subscribers ===
        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/object/duckieNearestBB", Float64MultiArray, self.cb_duckie_lane, queue_size=1)
        rospy.Subscriber(f"/{self._vehicle_name}/detect/object/duckieNearestRightBB", Float64MultiArray, self.cb_duckie_right, queue_size=1)

        # Store duckie positions to avoid false lane detections
        self.duckie_lane_x = None
        self.duckie_rightlane_x = None
        self.duckie_tolerance = 15

    # Callback for duckie bounding box in general lane
    def cb_duckie_lane(self, msg):
        if msg.data and len(msg.data) == 4:
            x1, _, x2, _ = msg.data
            self.duckie_lane_x = int((x1 + x2) / 2)

    # Callback for duckie bounding box on right lane
    def cb_duckie_right(self, msg):
        if msg.data and len(msg.data) == 4:
            x1, _, x2, _ = msg.data
            self.duckie_rightlane_x = int((x1 + x2) / 2)

    # Converts ROS image message to OpenCV format
    def image_callback(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self._timestamp = msg.header.stamp

    # Create ROI polygon from config
    def create_polygon(self):
        return np.array([[
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
        ]], dtype=np.int32)

    # Calculate target x from detected contours inside the polygon mask
    def compute_target_x_from_polygon(self, polygon, mask_white, mask_yellow, image):
        min_area = 100
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)
        mw = cv2.bitwise_and(mask_white, mask_poly)
        my = cv2.bitwise_and(mask_yellow, mask_poly)

        edges_white = cv2.Canny(cv2.GaussianBlur(mw, (5, 5), 0), 50, 150)
        edges_yellow = cv2.Canny(cv2.GaussianBlur(my, (5, 5), 0), 50, 150)

        if self.debug:
            cv2.imshow("edges-white", edges_white)
            cv2.imshow("edges-yellow", edges_yellow)

        contours_white, _ = cv2.findContours(edges_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(edges_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leftmost_x = None
        for i, cnt in enumerate(contours_white):
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            if leftmost_x is None or cx < leftmost_x:
                leftmost_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)  # draw white lane contour

        rightmost_x = None
        for i, cnt in enumerate(contours_yellow):
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])

            # Ignore contour if it's too close to a duckie
            ignore = False
            for duckie_x in [self.duckie_lane_x, self.duckie_rightlane_x]:
                if duckie_x is not None and abs(cx - duckie_x) <= self.duckie_tolerance:
                    ignore = True
                    break
            if ignore:
                continue

            if rightmost_x is None or cx > rightmost_x:
                rightmost_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)  # draw yellow lane contour

        # Decision logic based on contour positions
        if leftmost_x is not None and rightmost_x is not None and leftmost_x > rightmost_x:
            self.pub_right_x.publish(Float64(rightmost_x))
            self.pub_left_x.publish(Float64(leftmost_x))
            return ((leftmost_x + rightmost_x) / 2 - 30)
        elif rightmost_x is not None:
            self.pub_right_x.publish(Float64(rightmost_x))
            return rightmost_x + 170
        elif leftmost_x is not None:
            self.pub_left_x.publish(Float64(leftmost_x))
            return leftmost_x - 230
        else:
            return None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            start = time.time()
            image = self.image.copy()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            wh = self.conf['white']
            gh = self.conf['gelb']
            rd = self.conf['red']

            # Define red mask ranges (split around HSV hue wrap-around)
            lower_red1 = np.array([rd['hl'], rd['sl'], rd['vl']])
            upper_red1 = np.array([rd['hh'], rd['sh'], rd['vh']])
            lower_red2 = np.array([140, rd['sl'], rd['vl']])
            upper_red2 = np.array([255, rd['sh'], rd['vh']])

            # Apply red, white, and yellow masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask1, mask2)

            mask_white = cv2.inRange(hsv, (wh['hl'], wh['sl'], wh['vl']), (wh['hh'], wh['sh'], wh['vh']))
            mask_yellow = cv2.inRange(hsv, (gh['hl'], gh['sl'], gh['vl']), (gh['hh'], gh['sh'], gh['vh']))

            # Clean up noise in white and yellow masks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

            # Define bottom ROI for red line detection
            y_cutoff = 400
            threshold_pixel_count = 200
            mask_shape = mask_red.shape
            lower_part_mask = np.zeros_like(mask_red)
            lower_part_mask[y_cutoff:, :] = 255
            mask_red_low = cv2.bitwise_and(mask_red, lower_part_mask)
            red_pixels_low = cv2.countNonZero(mask_red_low)

            # Publish stop signal if enough red pixels below cutoff
            if red_pixels_low > threshold_pixel_count:
                self.pub_redline.publish(Bool(True))
                self.pub_lane.publish(Float64(0))  # Optional: stop vehicle
                if self.debug:
                    rospy.loginfo(f"[RedLine] STOP – {red_pixels_low} red pixels below y={y_cutoff}")
            else:
                self.pub_redline.publish(Bool(False))

            # Lane detection
            polygon = self.create_polygon()
            target_x = self.compute_target_x_from_polygon(polygon, mask_white, mask_yellow, image)

            if target_x is not None:
                self.target_x_buffer.append(target_x)
                if len(self.target_x_buffer) > 2:
                    self.target_x_buffer.pop(0)

                smoothed_x = int(np.mean(self.target_x_buffer))
                target_y = image.shape[0] - 50

                cv2.circle(image, (smoothed_x, target_y), 6, (255, 0, 255), -1)
                cv2.putText(image, "Target", (smoothed_x - 20, target_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                self.pub_lane.publish(Float64(smoothed_x))

            # Draw center mark for reference
            center_x = int(image.shape[1] / 2)
            center_y = image.shape[0] - 50
            cv2.circle(image, (center_x, center_y), 6, (0, 0, 255), -1)
            cv2.putText(image, "Center", (center_x - 25, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw lane polygon
            cv2.polylines(image, polygon, isClosed=True, color=(255, 255, 255), thickness=2)
            cv2.imshow(self._window, image)
            cv2.waitKey(1)

            if self.debug:
                rospy.loginfo(f"[Latency] Now: {rospy.Time.now().to_sec():.3f}, Image timestamp: {self._timestamp.to_sec():.3f}")
                rospy.loginfo(f"[Delay] {(rospy.Time.now() - self._timestamp).to_sec():.3f} seconds")
                rospy.loginfo(f"[Timer] Total time: {(time.time() - start):.3f}s")

            rate.sleep()

    # Save config on shutdown
    def fnShutDown(self):
        with open(self._config_path, 'w') as f:
            yaml.dump(self.conf, f)
        print("Config saved")

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    node.run()
