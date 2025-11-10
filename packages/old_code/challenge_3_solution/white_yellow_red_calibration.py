#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Int32



class WhiteYellowCalibrationNode(DTROS):
    def __init__(self, node_name):
        super(WhiteYellowCalibrationNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "calibration"
        self._config_path = 'packages/challenge_3_solution/detect_lane.yaml'
        self.played = False

        # Load HSV config from YAML
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Map for color modes
        self.mode_map = {0: 'white', 1: 'gelb', 2: 'red'}
        self.current_mode = 'white'
        self.image = None

        # Publisher for sound feedback
        self.sound_pub = rospy.Publisher("/play_sound_trigger", Int32, queue_size=1)
        self.played = False

        # Subscribe to camera topic
        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback)

        # Create OpenCV window and trackbars
        cv2.namedWindow(self._window)
        self.init_trackbars()

        rospy.loginfo(f"ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI')}")
        rospy.loginfo(f"ROS_IP: {os.environ.get('ROS_IP')}")
        rospy.loginfo(f"VEHICLE_NAME: {self._vehicle_name}")
        
    def init_trackbars(self):
        # Create HSV trackbars for calibration
        def nothing(x): pass

        for name in ['hl', 'hh', 'sl', 'sh', 'vl', 'vh']:
            val = self.conf[self.current_mode][name]
            cv2.createTrackbar(name, self._window, val, 255, nothing)

        # Trackbar to switch between color modes
        cv2.createTrackbar("mode", self._window, 0, 2, self.switch_mode)

    def switch_mode(self, val):
        # Change current color mode and update trackbars
        self.current_mode = self.mode_map.get(val, 'white')
        self.update_trackbars()

    def update_trackbars(self):
        # Set trackbars to current HSV values for selected mode
        for name in ['hl', 'hh', 'sl', 'sh', 'vl', 'vh']:
            val = self.conf[self.current_mode][name]
            cv2.setTrackbarPos(name, self._window, val)

    def get_trackbar_values(self):
        # Read current HSV values from trackbars
        return {
            'hl': cv2.getTrackbarPos('hl', self._window),
            'hh': cv2.getTrackbarPos('hh', self._window),
            'sl': cv2.getTrackbarPos('sl', self._window),
            'sh': cv2.getTrackbarPos('sh', self._window),
            'vl': cv2.getTrackbarPos('vl', self._window),
            'vh': cv2.getTrackbarPos('vh', self._window)
        }

    def image_callback(self, msg):
        # Convert compressed image to OpenCV format
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            # Play sound once at startup
            if not self.played:
                rospy.loginfo(f"📡 Publishing to /play_sound_trigger with value 1")
                self.sound_pub.publish(Int32(data=1))
                self.played = True

            # Get HSV values from trackbars and apply mask
            vals = self.get_trackbar_values()
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lower = (vals['hl'], vals['sl'], vals['vl'])
            upper = (vals['hh'], vals['sh'], vals['vh'])
            mask = cv2.inRange(hsv, lower, upper)

            # Visualize mask in color depending on mode
            output = self.image.copy()
            if self.current_mode == 'gelb':
                output[mask > 0] = (0, 255, 255)
            elif self.current_mode == 'red':
                output[mask > 0] = (0, 0, 255)
            else:
                output[mask > 0] = (255, 255, 255)

            cv2.imshow(self._window, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save current HSV values to YAML
                self.conf[self.current_mode] = vals
                with open(self._config_path, 'w') as f:
                    yaml.dump(self.conf, f)
                print(f"[✓] HSV values for '{self.current_mode}' saved.")

            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = WhiteYellowCalibrationNode(node_name='white_yellow_calibration_node')
    node.run()
