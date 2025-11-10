#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import yaml
import time

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ultralytics import YOLO


class DisplayDetectedDuckieNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DisplayDetectedDuckieNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        
        # Subscriber for detected duckie images
        # The camera topic is constructed using the vehicle name from the environment variable
        # self._yolo_topic = f"/{self._vehicle_name}/detect/duckie/image"
        # self.sub_image = rospy.Subscriber(self._yolo_topic, Image, self.cbShowImage, queue_size=1)
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbShowImage, queue_size=1)

        with open('packages/followlane/config/detect_duckie.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0
        
        self.last_save_time = time.time()
        self.save_path = "/home/QuackSquad/dataset_botSlot"

        self.myTime = 3
        

    def cbShowImage(
            self,
            image_msg
        ):
        """
        Callback function to process the incoming image message.
        :param image_msg: The incoming image message.
        """
        if self.conf['show_image'] == False:
            return
        else:
            #self.latest_img = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.latest_img = self._bridge.compressed_imgmsg_to_cv2(image_msg)
            cv2.imshow("duckie-detection", self.latest_img)
            cv2.waitKey(1)
            
            now = time.time()
            if now - self.last_save_time > self.myTime:
                filename = f"{self.save_path}/bild_{int(now)}.jpg"
                cv2.imwrite(filename, self.latest_img)
                print(f"Bild gespeichert: {filename}")
                self.last_save_time = now

if __name__ == '__main__':
    node = DisplayDetectedDuckieNode(node_name='display_detectedDuckie_node')
    rospy.spin()
