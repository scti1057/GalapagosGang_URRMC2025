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


class DisplayYoloResultNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DisplayYoloResultNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self.latest_yoloImg = None
        self.latest_parkingImg = None
        
        # Subscriber for detected duckie images
        # The camera topic is constructed using the vehicle name from the environment variable
        # self._yolo_topic = f"/{self._vehicle_name}/detect/duckie/image"
        # self.sub_image = rospy.Subscriber(self._yolo_topic, Image, self.cbShowImage, queue_size=1)
        self.sub_image = rospy.Subscriber(f"/{self._vehicle_name}/detect/object/image", Image, self.cbShowYoloImage, queue_size=1)
        self.sub_parkingImage = rospy.Subscriber(f"/{self._vehicle_name}/detect/parking/image", Image, self.cbShowParkingImage, queue_size=1)

        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)

        self._bridge = CvBridge()
        self.frame_count = 0
        

    def cbShowYoloImage(
            self,
            image_msg
        ):
        """
        Callback function to process the incoming image message.
        :param image_msg: The incoming image message.
        """
        self.latest_yoloImg = self._bridge.imgmsg_to_cv2(image_msg)


    def cbShowParkingImage(
            self,
            image_msg
        ):
        """
        Callback function to process the incoming image message.
        :param image_msg: The incoming image message.
        """
        self.latest_parkingImg = self._bridge.imgmsg_to_cv2(image_msg)


    def run(self):
        """
        Main loop of the node.
        """
        rospy.loginfo("Display YOLO Result Node is running.")
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.latest_yoloImg is not None and self.conf['show_yoloImage']:
                # Display the latest YOLO image
                cv2.imshow("YOLO-Result", self.latest_yoloImg)
            if self.latest_parkingImg is not None and self.conf['show_parkingImage']:
                # Display the latest parking image
                cv2.imshow("Parking-Result", self.latest_parkingImg)
            cv2.waitKey(1)
            rate.sleep()



if __name__ == '__main__':
    node = DisplayYoloResultNode(node_name='display_yoloResult_node')
    node.run()
