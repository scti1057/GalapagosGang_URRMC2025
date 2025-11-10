#!/usr/bin/env python3

import os

import numpy as np
import rospkg
import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool, Float64, Int32, Int32MultiArray, String
from sensor_msgs.msg import CompressedImage, Range

class CameraNode(DTROS):
    def __init__(self, node_name):
        # Initialisierung der Basisklasse
        super(CameraNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Konfigurationsvariablen
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.bridge = CvBridge()

        # === Subscriber (Abonnenten) ===
        # Abonniert die Kamera-Bilder vom Duckiebot
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cb_display_image, queue_size=1)
        rospy.loginfo(f"[{self.node_name}] Abonniert: {self._camera_topic}")

        # Frame-Verarbeitung
        self.counter = 0
        self.Xth_frame = 2  # Verarbeite jedes X-te Frame für bessere Performance


        #####################################################################
        # TODO Aufgabe 2.1:                                                 #
        # Hier ToF (Time of Flight) Subscriber einfuegen                   #
        #                                                                   #
        # Tipp: Nutze den richtigen Message-Typ fuer ToF-Sensordaten      #
        #####################################################################
        ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


        ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
        #####################################################################

    ##### ===== CALLBACK FUNCTIONS OF SUBSCRIBERS ===== #####
    def cb_display_image(self, image_msg):

        #####################################################################
        # Vorprogrammierte Bildverarbeitung                                #
        # - Konvertierung des komprimierten Bildes                        #
        # - Frame-Zaehlung und Verarbeitung                               #
        #####################################################################

        #####################################################################
        # TODO Aufgabe 2.2:                                                 #
        # ToF-Sensordaten im Kamerabild visualisieren                      #
        #                                                                   #
        # Tipp: Nutze cv2.putText um die Distanz anzuzeigen               #
        #####################################################################
        ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


        ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
        #####################################################################
        return

    ##### ===== SHUTDOWN FUNCTION ===== #####
    def on_shutdown(self):
        '''
        Wird beim Beenden des Nodes aufgerufen
        '''
        # Schließe alle offenen OpenCV-Fenster
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = CameraNode(node_name="CameraNode")
    rospy.spin()
