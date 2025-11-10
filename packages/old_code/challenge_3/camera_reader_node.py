#!/usr/bin/env python3

import os
import rospy
import cv2
import yaml
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float64


class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._window = "camera-reader"
        self._config_path = 'packages/challenge_3/detect_lane.yaml'

        # Lade Konfiguration aus YAML-Datei
        with open(self._config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        self.target_x_buffer = []  # Puffer zum Glätten der Ziel-X-Werte
        self.image = None

        # === Publisher ===
        self.pub_lane_x = rospy.Publisher(f"/{self._vehicle_name}/detect/lane_x", Float64, queue_size=1)

        # === Subscriber ===
        rospy.Subscriber(self._camera_topic, CompressedImage, self.image_callback, queue_size=1)

    # Konvertiert ROS-Bildnachricht in OpenCV-Format
    def image_callback(self, msg):
        self.image = self._bridge.compressed_imgmsg_to_cv2(msg)

    # Erstelle ROI-Polygon aus der Konfiguration
    def create_polygon(self):
        return np.array([[
            [self.conf['lane_image']['top_left_x'], self.conf['lane_image']['top_left_y']],
            [self.conf['lane_image']['top_right_x'], self.conf['lane_image']['top_right_y']],
            [self.conf['lane_image']['bottom_right_x'], self.conf['lane_image']['bottom_right_y']],
            [self.conf['lane_image']['bottom_left_x'], self.conf['lane_image']['bottom_left_y']],
        ]], dtype=np.int32)

    # Berechne Ziel-X aus erkannten Konturen innerhalb der Polygon-Maske
    def compute_target_x_from_polygon(self, polygon, mask_white, mask_yellow, image):
        min_area = 100

        # ROI-Polygon anwenden
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)
        mw = cv2.bitwise_and(mask_white, mask_poly)
        my = cv2.bitwise_and(mask_yellow, mask_poly)

        # Kantenerkennung
        edges_white = cv2.Canny(cv2.GaussianBlur(mw, (5, 5), 0), 50, 150)
        edges_yellow = cv2.Canny(cv2.GaussianBlur(my, (5, 5), 0), 50, 150)

        # Konturen finden
        contours_white, _ = cv2.findContours(edges_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(edges_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Erkenne linke (weiße) und rechte (gelbe) Grenzen
        right_lane_x = None
        for cnt in contours_white:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            if right_lane_x is None or cx < right_lane_x:
                right_lane_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)  # draw white lane contour

        left_lane_x = None
        for cnt in contours_yellow:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            if left_lane_x is None or cx > left_lane_x:
                left_lane_x = cx
                cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)  # Zeichne gelbe Fahrspurkontur

        ############################################################################
        # TODO Aufgabe 2:                                                          #
        # Berechne den Ziel-X-Wert basierend auf den erkannten Linien              #
        #                                                                          #
        # Tipp: Nutze right_lane_x und left_lane_x um die Mittellinie zu berechnen #
        # Tipp: Was passiert, wenn nur eine Linie erkannt wird?                    #
        ############################################################################
        ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> ####################

        # Mittelpunkt berechnen
        print("right_lane_x:", right_lane_x, " left_lane_x:", left_lane_x)
        if right_lane_x is not None and left_lane_x is not None and right_lane_x > left_lane_x:
            return 300
        else:
            return None

        ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
        #####################################################################

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue

            image = self.image.copy()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # HSV-Bereiche aus der Konfiguration
            wh = self.conf['white']
            gh = self.conf['gelb']

            # Erstelle binäre Masken für weiß und gelb
            mask_white = cv2.inRange(hsv,
                                     (wh['hl'], wh['sl'], wh['vl']),
                                     (wh['hh'], wh['sh'], wh['vh']))
            mask_yellow = cv2.inRange(hsv,
                                      (gh['hl'], gh['sl'], gh['vl']),
                                      (gh['hh'], gh['sh'], gh['vh']))

            # Morphologische Bereinigung
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

            # Lane detection inside polygon
            polygon = self.create_polygon()
            target_x = self.compute_target_x_from_polygon(polygon, mask_white, mask_yellow, image)

            if target_x is not None:
                self.target_x_buffer.append(target_x)
                if len(self.target_x_buffer) > 2:
                    self.target_x_buffer.pop(0)

                smoothed_x = int(np.mean(self.target_x_buffer))
                target_y = image.shape[0] - 50


                #####################################################################
                # TODO Aufgabe 1:                                                   #
                # Zeichne einen Kreis und Text für den Zielpunkt "smoothed_x"       #
                #                                                                   #
                # Tipp: Nutze cv2.circle und cv2.putText                            #
                #####################################################################
                ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############

                # Zielpunkt zeichnen
                
                
                # Text für Zielpunkt
                

                ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
                #####################################################################
                

                # publish target midpoint (zum fahren smothed_x publishen also # entfernen)
                #####################################################################
                # self.pub_lane_x.publish(Float64(smoothed_x))
                #####################################################################
                
                
            # Zeichne Fahrzeugmittelpunkt der Bildmitte
            center_x = int(image.shape[1] / 2)
            center_y = image.shape[0] - 50
            cv2.circle(image, (center_x, center_y), 6, (0, 0, 255), -1)
            cv2.putText(image, "Center", (center_x - 25, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Erkennungspolygon zeichnen
            cv2.polylines(image, polygon, isClosed=True, color=(255, 255, 255), thickness=2)

            # Debug-Fenster zur Visualisierung anzeigen
            cv2.imshow(self._window, image)
            cv2.waitKey(1)

            rate.sleep()

    def fnShutDown(self):
        '''
        Wird beim Beenden des Nodes aufgerufen
        '''
        # Konfiguration in YAML-Datei zurückschreiben
        with open(self._config_path, 'w') as f:
            yaml.dump(self.conf, f)
        print("Konfiguration gespeichert")


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    node.run()
