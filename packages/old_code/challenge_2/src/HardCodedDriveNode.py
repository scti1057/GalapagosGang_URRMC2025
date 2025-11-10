#!/usr/bin/env python3

import os

import time
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float64, Int32, Int32MultiArray
from sensor_msgs.msg import Range


class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        # Initialisierung der Basisklasse
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Konfigurationsvariablen
        self._vehicle_name = os.environ["VEHICLE_NAME"]

        # === Subscriber (Abonnenten) ===
        # ToF-Sensor Subscriber (auskommentiert)
        # self.sub_ToF = rospy.Subscriber(f"/{self._vehicle_name}/front_center_tof_driver_node/range", Range, self.cb_ToF, queue_size=1)

        # === Publisher (Veroeffentlicher) ===
        # Veroeffentlicht Geschwindigkeitskommandos fuer den Duckiebot
        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        # Initialisiere Fahrt und Shutdown-Handler
        self.drive()
        rospy.on_shutdown(self.fnShutDown)

    #####################################################################
    # TODO Aufgabe 2.3:                                                 #
    # Implementiere eine Notfall-Stopp Funktion                        #
    #                                                                   #
    # Tipp: Nutze die ToF-Sensordaten um Hindernisse zu erkennen      #
    # Tipp: Setze v=0 und omega=0 wenn ein Objekt zu nah ist          #
    #####################################################################
    ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


    ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
    #####################################################################

    ##### ===== DRIVING FUNCTIONS ===== #####
    def drive(self, v=0, omega=0, drive_time=0):
        '''
        Steuert den Duckiebot mit gegebener Geschwindigkeit fuer eine bestimmte Zeit
        :param v: Lineare Geschwindigkeit (vorwaerts/rueckwaerts)
        :param omega: Winkelgeschwindigkeit (links/rechts)
        :param drive_time: Fahrzeit in Sekunden
        '''
        #####################################################################
        # TODO Aufgabe 2.1:                                                 #
        # Implementiere die Fahrlogik                                       #
        #                                                                   #
        # Tipp: Erstelle ein Twist2DStamped Message                        #
        # Tipp: Nutze time.sleep() fuer die Fahrzeit                       #
        #####################################################################
        ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


        ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
        #####################################################################
        return


    ##### ===== SHUTDOWN FUNCTION ===== #####
    def fnShutDown(self):
        '''
        Wird beim Beenden des Nodes aufgerufen
        Stoppt den Duckiebot durch Setzen der Geschwindigkeit auf 0
        '''
        rospy.loginfo("[CONTROL_LANE] Node wird beendet. Geschwindigkeit wird auf 0 gesetzt")

        # Sende Stop-Kommando
        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)


if __name__ == "__main__":
    # Node erstellen und starten
    node = ControlLaneNode(node_name="control_lane_node")
    # Node am Laufen halten
    rospy.spin()
