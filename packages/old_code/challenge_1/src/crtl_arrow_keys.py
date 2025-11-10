import os
import rospy

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String, Float32


class RemoteControlNode(DTROS): # Class name adjusted, inherits conditionally
    def __init__(self, node_name):
        self.debug_prints = True
        self.strt_msg = True
        self.node_freq = 30

        if self.debug_prints:
            rospy.loginfo(f"[CRTL_ARROW_KEYS]: Initializing node.")

        super(RemoteControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # === Abonnent (Subscriber) ===
        # Abonniert das Topic mit der gedrueckten up/down-Taste
        pressed_key_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_v"
        self.last_pressedKeyV = ""
        rospy.Subscriber(pressed_key_topic, String, self.cbKeyPressedV, queue_size=1)
        # Abonniert das Topic mit der gedrueckten left/right-Taste
        pressed_key_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_omega"
        self.last_pressedKeyOmega = ""
        rospy.Subscriber(pressed_key_topic, String, self.cbKeyPressedOmega, queue_size=1)
        # Abonniert das Topic mit der gewuenschten Geschwindigkeit
        speed_topic = f"/{self._vehicle_name}/challenge_1/speed"
        rospy.Subscriber(speed_topic, Float32, self.cbSpeed, queue_size=1)

        # === Publisher (Veroeffentlicher) ===
        # Sendet Steuerbefehle an SwitchControlNode
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        # === Fahrparameter ===
        self.pressedKey_v = ""  # Gedrueckte Taste
        self.pressedKey_omega = ""
        self.last_speed = -1
        self.v = 0.0  # Lineare Geschwindigkeit
        self.omega = 0.0  # Winkelgeschwindigkeit
        self.speed = 0.4  # Maximale lineare Geschwindigkeit
        self.turn_speed = 6.0  # Maximale Winkelgeschwindigkeit
        self.damping_factor = 0  # Wie schnell es abbremst (hoeher = langsameres Bremsen)
        self.acceleration_factor = 0.1  # Wie schnell es beschleunigt (kleiner = sanftere Beschleunigung)

        # === Shutdown-Registrierung ===
        rospy.on_shutdown(self.fnShutDown)


        
    ##### ===== CALLBACK-FUNKTIONEN DER ABONNENTEN ===== #####
    def cbKeyPressedV(self, msg: String):
        self.pressedKey_v = msg.data
        if (self.pressedKey_v != self.last_pressedKeyV) and self.debug_prints:
            #####################################################################
            # TODO Aufgabe 3.1:                                                 #
            # Ausgabe, welche Taste gedrueckt wurde                             #
            #                                                                   #
            # Tipp: Hier geht es um vorwaerts / rueckwaerts fahren              #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################
            None

    def cbKeyPressedOmega(self, msg: String):
        self.pressedKey_omega = msg.data
        if (self.pressedKey_omega != self.last_pressedKeyOmega) and self.debug_prints:
            #####################################################################
            # TODO Aufgabe 3.2:                                                 #
            # Ausgabe, welche Taste gedrueckt wurde                             #
            #                                                                   #
            # Tipp: Hier geht es um links / rechts Kurven fahren                #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################
            None
    
    def cbSpeed(self, msg: Float32):
        self.speed = msg.data
        if (self.speed != self.last_speed) and self.debug_prints:
            #####################################################################
            # TODO Aufgabe 3.3:                                                 #
            # Ausgabe, welche Taste gedrueckt wurde                             #
            #                                                                   #
            # Tipp: Hier geht es um die eingestellte Geschwindigkeit            #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################
            None



    ##### ===== ANDERE FUNKTIONEN ===== #####
    def fnShutDown(self):
        '''
        Called on shutdown to ensure vehicle stops
        '''
        if self.debug_prints:
            rospy.loginfo("[CRTL_ARROW_KEYS] Stopping vehicle.")
            rospy.loginfo("[CRTL_ARROW_KEYS] Node shutting down.")
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        for _ in range(5):
            self.pub_lane_twist.publish(stop_msg)
            rospy.sleep(0.1)



    ##### ========== HAUPTLAUF-FUNKTION ========== #####
    def run(self):
        '''
        Main run function. Runs the parking state machine in a loop.
        '''
        rate = rospy.Rate(self.node_freq)

        if self.debug_prints and self.strt_msg:
            rospy.loginfo("[CRTL_ARROW_KEYS]: Key-Cruising active.")
            self.strt_msg = False

        target_v = 0.0
        target_omega = 0.0

        while not rospy.is_shutdown():
            # --- Verarbeitung gedrueckter Tasten ---
            # Verarbeitung lineare Geschwindigkeit (v)
            #####################################################################
            # TODO Aufgabe 1:                                                   #
            # vorwaerts fahren      (Taste "up")                                #
            # rueckwaerts fahren    (Taste "down")                              #
            # bei Leertaste Stopp    (Taste "space")                            #
            #                                                                   #
            # Tipp: Was soll passieren, wenn keine Taste mehr gedrueckt wird?   #
            # Tipp: Welche Taste soll die hoeherste Prioritaet haben?           #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################

            # Verarbeitung Winkelgeschwindigkeit (omega)
            #####################################################################
            # TODO Aufgabe 2:                                                   #
            # Kurve links fahren   (Taste "left")                               #
            # Kurve rechts fahren  (Taste "right")                              #
            # bei Leertaste Stopp    (Taste "space")                            #
            #                                                                   #
            # Tipp: Was soll passieren, wenn keine Taste mehr gedrueckt wird?   #
            # Tipp: Welche Taste soll die hoeherste Prioritaet haben?           #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################

            # --- Sanftes Hochfahren/Abbremsen zu Zielwerten ---
            #####################################################################
            # TODO Aufgabe 4:                                                   #
            # sanftes Anfahren                                                  #
            # sanftes Abbremsen                                                 #
            #                                                                   #
            # Tipp: Es gibt bereits Variablen fuer den Beschleunigungsfaktor    #
            # Tipp: Es gibt bereits Variablen fuer den Abbremsfaktor            #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################

            # Setze auf 0, wenn sehr nahe bei 0, um Drift zu verhindern
            if abs(self.v) < 1e-4:
                self.v = 0.0
            if abs(self.omega) < 1e-4:
                self.omega = 0.0

            if target_omega != 0 and self.v == 0:
                self.omega = target_omega
                self.v = 0.0
            else:
                self.omega = target_omega
                
            # --- Erzeuge und publiziere den Steuerbefehl ---
            # Erzeuge eine Twist2DStamped-Nachricht
            twist_msg = Twist2DStamped()
            twist_msg.v = self.v
            twist_msg.omega = self.omega
            
            # Publiziere die Nachricht
            self.pub_lane_twist.publish(twist_msg)

            if self.debug_prints and (self.v != 0 or self.omega != 0):
                rospy.loginfo(f"[CRTL_ARROW_KEYS]: Publishing command: v={self.v:.2f}, omega={self.omega:.2f}")
            
            rate.sleep()

if __name__ == '__main__':
    node = RemoteControlNode(node_name='crtl_arrow_keys_node')
    node.run()