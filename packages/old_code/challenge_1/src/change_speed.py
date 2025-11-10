import os
import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Float32


class ChangeSpeedNode(DTROS): # Class name adjusted, inherits conditionally
    def __init__(self, node_name):
        self.debug_prints = True
        self.strt_msg = True
        self.node_freq = 10

        if self.debug_prints:
            rospy.loginfo(f"[CHANGE_SPEED]: Initializing node.")

        super(ChangeSpeedNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # === Abonnent (Subscriber) ===
        # Abonniert das Topic mit der gedrueckten Taste
        pressed_key_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_speed"
        self.last_pressedKeySpeed = ""
        rospy.Subscriber(pressed_key_topic, String, self.cbKeyPressedSpeed, queue_size=1)

        # === Publisher (Veroeffentlicher) ===
        # Veroeffentlicht das Topic mit der gewuenschten Geschwindigkeit
        speed_topic = f"/{self._vehicle_name}/challenge_1/speed"
        self.pub_speed = rospy.Publisher(speed_topic, Float32, queue_size=1)

        # === Fahrparameter ===
        # Ein Dictionary, um Tastendruecke auf Geschwindigkeitswerte abzubilden
        self.speed_levels = {
            "0": 0.0,
            "1": 0.2,
            "2": 0.4,
            "3": 0.6,
            "4": 0.8,
            "5": 1.0,
        }
        self.pressedKey = "[2]"  # Aktuell gedrueckte Taste
        self.v = self.speed_levels.get(self.pressedKey, 0.4) # Aktuelle Geschwindigkeit

        # === Shutdown-Registrierung ===
        rospy.on_shutdown(self.fnShutDown)


        
    ##### ===== CALLBACK-FUNKTIONEN DER ABONNENTEN ===== #####
    def cbKeyPressedSpeed(self, msg: String):
        # Entferne eckige Klammern, falls vorhanden
        self.pressedKey = msg.data.strip("[]")
        if self.debug_prints:
            rospy.loginfo(f"[CHANGE_SPEED]: Key pressed: {self.pressedKey}")



    ##### ===== ANDERE FUNKTIONEN ===== #####
    def fnShutDown(self):
        '''
        Called on shutdown to ensure vehicle stops
        '''
        if self.debug_prints:
            rospy.loginfo("[CHANGE_SPEED] Node shutting down.")



    ##### ========== HAUPTLAUF-FUNKTION ========== #####
    def run(self):
        '''
        Main run function. Runs the parking state machine in a loop.
        '''
        rate = rospy.Rate(self.node_freq)

        if self.debug_prints and self.strt_msg:
            rospy.loginfo("[CHANGE_SPEED]: Speed control active.")
            self.strt_msg = False

        last_published_v = -1.0 # Initialisiere mit einem Wert, den v nicht haben wird

        while not rospy.is_shutdown():
            # --- Verarbeitung der gedrueckten Taste ---
            # Schlage die Geschwindigkeit im Dictionary nach. Wenn die Taste
            # nicht gefunden wird, bleibt die bisherige Geschwindigkeit erhalten.
            #####################################################################
            # TODO Aufgabe 5:                                                   #
            # Geschwindigkeit der gedrueckten Taste zuordnen                    #
            # Wert veroeffentlichen (publish)                                   #
            #                                                                   #
            # Tipp: Oben ist bereits ein Dictionary definiert                   #
            # Tipp: Oben ist bereits ein Publisher angelegt                     #
            #####################################################################
            ############# >>>>>>>>>> HIER CODE EINFUEGEN >>>>>>>>>> #############


            ############# <<<<<<<<<< HIER CODE EINFUEGEN <<<<<<<<<< #############
            #####################################################################

            if self.debug_prints and self.v != last_published_v:
                rospy.loginfo(f"[CHANGE_SPEED]: Publishing speed: {self.v:.2f}")
                last_published_v = self.v
            
            rate.sleep()

if __name__ == '__main__':
    node = ChangeSpeedNode(node_name='change_speed_node')
    node.run()