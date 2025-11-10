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

        # === Subscriber ===
        # Subscribes the topic with the pressed up/down key
        pressed_key_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_v"
        self.last_pressedKeyV = ""
        rospy.Subscriber(pressed_key_topic, String, self.cbKeyPressedV, queue_size=1)
        # Subscribes the topic with the pressed left/right key
        pressed_key_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_omega"
        self.last_pressedKeyOmega = ""
        rospy.Subscriber(pressed_key_topic, String, self.cbKeyPressedOmega, queue_size=1)
        # Subscribes  the topic with the desired speed
        speed_topic = f"/{self._vehicle_name}/challenge_1/speed"
        rospy.Subscriber(speed_topic, Float32, self.cbSpeed, queue_size=1)

        # === Publisher ===
        # Sends control commands to SwitchControlNode
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        # === Driving parameters ===
        self.pressedKey_v = "" # Pressed key
        self.pressedKey_omega = ""
        self.last_speed = -1
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity
        self.speed = 0.4 # Max linear speed
        self.turn_speed = 6.0  # Max angular speed
        self.damping_factor = 0  # How quickly it slows down (higher = slower braking)
        self.acceleration_factor = 0.1  # How quickly it speeds up (lower = smoother acceleration)

        # === Register Shutdown-ToDos ===
        rospy.on_shutdown(self.fnShutDown)


        
    ##### ===== CALLBACK FUNCTIONS OF SUBSCRIBERS ===== #####
    def cbKeyPressedV(self, msg: String):
        self.pressedKey_v = msg.data
        if (self.pressedKey_v != self.last_pressedKeyV) and self.debug_prints:
            rospy.loginfo(f"[CRTL_ARROW_KEYS]: Key pressed v: {self.pressedKey_v}")

    def cbKeyPressedOmega(self, msg: String):
        self.pressedKey_omega = msg.data
        if (self.pressedKey_omega != self.last_pressedKeyOmega) and self.debug_prints:
            rospy.loginfo(f"[CRTL_ARROW_KEYS]: Key pressed v: {self.pressedKey_omega}")
    
    def cbSpeed(self, msg: Float32):
        self.speed = msg.data
        if (self.speed != self.last_speed) and self.debug_prints:
            rospy.loginfo(f"[CRTL_ARROW_KEYS]: Speed set to: {self.speed}")
            self.last_speed = self.speed



    ##### ===== OTHER FUNCTIONS ===== #####
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



    ##### ========== MAIN RUN FUNCTION ========== #####
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
            # --- Pressed key processing ---
            # Handle linear velocity (v)
            if self.pressedKey_v in ["space", "up", "down"]:
                if self.pressedKey_v == "space":
                    target_v = 0.0
                    self.v = 0.0
                elif self.pressedKey_v == "up":
                    target_v = self.speed
                elif self.pressedKey_v == "down":
                    target_v = -self.speed
            else:
                # No movement key for linear velocity is pressed
                target_v = 0.0

            # Handle angular velocity (omega)
            if self.pressedKey_omega in ["space", "left", "right"]:
                if self.pressedKey_omega == "space":
                    target_omega = 0.0
                    self.omega = 0.0
                elif self.pressedKey_omega == "left":
                    target_omega = self.turn_speed
                elif self.pressedKey_omega == "right":
                    target_omega = -self.turn_speed
            else:
                # No movement key for angular velocity is pressed
                target_omega = 0.0

            # --- Smoothly interpolate to the target speed and omega ---
            # If the target is to move, accelerate towards it
            if target_v != 0:
                self.v += (target_v - self.v) * self.acceleration_factor
            else: # If the target is to stop, apply damping
                self.v *= self.damping_factor

            # Set to zero if very close to zero to prevent drifting
            if abs(self.v) < 1e-4:
                self.v = 0.0
            if abs(self.omega) < 1e-4:
                self.omega = 0.0

            if target_omega != 0 and self.v == 0:
                self.omega = target_omega
                self.v = 0.0
            else:
                self.omega = target_omega
                
            # --- Create and publish the control command ---
            # Create a Twist2DStamped message
            twist_msg = Twist2DStamped()
            twist_msg.v = self.v
            twist_msg.omega = self.omega
            
            # Publish the message
            self.pub_lane_twist.publish(twist_msg)

            if self.debug_prints and (self.v != 0 or self.omega != 0):
                rospy.loginfo(f"[CRTL_ARROW_KEYS]: Publishing command: v={self.v:.2f}, omega={self.omega:.2f}")
            
            rate.sleep()

if __name__ == '__main__':
    node = RemoteControlNode(node_name='crtl_arrow_keys_node')
    node.run()