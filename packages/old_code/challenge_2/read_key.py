import os
import pygame
import rospy

from duckietown.dtros import DTROS, NodeType

from std_msgs.msg import String


class ReadKeyNode(DTROS):
    def __init__(self, node_name):
        self.debug_prints = False
        self.strt_msg = True
        self.node_freq = 50

        if self.debug_prints:
            rospy.loginfo(f"[READ_KEY]: Initializing node.")
            
        super(ReadKeyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # === Publisher ===
        # Publishes the topic with the pressed key for speed (numbers)
        pressed_key_speed_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_speed"
        self.pub_key_speed = rospy.Publisher(pressed_key_speed_topic, String, queue_size=1)
        # Publishes the topic with the pressed key for linear velocity (up/down)
        pressed_key_v_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_v"
        self.pub_key_v = rospy.Publisher(pressed_key_v_topic, String, queue_size=1)
        # Publishes the topic with the pressed key for angular velocity (left/right)
        pressed_key_omega_topic = f"/{self._vehicle_name}/challenge_1/pressed_key_omega"
        self.pub_key_omega = rospy.Publisher(pressed_key_omega_topic, String, queue_size=1)

        # === Pygame initialisation ===
        pygame.init() # Pygame must be initialized to read pressed keys
        self.screen = pygame.display.set_mode((100, 100)) # Small, not visible window
        pygame.display.set_caption("Keyboard reading") # Title

        # === Register Shutdown-ToDos ===
        self.running = True
        rospy.on_shutdown(self.fnShutDown)

        

    ##### ===== CALLBACK FUNCTIONS OF SUBSCRIBERS ===== #####


    ##### ===== OTHER FUNCTIONS ===== #####
    def fnShutDown(self):
        '''
        Called on node shutdown
        '''
        pygame.quit() # Pygame sauber beenden
        if self.debug_prints:
            rospy.loginfo("[READ_KEY] Node shutting down.")


    ##### ========== MAIN RUN FUNCTION ========== #####
    def run(self):
        '''
        Main run function. Runs the parking state machine in a loop.
        '''
        rate = rospy.Rate(self.node_freq)

        if self.debug_prints and self.strt_msg:
            rospy.loginfo("[READ_KEY]: Keyboard reading activ --> publishing pressed keys.")
            self.strt_msg = False

        while self.running and not rospy.is_shutdown():
            # --- Pygame event processing ---
            for event in pygame.event.get():
                # If the window is closed
                if event.type == pygame.QUIT:
                    self.running = False
                    if self.debug_prints:
                        rospy.loginfo("[READ_KEY]: Pygame window closed. Shutting down.")

                # When a key is pressed
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    if self.debug_prints:
                        rospy.loginfo(f"[READ_KEY]: Key down: {key_name}")

                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        if self.debug_prints:
                            rospy.loginfo("[READ_KEY]: ESC pressed. Shutting down.")
                    else:
                        # Map keys to the correct publishers
                        if event.key in (pygame.K_UP, pygame.K_DOWN):
                            # publish up/down on v topic
                            self.pub_key_v.publish(String(data=key_name))
                        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                            # publish left/right on omega topic
                            self.pub_key_omega.publish(String(data=key_name))
                        else:
                            # publish digit keys on speed topic (numbers)
                            # pygame.key.name(event.key) returns '1','2',... for number keys
                            if key_name.isdigit():
                                self.pub_key_speed.publish(String(data=key_name))
                            else:
                                # other keys are ignored (or extend here if needed)
                                pass

                # When a key is released
                elif event.type == pygame.KEYUP:
                    key_name = pygame.key.name(event.key)
                    if self.debug_prints:
                        rospy.loginfo(f"[READ_KEY]: Key up: {key_name}")

                    # Publish an empty string to signal that the specific key is no longer pressed
                    if event.key in (pygame.K_UP, pygame.K_DOWN):
                        self.pub_key_v.publish(String(data=""))
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self.pub_key_omega.publish(String(data=""))
                    elif key_name.isdigit():
                        self.pub_key_speed.publish(String(data=""))
                    else:
                        # other keys ignored
                        pass

            # If running is set to False, initiate a ROS shutdown
            if not self.running:
                rospy.signal_shutdown("[READ_KEY]: Requested shutdown.")

            rate.sleep()

if __name__ == '__main__':
    node = ReadKeyNode(node_name='read_key_node')
    node.run()
