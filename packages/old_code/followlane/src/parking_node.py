import cv2
import os
import rospy
import threading
import yaml
import time
import numpy as np
import functools
import math

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32, Float64MultiArray, ColorRGBA


class ParkingNode(DTROS):
    def __init__(self, node_name):
        # Initialize ROS node and set node type
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Image and detection state
        self.camImage = None         # Latest camera image (raw)
        self.cv_image = None         # Latest YOLO-annotated image
        self.curr_bbox = None        # Current bounding box for parking slot
        self.linedImage = None       # Image with detected lines drawn
        self.parkstatusImage = None  # (Unused) Image for parking status

        # State machine variables
        self.state = "IDLE"          # Current state: IDLE, SLOW, PARKING, WAIT, EXIT
        self.last_slot = None        # (Unused) Last slot state
        self.transition_lock = threading.Lock()  # Lock for state transitions

        # Status text for overlay/debug
        self.status_text = "Status: IDLE"

        # CV bridge for image conversion
        self._bridge_yoloImage = CvBridge()
        self._bridge_camImage = CvBridge()

        # Timers for state transitions
        self.time_lastFree = time.time()
        self.time_lastOccupied = time.time()
        self.time_startPark = None
        self.time_inPark = None
        self.time_doExit = None

        # PID control variables for parking
        self.prev_lateral_error = 0.0
        self.integral_lateral_error = 0.0
        self.prev_angle_error = 0.0
        self.integral_angle_error = 0.0
        self.last_time = time.time()

        # Parking slot state
        self.slot = 'occupied'  # Start as occupied

        # LED blinking pattern setup
        self.pattern_on = LEDPattern()
        self.pattern_on.frequency = 2.0
        self.blink_on = True
        self.blink_timer = None  # Timer for blinking
        self.pattern_on.color_mask = [False, True, False, False, True]
        self.pattern_on.frequency_mask = [False, True, False, False, True]
        self.pattern_on.rgb_vals = [
            ColorRGBA(0, 0, 0, 1.0),
            ColorRGBA(1.0, 1.0, 0.0, 1.0),
            ColorRGBA(0, 0, 0, 1.0),
            ColorRGBA(0, 0, 0, 1.0),
            ColorRGBA(1.0, 1.0, 0.0, 1.0)
        ]

        # Read configuration files for parking and lane detection
        with open('packages/followlane/config/detect_duckieBotSlot.yaml', 'r') as f:
            self.conf = yaml.safe_load(f)
        with open('packages/followlane/config/detect_lane.yaml', 'r') as f:
            self.conf_lane = yaml.safe_load(f)

        # === ROS Subscribers ===
        # Bounding box for nearest free slot
        self.sub_freeSlot = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/object/parkingBB",
            Float64MultiArray, self.cbFreeSlot, queue_size=1)
        # Bounding box for nearest occupied slot
        self.sub_occupiedSlot = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/object/parkingOccupiedBB",
            Float64MultiArray, self.cbOccupiedSlot, queue_size=1)
        # Camera images (compressed)
        self.sub_camImage = rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage, self.cbCamImage, queue_size=1)
        # (Optional) YOLO-annotated image
        # self.sub_YoloImage = rospy.Subscriber(
        #     f"/{self._vehicle_name}/detect/object/image",
        #     Image, self.cbYoloImage, queue_size=1)

        # === ROS Publishers ===
        # State for switch_control_node
        self.pub_state = rospy.Publisher(
            f"/{self._vehicle_name}/detect/object/slow4park",
            Int32, queue_size=1)
        # Driving command
        self.pub_lane_twist = rospy.Publisher(
            f"/{self._vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped, queue_size=1)
        # LED pattern
        self.led_pub = rospy.Publisher(
            f"/{self._vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern, queue_size=1)
        # Parking image for visualization
        self.pub_parkingImage = rospy.Publisher(
            f"/{self._vehicle_name}/detect/parking/image",
            Image, queue_size=1)

        # History for smoothing line detection and control
        self.m_yellow_hist = []
        self.b_yellow_hist = []
        self.smooth_N = 10  # Number of frames for smoothing

        self.lateral_error_hist = []
        self.angle_error_hist = []
        self.smooth_error_N = 10  # Number of frames for error smoothing

    ##### ===== CALLBACK FUNCTIONS OF SUBSCRIBERS ===== #####
    def cbYoloImage(self, msg):
        '''
        Callback for YOLO-annotated image (if enabled).
        Overlays status text and republishes for visualization.
        '''
        try:
            self.cv_image = self._bridge_yoloImage.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        # Draw status text with timing info
        myText = self.status_text
        if self.status_text == "Status: SLOW --> approaching slot":
            if self.time_startPark is not None:
                self.delay_startPark = time.time()-self.time_startPark
                myText = f"{self.status_text}, {self.delay_startPark:.2f}s"
        elif self.status_text == "Status: PARKING --> is parking (reverse)":
            if self.time_doPark is not None:
                self.delay_doPark = time.time()-self.time_doPark
                myText = f"{self.status_text}, {self.delay_doPark:.2f}s"
        elif self.status_text == "Status: WAIT --> parked":
            if self.time_inPark is not None:
                self.delay_inPark = time.time()-self.time_inPark
                myText = f"{self.status_text}, {self.delay_inPark:.2f}s"
        elif self.status_text == "Status: EXIT":
            if self.time_doExit is not None:
                self.delay_doExit = time.time()-self.time_doExit
                myText = f"{self.status_text}, {self.delay_doExit:.2f}s"

        cv2.putText(self.cv_image, myText, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        img_msg = self._bridge_yoloImage.cv2_to_imgmsg(self.cv_image, encoding="bgr8")
        self.pub_parkingImage.publish(img_msg)

    def cbFreeSlot(self, msg):
        '''
        Callback for bounding box of free parking slot.
        Updates current bounding box, slot state, and timestamp.
        '''
        if not msg.data or len(msg.data) != 4:
            self.curr_bbox = None
            return
        else:
            self.curr_bbox = msg
            self.slot = 'free'
            self.time_lastFree = time.time()
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] free Slot detected")
            
    def cbOccupiedSlot(self, msg):
        '''
        Callback for bounding box of occupied parking slot.
        Updates slot state and timestamp.
        '''
        if not msg.data or len(msg.data) != 4:
            self.curr_bbox = None
            return
        else:
            self.curr_bbox = None
            self.slot = 'occupied'
            self.time_lastOccupied = time.time()
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] occupied Slot detected")

    def cbCamImage(self, msg):
        '''
        Callback for compressed camera image.
        Converts to OpenCV format for processing.
        '''
        self.camImage = self._bridge_camImage.compressed_imgmsg_to_cv2(msg)

    ##### ===== OTHER FUNCTIONS ===== #####
    def create_polygon(self):
        # Creates a polygon mask for the parking area based on config
        return np.array([[
            [self.conf_lane['parking_image']['top_left_x'], self.conf_lane['parking_image']['top_left_y']],
            [self.conf_lane['parking_image']['top_right_x'], self.conf_lane['parking_image']['top_right_y']],
            [self.conf_lane['parking_image']['bottom_right_x'], self.conf_lane['parking_image']['bottom_right_y']],
            [self.conf_lane['parking_image']['bottom_left_x'], self.conf_lane['parking_image']['bottom_left_y']],
        ]], dtype=np.int32)
    
    def detect_yellow(self, image):
        '''
        Detects yellow and white lines in the image using color masks and contours.
        Fits lines through detected centroids and draws them.
        Returns the slope and intercept of the yellow line.
        '''
        min_area = 1
        m_yellow = b_yellow = None

        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get config values for white and yellow
        wh = self.conf_lane['white']
        gh = self.conf_lane['gelb']

        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Create mask for white detection
        mask_white = cv2.inRange(hsv, (wh['hl'], wh['sl'], wh['vl']), (wh['hh'], wh['sh'], wh['vh']))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

        # Create mask for yellow detection
        mask_yellow = cv2.inRange(hsv, (gh['hl'], gh['sl'], gh['vl']), (gh['hh'], gh['sh'], gh['vh']))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # Mask with polygon (region of interest)
        polygon = self.create_polygon()
        mask_poly = np.zeros_like(mask_white)
        cv2.fillPoly(mask_poly, polygon, 255)
        mw = cv2.bitwise_and(mask_white, mask_poly)
        my = cv2.bitwise_and(mask_yellow, mask_poly)

        # Edge detection
        edges_white = cv2.Canny(cv2.GaussianBlur(mw, (5, 5), 0), 50, 150)
        edges_yellow = cv2.Canny(cv2.GaussianBlur(my, (5, 5), 0), 50, 150)

        # Find contours for white and yellow
        contours_white, _ = cv2.findContours(edges_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(edges_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids_white = []
        centroids_yellow = []

        # Find centroids of white contours
        for cnt in contours_white:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids_white.append((cx, cy))
                cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        # Find centroids of yellow contours
        for cnt in contours_yellow:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids_yellow.append((cx, cy))
                cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)

        # Fit line through white centroids (if enough points)
        if len(centroids_white) >= 2:
            xs = np.array([pt[0] for pt in centroids_white])
            ys = np.array([pt[1] for pt in centroids_white])
            m_white, b_white = np.polyfit(xs, ys, 1)
            # Optionally draw the line

        # Fit line through yellow centroids (if enough points)
        if len(centroids_yellow) >= 2:
            xs = np.array([pt[0] for pt in centroids_yellow])
            ys = np.array([pt[1] for pt in centroids_yellow])
            m_yellow, b_yellow = np.polyfit(xs, ys, 1)
            # Draw yellow line on image
            x1, x2 = 0, image.shape[1]
            y1 = int(m_yellow * x1 + b_yellow)
            y2 = int(m_yellow * x2 + b_yellow)
            cv2.line(image, (x1, y1), (x2, y2), (100, 255, 255), 2)

        # Draw target line for yellow (reference for parking)
        self.m_target = 0
        self.b_target = 270
        x1, x2 = 0, image.shape[1]
        y1 = int(self.m_target * x1 + self.b_target)
        y2 = int(self.m_target * x2 + self.b_target)
        cv2.line(image, (x1, y1), (x2, y2), (180, 105, 255), 2)

        self.linedImage = image

        return m_yellow, b_yellow

    def calculate_control(self, m_yellow, b_yellow):
        '''
        Calculate speed (v) and angular velocity (omega) to control the parking maneuver.
        Uses PID control based on yellow line and target line.
        '''
        if m_yellow is None or b_yellow is None:
            if self.conf["debugPrints_parking"]:
                rospy.loginfo(f"[PARKING] m_yellow={m_yellow}; b_yellow={b_yellow}")
            return None, None  # Yellow line not detected

        # Get image width and define two x-positions: left and right
        width = self.camImage.shape[1]
        x_left = int(width * 0.1)
        x_right = int(width * 0.9)

        # Calculate corresponding y-values for the yellow and target line at those x positions
        y_yellow_left = m_yellow * x_left + b_yellow
        y_target_left = self.m_target * x_left + self.b_target

        y_yellow_right = m_yellow * x_right + b_yellow
        y_target_right = self.m_target * x_right + self.b_target

        # Compute lateral errors at left and right
        lateral_error_left = y_target_left - y_yellow_left
        lateral_error_right = y_target_right - y_yellow_right

        # Average lateral error as main input for steering correction
        if lateral_error_left > 0 and lateral_error_right > 0:
            lateral_error = 0
            v = omega = 0
            self.state = "WAIT"
            self.status_text = "Status: WAIT --> parked"
            self.time_inPark = time.time()
        else:
            lateral_error = (lateral_error_left + lateral_error_right) / 2

        # Estimate angular deviation from difference between left and right error
        angle_error = lateral_error_right - lateral_error_left

        # Smooth errors over N frames
        self.lateral_error_hist.append(lateral_error)
        self.angle_error_hist.append(angle_error)
        if len(self.lateral_error_hist) > self.smooth_error_N:
            self.lateral_error_hist.pop(0)
            self.angle_error_hist.pop(0)
        lateral_error = np.median(self.lateral_error_hist)
        angle_error = np.median(self.angle_error_hist)

        # Time step for PID
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        # PID gains for lateral and angle control
        Kp_lat = 0.007
        Ki_lat = 0.0005
        Kd_lat = 0.002
        Kp_ang = 0.04
        Ki_ang = 0.0001
        Kd_ang = 0.01

        # PID for lateral error
        self.integral_lateral_error += lateral_error * dt
        self.integral_lateral_error = max(min(self.integral_lateral_error, 100), -100)
        derivative_lateral_error = (lateral_error - self.prev_lateral_error) / dt if dt > 0 else 0
        self.prev_lateral_error = lateral_error

        omega_lat = (Kp_lat * lateral_error +
                     Ki_lat * self.integral_lateral_error +
                     Kd_lat * derivative_lateral_error)

        # PID for angle error
        self.integral_angle_error += angle_error * dt
        self.integral_angle_error = max(min(self.integral_angle_error, 100), -100)
        derivative_angle_error = (angle_error - self.prev_angle_error) / dt if dt > 0 else 0
        self.prev_angle_error = angle_error

        omega_ang = (Kp_ang * angle_error +
                     Ki_ang * self.integral_angle_error +
                     Kd_ang * derivative_angle_error)

        omega = omega_lat + omega_ang
        omega = max(min(omega, 5.0), -5.0)  # Clamp to [-5, 5]

        if abs(omega) < 0.1:
            omega = 0.0

        # Dynamically adjust speed (v) based on total error
        total_error = abs(lateral_error)
        min_v = 0.15
        max_v = 0.15
        v = -min(max_v, 0.02 * total_error)

        if abs(omega) > 0.1 and abs(v) < abs(min_v):
            v = -max(abs(v), abs(min_v))

        # Debug output
        if self.conf["debugPrints_parking"]:
            rospy.loginfo(f"[PARKING] lat_err L/R/ges: {lateral_error_left:.2f}/{lateral_error_right:.2f}/{lateral_error:.2f}, "
                        f"angle_err: {angle_error:.2f}, omega: {omega:.2f}, v: {v:.2f}")

        # Stop condition if errors are small enough
        if abs(lateral_error) < 5 and abs(angle_error) < 5:
            v = omega = 0
            self.state = "WAIT"
            self.status_text = "Status: WAIT --> parked"
            self.time_inPark = time.time()

        return v, omega
    
    def blink_all_leds(self, event):
        '''
        Toggles all LEDs on/off for blinking effect.
        '''
        if not hasattr(self, 'pattern_on'):
            return

        pattern_for_publish = LEDPattern()

        if self.blink_on:
            pattern_for_publish = self.pattern_on
        else:
            # Turn off all 5 LEDs
            pattern_for_publish.color_mask = [False] * 5
            pattern_for_publish.frequency = 0.0
            pattern_for_publish.frequency_mask = [False] * 5
            pattern_for_publish.rgb_vals = [ColorRGBA(0, 0, 0, 1.0)] * 5

        self.led_pub.publish(pattern_for_publish)
        self.blink_on = not self.blink_on

    def turn_off_leds(self):
        '''
        Turns off all LEDs except for the default pattern.
        '''
        pattern_default = LEDPattern()
        # LED 0, 1, 3, 4: use; LED 2: ignore
        pattern_default.color_mask = [True, True, False, True, True]
        pattern_default.frequency_mask = [False, False, False, False, False]
        pattern_default.frequency = 0.0
        # Mapping for each LED
        pattern_default.rgb_vals = [
            ColorRGBA(1.0, 1.0, 1.0, 1.0),  # [0] Front left → white
            ColorRGBA(1.0, 0.0, 0.0, 1.0),  # [1] Rear right → red
            ColorRGBA(0.0, 0.0, 0.0, 1.0),  # [2] Ignore
            ColorRGBA(1.0, 0.0, 0.0, 1.0),  # [3] Rear left → red
            ColorRGBA(1.0, 1.0, 1.0, 1.0)   # [4] Front right → white
        ]
        self.led_pub.publish(pattern_default)

    ##### ========== MAIN RUN FUNCTION ========== #####
    def run(self):
        '''
        Main run function. Runs the parking state machine in a loop.
        '''
        rate = rospy.Rate(10)  # Set loop rate to 10 Hz
        while not rospy.is_shutdown():
            # --- Check if parking is enabled and camera image is available ---
            if self.camImage is None or not self.conf["go_parking"]:
                if self.conf["debugPrints_parking"] and not self.conf["go_parking"]:
                    rospy.loginfo(f"[PARKING] deactivated")
                rate.sleep()
                continue

            # --- Wait for a bounding box if in IDLE state ---
            elif self.curr_bbox is None and self.state == "IDLE":
                rate.sleep()
                continue

            # --- If slot is occupied or recently occupied, wait ---
            elif self.slot == 'occupied' or time.time() - self.time_lastOccupied < 5:
                if self.conf["debugPrints_parking"]:
                    rospy.loginfo(f"[PARKING] time since last occupied {time.time()-self.time_lastOccupied:.2f}s")
                rate.sleep()
                continue

            # --- If a bounding box is available, extract coordinates ---
            elif self.curr_bbox is not None:
                timeDelta_toLastFree = time.time() - self.time_lastFree
                if timeDelta_toLastFree > 2:
                    x1 = x2 = y1 = y2 = 0
                else:
                    x1, y1, x2, y2 = self.curr_bbox.data

            # --- Update timers for state transitions ---
            if self.time_startPark is not None:
                self.delay_startPark = time.time() - self.time_startPark
            if self.time_inPark is not None:
                self.delay_inPark = time.time() - self.time_inPark
            if self.time_doExit is not None:
                self.delay_doExit = time.time() - self.time_doExit

            # --- State machine for parking process ---
            with self.transition_lock:
                # --- IDLE: Wait for slot to be in the right position to start slow approach ---
                if self.state == "IDLE":
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State IDLE")
                    if x1 > self.conf["xForSlowDrive"] and y2 > self.conf["yForSlowDrive"]:
                        if self.blink_timer is None:
                            self.blink_timer = rospy.Timer(rospy.Duration(0.5), self.blink_all_leds)
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                    elif False:
                        self.state = "SLOW"
                        self.status_text = "Status: SLOW --> approaching slot"
                        self.time_startPark = time.time()
                        self.publish_state(1)

                # --- SLOW: After 2 seconds, transition to PARKING ---
                elif self.state == "SLOW" and self.delay_startPark > 2:
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State SLOW")
                    self.state = "PARKING"
                    self.status_text = "Status: PARKING --> is parking (reverse)"
                    self.time_doPark = time.time()

                # --- PARKING: Use yellow line detection and PID to park ---
                elif self.state == "PARKING":
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State PARKING")
                    if self.camImage is not None:
                        m_yellow, b_yellow = self.detect_yellow(self.camImage)
                        if self.conf["debugPrints_parking"]:
                            rospy.loginfo(f"[PARKING] m_yellow: {m_yellow}; b_yellow: {b_yellow}")
                        # Smooth yellow line parameters over last N frames
                        if m_yellow is not None and b_yellow is not None:
                            self.m_yellow_hist.append(m_yellow)
                            self.b_yellow_hist.append(b_yellow)
                            if len(self.m_yellow_hist) > self.smooth_N:
                                self.m_yellow_hist.pop(0)
                                self.b_yellow_hist.pop(0)
                            m_yellow_smooth = np.median(self.m_yellow_hist)
                            b_yellow_smooth = np.median(self.b_yellow_hist)
                        else:
                            m_yellow_smooth = m_yellow
                            b_yellow_smooth = b_yellow

                        v, omega = self.calculate_control(m_yellow_smooth, b_yellow_smooth)
                        if v is None and omega is None:
                            v = omega = 0
                        reverse_turn = Twist2DStamped(v=v, omega=omega)
                        self.pub_lane_twist.publish(reverse_turn)

                # --- WAIT: After parking, stop for up to 5 seconds ---
                elif self.state == "WAIT" and self.delay_inPark <= 5:
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State WAIT")
                    if self.blink_timer is not None:
                        self.blink_timer.shutdown()
                        self.blink_timer = None
                    stopBot = Twist2DStamped(v=0, omega=0)
                    self.pub_lane_twist.publish(stopBot)

                # --- WAIT: After 5 seconds, prepare to exit ---
                elif self.state == "WAIT":
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State WAIT")
                    if self.blink_timer is None:
                        self.blink_timer = rospy.Timer(rospy.Duration(0.5), self.blink_all_leds)
                    self.time_doExit = time.time()
                    self.state = "EXIT"
                    self.status_text = "Status: EXIT"

                # --- EXIT: For 1 second, drive out of the slot ---
                elif self.state == "EXIT" and self.delay_doExit <= 1:
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State EXIT")
                    exit_turn = Twist2DStamped(v=0.2, omega=-3.5)
                    self.pub_lane_twist.publish(exit_turn)
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] Exiting parking")

                # --- EXIT: After 1 second, stop and return to IDLE ---
                elif self.state == "EXIT":
                    if self.conf["debugPrints_parking"]:
                        rospy.loginfo(f"[PARKING] In State EXIT but stopping now")
                    if self.blink_timer is not None:
                        self.blink_timer.shutdown()
                        self.blink_timer = None
                    self.turn_off_leds()
                    stopBot = Twist2DStamped(v=0, omega=0)
                    self.pub_lane_twist.publish(stopBot)
                    self.state = "IDLE"
                    self.status_text = "Status: IDLE"

            # --- Show debug image with detected lines if enabled ---
            if self.conf['show_lineDetectImage']:
                if self.linedImage is not None:
                    cv2.imshow("Parking", self.linedImage)
                cv2.waitKey(1)

            # --- Publish state to switch_control_node based on current state ---
            if self.state in ["PARKING", "WAIT", "EXIT"]:
                if self.conf["debugPrints_parking"]:
                    rospy.loginfo(f"[PARKING] publish val 5")
                self.pub_state.publish(Int32(5))  # Signal: parking in progress or done
            elif self.state == "SLOW":
                if self.conf["debugPrints_parking"]:
                    rospy.loginfo(f"[PARKING] publish val 1")
                self.pub_state.publish(Int32(1))  # Signal: slow approach
            else:
                if self.conf["debugPrints_parking"]:
                    rospy.loginfo(f"[PARKING] publish nothing")
                # No state published in IDLE

if __name__ == '__main__':
    node = ParkingNode(node_name='parking_node')
    node.run()