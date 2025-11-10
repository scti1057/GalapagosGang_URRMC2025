#!/usr/bin/env python3

import rospy
import os
import yaml
from std_msgs.msg import Float64, Int32
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType
from switch_control_node import ControlState

class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self.enable = False
        self.debug = False  # Debug mode for console output
        self.duckie_info = 0  # 0 = no duckie, 1 = far, 2 = close

        # Load configuration from YAML
        config_path = 'packages/followlane/config/detect_lane.yaml'
        with open(config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Speed limits
        self.v_min = self.conf.get("v_min", 0.1)
        self.v_max = self.conf.get("v_max", 0.3)

        # === Publisher ===
        # Sends control commands to SwitchControlNode
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        # === Subscribers ===
        # Target center X from SwitchControlNode
        self.sub_target_x = rospy.Subscriber(
            f"/{self._vehicle_name}/control/selected_x", Float64, self.cbFollowLane, queue_size=1
        )
        # Duckie detection info (0, 1, 2)
        self.sub_duckie_info = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/duckie/info", Int32, self.cbDuckieInfo, queue_size=1
        )
        # Vehicle control mode from SwitchControlNode
        self.sub_vehicle_status = rospy.Subscriber(
            f"/{self._vehicle_name}/switch/control", Int32, self.cbVehicleStatus, queue_size=1
        )

        self.vehicle_status = 0  # Current state from switch_control_node

        # Stop handling
        self.stop_cmd_count = 0
        self.stop_sent = False

        # === PID Controller parameters ===
        self.kp = 8
        self.ki = 0.1
        self.kd = 0.4

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()

        rospy.on_shutdown(self.fnShutDown)

    # Callback for duckie info (used to influence speed)
    def cbDuckieInfo(self, msg: Int32):
        self.duckie_info = msg.data

    # Callback for vehicle status (e.g. lane mode, intersection, etc.)
    def cbVehicleStatus(self, msg: Int32):
        self.vehicle_status = msg.data

        # Enable controller if vehicle is in an active driving state
        self.enable = msg.data in [ControlState.LANE_NORMAL.value, ControlState.INTERSECTION.value, ControlState.LANE_SLOW.value, ControlState.OBSTACLE.value]

        if not self.enable:
            if not self.stop_sent:
                # Send up to 3 STOP commands when disabled
                stop_twist = Twist2DStamped()
                stop_twist.header.stamp = rospy.Time.now()
                stop_twist.v = 0.0
                stop_twist.omega = 0.0
                self.pub_lane_twist.publish(stop_twist)
                self.stop_cmd_count += 1

                if self.debug:
                    rospy.loginfo(f"[STATUS] Sent STOP command {self.stop_cmd_count}/3")

                if self.stop_cmd_count >= 3:
                    self.stop_sent = True  # After 3 stops, do not send more
            else:
                if self.debug:
                    rospy.loginfo_throttle(3, "[STATUS] STOP already sent, suppressing further commands")
        else:
            if self.debug:
                rospy.loginfo(f"[STATUS] Control enabled: {self.enable}")

    # Callback for new target center x (used to compute error)
    def cbFollowLane(self, desired_center):
        # Reset stop state if new input received while enabled
        if self.stop_sent:
            self.stop_cmd_count = 0
            self.stop_sent = False
            if self.debug:
                rospy.loginfo("[STATUS] Movement resumed → STOP state reset")

        if not self.enable:
            if self.debug:
                rospy.loginfo("[CONTROL] Control disabled – no command sent")
            return

        if self.debug:
            rospy.loginfo(f"[CONTROL] Received target X: {desired_center.data}")
        self.followLane(desired_center.data)

    # PID lane-following logic
    def followLane(self, center):
        image_center = 640 / 2  # Assuming 640px image width
        error = (image_center - center) / image_center  # Normalized error in range [-1, 1]

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # === PID calculation ===
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative
        omega = max(min(omega, 5.0), -5.0)  # Clamp omega

        # Adjust velocity based on vehicle status and tracking error
        if self.vehicle_status in [1, 2, 4]:  # Specific driving modes
            v = self.v_min
            if self.debug:
                rospy.loginfo(f"[Speed] Vehicle status {self.vehicle_status} → v_min used")
        else:
            error_abs = min(abs(error), 1.0)
            v = self.v_max - (self.v_max - self.v_min) * error_abs  # Slow down on large error

        # Publish twist command
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v     = v
        twist.omega = omega
        self.pub_lane_twist.publish(twist)

        if self.debug:
            rospy.loginfo(
                f"[PID] e={error:.3f}, P={self.kp * error:.3f}, I={self.ki * self.integral:.3f}, "
                f"D={self.kd * derivative:.3f}, ω={omega:.3f}, v={v:.3f} [Duckie-Info: {self.duckie_info}]"
            )

    # Called on shutdown to ensure vehicle stops
    def fnShutDown(self):
        if self.debug:
            rospy.loginfo("[SHUTDOWN] Stopping vehicle")
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        for _ in range(5):
            self.pub_lane_twist.publish(stop_msg)
            rospy.sleep(0.1)

if __name__ == '__main__':
    node = ControlLaneNode(node_name='control_lane_node')
    rospy.spin()
