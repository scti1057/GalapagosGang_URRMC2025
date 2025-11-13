#!/usr/bin/env python3

import rospy
import os
import yaml
from std_msgs.msg import Float64
from duckietown_msgs.msg import Twist2DStamped
from duckietown.dtros import DTROS, NodeType


class ControlLaneNode(DTROS):
    def __init__(self, node_name):
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ['VEHICLE_NAME']

        # flags / state
        self.enable = False
        self.debug = False              # console logging toggle
        self.lane_x = None              # last received lane center in pixels

        # Load configuration from YAML
        config_path = 'packages/challenge_3_solution/detect_lane.yaml'
        with open(config_path, 'r') as f:
            self.conf = yaml.safe_load(f)

        # Speed limits
        self.v_min = self.conf.get("v_min", 0.1)
        self.v_max = self.conf.get("v_max", 0.3)

        # === Publisher ===
        # Sends control commands to the robot
        lane_cmd_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_lane_twist = rospy.Publisher(lane_cmd_topic, Twist2DStamped, queue_size=1)

        # === Subscriber ===
        # Target center X (in pixels) from the lane detection node
        rospy.Subscriber(
            f"/{self._vehicle_name}/detect/lane_x",
            Float64,
            self.cbFollowLane,
            queue_size=1
        )

        # === PID Controller parameters ===
        self.kp = 8.0
        self.ki = 0.1
        self.kd = 0.4

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = rospy.Time.now()

        rospy.on_shutdown(self.fnShutDown)

    # callback: lane_x updated
    def cbFollowLane(self, msg):
        """
        Gets called whenever a new lane_x is published from the vision node.
        We store it and immediately run lane following control.
        """
        self.lane_x = msg.data

        if self.lane_x is not None:
            self.followLane(self.lane_x)

    # PID lane-following logic
    def followLane(self, center_pixel_x):
        """
        center_pixel_x: detected desired x-position in the image (pixel coord)
        We convert that into an error vs. image center, run PID, publish Twist2DStamped.
        """

        # assume a 640px wide camera image -> adjust if your camera is different
        image_width = 640.0
        image_center = image_width / 2.0

        # steering error: positive if target is left of center (-> turn left)
        error = (image_center - center_pixel_x) / image_center  # normalize to [-1,1]

        # timing for PID
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # === PID calculation (P + I + D) ===
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        omega = self.kp * error + self.ki * self.integral + self.kd * derivative

        # clamp steering rate
        if omega > 5.0:
            omega = 5.0
        elif omega < -5.0:
            omega = -5.0

        # === forward velocity selection ===
        # basic slowdown on large steering error
        error_abs = abs(error)
        if error_abs > 1.0:
            error_abs = 1.0
        v = self.v_max - (self.v_max - self.v_min) * error_abs

        # === publish twist command ===
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v = v
        twist.omega = omega
        self.pub_lane_twist.publish(twist)

        if self.debug:
            rospy.loginfo(
                "[PID] e=%.3f  P=%.3f  I=%.3f  D=%.3f  ω=%.3f  v=%.3f" %
                (error,
                 self.kp * error,
                 self.ki * self.integral,
                 self.kd * derivative,
                 omega,
                 v)
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
