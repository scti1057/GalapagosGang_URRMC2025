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
        super(ControlLaneNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]


        self.sub_ToF = rospy.Subscriber(f"/{self._vehicle_name}/front_center_tof_driver_node/range", Range, self.cb_ToF, queue_size=1)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        
        self.stop = False
    
        self.HardCoded()
        rospy.on_shutdown(self.fnShutDown)

    def cb_ToF(self, msg):
        self.ToF_data = msg.range
        if self.ToF_data < 0.25:
            rospy.loginfo("Obstacle detected! Stopping the robot.")
            self.drive(0.0, 0.0, 1.0)  # Stop the robot
            self.stop = True
        else:
            self.stop = False
        # rospy.loginfo(f"ToF distance: {self.ToF_data} m")


    def drive(self,v , omega, drive_time):
        # Create message for velocity commands
        cmd_msg = Twist2DStamped()

        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz control loop
        while time.time() - start_time < drive_time:
            if self.stop:
                v = 0.0
                omega = 0.0
            cmd_msg.v = v
            cmd_msg.omega = omega
            self.pub_cmd_vel.publish(cmd_msg)
            rate.sleep()

    def HardCoded(self):

        # drive forwards
        self.drive(0.3, 0.0, 4)

        # turn left
        self.drive(0.2, 2, 3)
        
        # drive forwards
        self.drive(0.3, 0.0, 1)
        
        # turn left
        self.drive(0.2, 2, 3)
        
        # # drive forwards
        self.drive(0.3, 0.0, 4.0)

        # Stop the robot
        self.drive(0.0, 0.0, 1.0)

        # warten
        rospy.sleep(5.0)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)


if __name__ == "__main__":
    # create the node
    node = ControlLaneNode(node_name="control_lane_node")
    # keep the process from terminating
    rospy.spin()
