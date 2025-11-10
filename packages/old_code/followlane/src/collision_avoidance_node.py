#!/usr/bin/env python3
import rospy
import os
from sensor_msgs.msg import Range
from std_msgs.msg import Int32

class ToFCollisionAvoidanceNode:
    def __init__(self):
        rospy.init_node('tof_collision_avoidance_node')

        # === Parameters ===
        self.threshold = 0.2          # Meters: full STOP below this distance
        self.slow_threshold = 0.4     # Meters: SLOW between threshold and slow_threshold

        self.debug = rospy.get_param("~debug", False)
        self.vehicle_name = os.environ.get("VEHICLE_NAME", "default_bot")

        # === Internal state ===
        self.distance = None  # Most recent ToF sensor reading

        # === Publisher ===
        # Publishes 1 (slow), 3 (stop), or nothing depending on distance
        self.ToFInfo_pub = rospy.Publisher(
            f'/{self.vehicle_name}/tof/avoidance/info', Int32, queue_size=1
        )

        # === Subscriber ===
        # Subscribe to front-center Time-of-Flight sensor topic
        tof_topic = f'/{self.vehicle_name}/front_center_tof_driver_node/range'
        rospy.Subscriber(tof_topic, Range, self.range_callback, queue_size=1)

        if self.debug:
            rospy.loginfo(f"[ToF] Node started for vehicle '{self.vehicle_name}' on topic: {tof_topic}")

    # Callback function for ToF sensor measurements
    def range_callback(self, msg):
        self.distance = msg.range
        if self.debug:
            rospy.loginfo(f"[ToF] New measurement: {self.distance:.2f} m")

    def run(self):
        rate = rospy.Rate(20)  # Loop at 20 Hz
        while not rospy.is_shutdown():
            if self.distance is not None:
                # Too close: issue STOP signal
                if self.distance < self.threshold:
                    if self.debug:
                        rospy.logwarn(f"[ToF] STOP – distance below threshold: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(3))

                # Close but not critical: issue SLOW signal
                elif self.distance < self.slow_threshold:
                    if self.debug:
                        rospy.loginfo(f"[ToF] SLOW – distance critical: {self.distance:.2f} m")
                    self.ToFInfo_pub.publish(Int32(1))

                # Safe distance: do not publish (fallback to 0 handled externally)
                else:
                    if self.debug:
                        rospy.loginfo_throttle(5, f"[ToF] OK – distance: {self.distance:.2f} m")

            rate.sleep()

if __name__ == '__main__':
    node = ToFCollisionAvoidanceNode()
    node.run()
