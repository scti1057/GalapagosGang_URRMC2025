#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tb3_lane_fusion',
            namespace='tb3_lane_fusion',
            executable='lane_bev_node',
            name='lane_bev_node',
            output='screen'
        ),
        Node(
            package='tb3_lane_fusion',
            namespace='tb3_lane_fusion',
            executable='lane_map_node',
            name='lane_map_node',
            output='screen'
        ),
        Node(
            package='tb3_lane_fusion',
            namespace='tb3_lane_fusion',
            executable='slam_interface_node',
            name='slam_interface_node',
            output='screen'
        ),
    ])
