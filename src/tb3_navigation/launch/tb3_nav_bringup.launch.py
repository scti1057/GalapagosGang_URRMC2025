#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Pfade zu deinem Nav-Package und zu nav2_bringup
    tb3_navigation_dir = get_package_share_directory('tb3_navigation')

    # ---- red_sign_detector Node ----
    red_sign_detector_params = os.path.join(tb3_navigation_dir, 'config', 'red_sign_detector_params.yaml')
    red_sign_detector_node = Node(
        package='tb3_navigation',
        executable='red_sign_detector',
        name='red_sign_detector',
        output='log',
        parameters=[red_sign_detector_params],
    )

    # ---- reference_line_node Node ----
    reference_line_node_params = os.path.join(tb3_navigation_dir, 'config', 'reference_line_node_params.yaml')
    reference_line_node = Node(
        package='tb3_navigation',
        executable='reference_line_node',
        name='reference_line_node',
        output='log',
        parameters=[reference_line_node_params],
    )

    # ---- reference_line_node Node ----
    red_sign_localizer_minimalized_params = os.path.join(tb3_navigation_dir, 'config', 'red_sign_localizer_minimalized_params.yaml')
    red_sign_localizer = Node(
        package='tb3_navigation',
        executable='red_sign_localizer',
        name='red_sign_localizer',
        output='log',
        parameters=[red_sign_localizer_minimalized_params],
    )

    return LaunchDescription([
        red_sign_detector_node,
        reference_line_node,
        # red_sign_localizer,
        #lane_sign_node,
    ])
