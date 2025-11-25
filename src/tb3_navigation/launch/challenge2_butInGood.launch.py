#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():    # Pfade zu deinem Nav-Package und zu nav2_bringup
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    tb3_navigation_dir = get_package_share_directory('tb3_navigation')
    tb3_lane_fusion_dir = get_package_share_directory('tb3_lane_fusion')

    # ---- Launch-Argument: welches Nav2-Parameterfile? ----
    params_file = LaunchConfiguration('params_file')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(tb3_navigation_dir, 'config', 'nav2_params_ch2.yaml'),
        description='Full path to the Nav2 parameters file'
    )

    # ---- Cartographer (SLAM) starten, damit /map & TF map->odom entstehen ----
    # cartographer = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')
    #     ),
    #     launch_arguments={
    #         'use_sim_time': 'false',   # echter Roboter, kein /clock
    #         'resolution': '0.005',
    #     }.items(),
    # )


    # ---- Nav2-Stack starten (für REALROBOT, kein use_sim_time) ----
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'params_file': params_file,
            'use_sim_time': 'false',   # wichtig: echte Zeit, kein /clock
            'autostart': 'true',
            'use_composition': 'False',  # einfacher zum Debuggen
            'use_respawn': 'False',
            'log_level': 'info',
        }.items(),
    )

    # ---- lane_bev_left Node ----
    lane_bev_node = Node(
        package='tb3_lane_fusion',
        executable='lane_bev_node',
        name='lane_bev_node',
        output='log',
        parameters=[
        ],
    )

    # ---- lane_mask_left Node ----
    lane_mask_left = Node(
        package='tb3_lane_fusion',
        executable='lane_bev_node',
        name='lane_bev_node',
        output='log',
        parameters=[{
            'base_frame': 'base_footprint',   # wichtig: echte Zeit, kein /clock
            'bev_mask_topic': 'lane_bev/mask_yellow',
            'lane_map_topic': '/left_lane_map',  # einfacher zum Debuggen
            'x_near_m': 0.15,
            'x_far_m': 0.39,
            'y_left_m': -0.16,
            'y_right_m': 0.19,  # einfacher zum Debuggen
            'pixel_step': 6,
        }],
    )

    # ---- lane_mask_right Node ----
    lane_mask_right = Node(
        package='tb3_lane_fusion',
        executable='lane_bev_node',
        name='lane_bev_node',
        output='log',
        parameters=[{
            'base_frame': 'base_footprint',   # wichtig: echte Zeit, kein /clock
            'bev_mask_topic': 'lane_bev/mask_white',
            'lane_map_topic': '/right_lane_map',  # einfacher zum Debuggen
            'x_near_m': 0.15,
            'x_far_m': 0.39,
            'y_left_m': -0.16,
            'y_right_m': 0.19,  # einfacher zum Debuggen
            'pixel_step': 6,
        }],
    )


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
    red_sign_localizer_minimalized_params = os.path.join(tb3_navigation_dir, 'config', 'red_sign_localizer_minimalized_params.yaml')
    red_sign_localizer = Node(
        package='tb3_navigation',
        executable='red_sign_localizer',
        name='red_sign_localizer',
        output='log',
        parameters=[red_sign_localizer_minimalized_params],
    )

    return LaunchDescription([
        declare_params_file,
        nav2,
        lane_bev_node,
        lane_mask_left,
        lane_mask_right,
        red_sign_detector_node,
        red_sign_localizer,
    ])
