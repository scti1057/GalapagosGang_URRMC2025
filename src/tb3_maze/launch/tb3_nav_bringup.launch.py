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
    tb3_maze_dir = get_package_share_directory('tb3_maze')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')

    # ---- Launch-Argument: welches Nav2-Parameterfile? ----
    params_file = LaunchConfiguration('params_file')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(tb3_maze_dir, 'config', 'nav2_params.yaml'),
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


    # # ---- Nav2-Stack starten (für REALROBOT, kein use_sim_time) ----
    # nav2 = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
    #     ),
    #     launch_arguments={
    #         'params_file': params_file,
    #         'use_sim_time': 'false',   # wichtig: echte Zeit, kein /clock
    #         'autostart': 'true',
    #         'use_composition': 'False',  # einfacher zum Debuggen
    #         'use_respawn': 'False',
    #         'log_level': 'info',
    #     }.items(),
    # )

    # # ---- BT python Node ----
    mission3_bt_node = Node(
        package='tb3_maze',              # <-- dein Vision-Package-Name
        executable='mission3_bt', # <-- dein Console-Script / Node
        name='mission3_bt',
        output='screen',
        parameters=[{
            'use_sim_time': False,        # auch hier: echte Zeit
        }],
    )


    # # ---- Explorer Node ----
    frontier_explorer_node = Node(
        package='tb3_maze',              # <-- dein Vision-Package-Name
        executable='frontier_explorer', # <-- dein Console-Script / Node
        name='frontier_explorer',
        output='screen',
        parameters=[{
            'use_sim_time': False,        # auch hier: echte Zeit
        }],
    )


    # # ---- entrance Node ----
    drive_in_box = Node(
        package='tb3_maze',              # <-- dein Vision-Package-Name
        executable='drive_in_box', # <-- dein Console-Script / Node
        name='drive_in_box',
        output='screen',
        parameters=[{
            'use_sim_time': False,        # auch hier: echte Zeit
        }],
    )


    # # # ---- GUI Node ----
    # GUI_node = Node(
    #     package='tb3_navigation',              # <-- dein Vision-Package-Name
    #     executable='nav_goal_gui', # <-- dein Console-Script / Node
    #     name='nav_goal_gui',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': False,        # auch hier: echte Zeit
    #     }],
    # )


    # # ---- Beispiel: dein Vision-/Lane-Node aus einem anderen Package ----
    # lane_sign_node = Node(
    #     package='tb3_vision',              # <-- dein Vision-Package-Name
    #     executable='lane_sign_perception', # <-- dein Console-Script / Node
    #     name='lane_sign_perception',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': False,        # auch hier: echte Zeit
    #     }],
    # )

    # Hier kannst du später weitere Nodes eintragen, z.B.:
    # - closed_area_explorer
    # - pallet_handler
    # - mission_manager
    # usw.

    return LaunchDescription([
        declare_params_file,
        # cartographer,
        # nav2,
        mission3_bt_node,
        frontier_explorer_node,
        drive_in_box,
        #lane_sign_node,
    ])
