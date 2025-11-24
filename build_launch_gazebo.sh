colcon build
source install/setup.bash
ros2 launch turtlebot3_gazebo circuit2025.launch.py use_sim_time:=true x_pose:=0.0 y_pose:=-1.8 model:=burger_cam