<img src="image.png" width="200" height="100" align="right">

# Upper Rhine Mobile Robotics Challenge 2025 - Galapagos Gang

## Introduction

This repository contains the source code and configuration files for the **Galapagos Gang** team's entry in the **Upper Rhine Robotic Mobile Challenge (URRMC) 2025**.

The project is designed to control a **TurtleBot3 Burger** robot to autonomously complete a series of missions as defined in the challenge regulations. These missions include:

* **Track Following:** Autonomous navigation along a defined path.
* **Obstacle Avoidance:** Detecting and navigating around static and dynamic obstacles.
* **Restricted Area Navigation (Tunnel/Maze):** Entering and exiting a closed zone without wall contact.
* **Pallet Management:** Autonomous detection, pickup, transport, and drop-off of pallets.

The codebase utilizes **ROS 2** (Robot Operating System) to orchestrate sensor fusion, navigation, and state management.

## Prerequisites

Before running the code, ensure your system meets the following hardware and software requirements.

### Hardware

* **Robot:** TurtleBot3 Burger
* **Sensors:**
    * **LDS-02 LiDAR**
    * Raspberry Pi Camera (for lane, sign, and pallet detection)
* **Actuators:**
    * **Electro-magnetic module:** Mounted at the front of the robot for picking up and dropping pallets (activated/deactivated via GPIO/ROS topic).
* **Computing:** Raspberry Pi 4 (onboard) / PC (for simulation and monitoring)

### Software

* **OS:** Ubuntu 22.04 LTS (Jammy Jellyfish) or Ubuntu 24.04 LTS (Noble Numbat)
* **Middleware:** ROS 2 **Humble Hawksbill** or **Jazzy Jalisco** (Check your specific install)
* **Dependencies:**
    * `gazebo_ros` (for simulation)
    * `nav2_bringup` (Navigation2 stack)
    * `turtlebot3_msgs`
    * `turtlebot3_gazebo`
    * Python dependencies (see `requirements.txt` if available, or install standard science stack: `numpy`, `opencv-python`, `ultralytics`)

## Installation

Follow these steps to set up the development environment on your TurtleBot3 or simulation PC.

1. **Clone the Repository**
    Create a ROS 2 workspace (if you haven't already) and clone this repo into the `src` directory.

    ```bash
    mkdir -p ~/urrmc_ws/src
    cd ~/urrmc_ws/src
    git clone https://github.com/scti1057/GalapagosGang_URRMC2025.git .
    ```

1. **Install Dependencies**
    Use `rosdep` to install system dependencies required by the packages.

    ```bash
    cd ~/urrmc_ws
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y
    ```

1. **Build the Workspace**
    Compile the packages using `colcon`.

    ```bash
    colcon build --symlink-install
    ```

1. **Source the Overlay**
    Source the setup script to add the new packages to your environment path.

    ```bash
    source install/setup.bash
    ```

    *(Optional: Add this line to your `~/.bashrc` for convenience)*

## Usage

The repository is organized into modular packages. Ensure the robot drivers are running before launching specific challenges.

### 1. Launching the Robot (Real Hardware)

On the TurtleBot3, you must start the basic drivers and the camera node. Open two terminals (or use a multiplexer) on the robot:

#### Terminal 1: Basic Drivers

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

#### Terminal 2: Camera

```bash
ros2 launch turtlebot3_bringup camera_robot.launch.py
```

### 2. Running Specific Challenges

#### Challenge 1: Lane Following & Sign Detection

For this challenge, the robot follows the track markers. You must launch the lane following logic and ensuring the YOLO object detector is active for sign recognition.

1. **Launch Lane Following:**

    ```bash
    ros2 launch galapagos_regelt lane_following.launch.py
    ```

1. **Activate YOLO (in a separate terminal):**

    ```bash
    ros2 run galapagos_checked_yolo yolo_detector_node
    ```

#### Challenge 2: Parkour (Obstacle Avoidance)

*Note: The autonomous state machine in galapagos_regelt was not fully finalized for this specific stage.*

For the competition run, we utilized the **Nav2 stack** directly. The strategy involves launching the navigation stack and manually sending the robot a goal pose corresponding to the end of the obstacle course.

1. **Launch Red Sign Detector:**

    ```bash
    ros2 launch tb3_navigation tb3_nav_bringup.launch.py
    ```

1. **Launch Cartographer, Navigator and GUI:**

    ```bash
    ros2 launch tb3_nav tb3_nav_bringup.launch.py
   ```

1. **Send Goal:** Use the launched app to set the target pose at the end of the course.

#### Challenge 3: Tunnel (Restricted Area)

This challenge requires navigating a maze-like environment (the "Tunnel") without using the camera (LiDAR-only) and without touching the walls. We use the ```tb3_maze package``` for this task.

```bash
ros2 launch tb3_maze tb3_nav_bringup.launch.py
```

#### Challenge 4: Palleting

This launches the palleting logic, including pallet detection and control of the electro-magnetic module to pick up and drop pallets.

```bash
ros2 launch galapagos_regelt challenge4_palleting.launch.py
```

### 3. Simulation

To test the logic without hardware, use the provided simulation launch script.

```bash
# Example: Launching the Gazebo world
./src/build_launch_gazebo.sh
```

## Repository Structure

```galapagos_regelt```: The high-level state machine and logic controller ("The Brain").

```galapagos_checked_yolo```: Object detection node integrating YOLO for sign/pallet recognition.

```tb3_lane_fusion```: Sensor fusion package combining camera data (lane lines) with LiDAR data.

```tb3_navigation```: Configuration for the Navigation2 stack (maps, costmaps, planners).

```tb3_maze```: Solvers and logic for the Tunnel/Maze environment.

```turtlebot3_simulations```: Custom Gazebo worlds and models for the URRMC 2025 track.

## Regulations Compliance

This code is engineered to comply with the **URRMC 2025 Contest Rules**:

* **Fair Play:** The code allows for autonomous operation during scored runs.

* **Safety:** The obstacle avoidance modules utilize the LDS-02 LiDAR to maintain safe distances, preventing collisions (Malus points).

* **Task Execution:** The specific launch files allow for targeted execution of the points defined in "Table 1 Points attribution" (e.g., Track following, Area navigation, Pallet manipulation).
