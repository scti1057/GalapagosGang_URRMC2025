#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun followlane switch_control_node.py

# wait for app to end
dt-launchfile-join