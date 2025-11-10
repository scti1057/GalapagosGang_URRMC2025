#!/bin/bash

source /environment.sh

# Initialize launch file
dt-launchfile-init

# ----------------------------------------------------------------------------
# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# 1. Start the main control switch node (decision logic)
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/switch_control_node.py" &

# 2. Start camera reader node (lane and image processing)
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/camera_reader_node.py" &

# 3. Start lane following controller
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/control_lane_node.py" &

# 4. Start object detection node (YOLO, duckies, bots, parking slots)
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/detect_object_node.py" &

# 5. Start YOLO result display node (visualization only)
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/display_yoloResult_node.py" &

# 6. Start intersection handling node (red line detection and intersection logic)
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/intersection_handling_node.py" 

# Optional: Start parking node (uncomment if needed)
# dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/parking_node.py" &

# Optional: Start collision avoidance node (uncomment if needed)
# dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/collision_avoidance_node.py" &

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
# ----------------------------------------------------------------------------

# Wait for all background processes to finish
dt-launchfile-join
