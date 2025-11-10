#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch camera node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/camera_reader_node.py" &

# launch duckie detection node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/detect_object_node.py" &

# launch display node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/display_yoloResult_node.py" & 

# launch parking node
dt-exec python3 "$DT_REPO_PATH/packages/followlane/src/parking_node.py"

# wait for app to end
dt-launchfile-join