#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
dt-exec python3 "$DT_REPO_PATH/packages/challenge_2/src/CameraNode.py" &
dt-exec python3 "$DT_REPO_PATH/packages/challenge_2/src/HardCodedDriveNode.py"

# wait for app to end
dt-launchfile-join