#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
dt-exec python3 "$DT_REPO_PATH/packages/challenge_3/white_yellow_red_calibration.py"

# wait for app to end
dt-launchfile-join
