#!/bin/bash
source /environment.sh
dt-launchfile-init

dt-exec python3 "$DT_REPO_PATH/packages/challenge_1/src/read_key.py" &
dt-exec python3 "$DT_REPO_PATH/packages/challenge_1/src/change_speed.py" & 
dt-exec python3 "$DT_REPO_PATH/packages/challenge_1/src/crtl_arrow_keys.py"

dt-launchfile-join
