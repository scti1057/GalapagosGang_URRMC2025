import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/duckie5/turtlebot3_ws/install/tb3_lane_fusion'
