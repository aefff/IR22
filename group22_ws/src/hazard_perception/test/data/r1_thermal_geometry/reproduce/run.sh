#!/bin/bash
# Re-record the R1 thermal-geometry validation in an isolated headless Gazebo
# session. Requires a built workspace. Writes into a new temporary directory
# (printed at the end); compare results.json with ../validation_record.json
# "raw_results".
D=$(cd "$(dirname "$0")" && pwd)
WS=$(cd "$D/../../../../../.." && pwd)  # group22_ws
SH=$WS/install/warehouse_sim/share/warehouse_sim
RUN=$(mktemp -d -t r1_geometry.XXXXXX) && cd "$RUN" || exit 1
source /opt/ros/jazzy/setup.bash; source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=82 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST GZ_PARTITION=r1_geometry
sed 's/@INITIAL_TEMPERATURE_K@/600.0/' "$SH/models/fire/model.sdf" > fire_600K.sdf
PIDS=""
cleanup() { for p in $PIDS; do kill -INT -- -"$p" 2>/dev/null; done; sleep 3
            for p in $PIDS; do kill -KILL -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
setsid gz sim -s -r --headless-rendering "$SH/worlds/warehouse_a.sdf" > gz.log 2>&1 & PIDS="$PIDS $!"
sleep 8
ros2 run ros_gz_sim create -world warehouse_a -file "$SH/models/warehouse_robot/model.sdf" -name warehouse_robot -x 0 -y 0 -z 0
ros2 run ros_gz_sim create -world warehouse_a -file "$D/hot_sphere.sdf" -name r1_target -x 0 -y 0 -z -5
ros2 run ros_gz_sim create -world warehouse_a -file fire_600K.sdf -name fire_cone -x 0 -y 0 -z -8
setsid ros2 run ros_gz_bridge parameter_bridge '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' \
  '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V' '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
  '/thermal/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/thermal/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' \
  --ros-args -p use_sim_time:=true > bridge.log 2>&1 & PIDS="$PIDS $!"
setsid ros2 run tf2_ros static_transform_publisher --x 0.185 --y -0.04 --z 0.13 \
  --frame-id base_link --child-frame-id thermal_link --ros-args -p use_sim_time:=true > tf_static.log 2>&1 & PIDS="$PIDS $!"
setsid ros2 run warehouse_sim fire_detection --ros-args -p use_sim_time:=true > detector.log 2>&1 & PIDS="$PIDS $!"
sleep 5
python3 "$D/record_geometry.py" "$RUN/results.json" "$(cat "$D/cases.json")"
echo "results: $RUN/results.json"
