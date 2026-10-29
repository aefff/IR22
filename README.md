# IR-Group22-Final-Project

Intelligent Robotics Final Project, Group 22: a warehouse robot that explores an unknown warehouse.

This README covers **Part 1: simulation environment and robot** (Yichen Wang). Everything for Part 1 lives in one ROS 2 package, `warehouse_sim`.

---

## 1. Repository structure

```
IRgroup/                              <- repository root
├── README.md
├── .gitignore                        <- ignores build/ install/ log/ .vscode/
└── group22_ws/                       <- colcon workspace
    └── src/
        └── warehouse_sim/            <- Part 1 package (ament_python)
            ├── package.xml           <- package info and run dependencies
            ├── setup.py              <- installs worlds/, models/, launch/
            ├── setup.cfg
            ├── resource/warehouse_sim
            ├── warehouse_sim/        <- Python module (empty for now)
            ├── test/                 <- default ROS 2 style checks
            ├── worlds/
            │   └── warehouse_a.sdf   <- warehouse world A
            ├── models/
            │   └── warehouse_robot/
            │       ├── model.sdf     <- the robot (body, wheels, sensors, drive plugin)
            │       └── model.config
            └── launch/
                └── warehouse_robot.launch.py   <- starts world + robot + ROS bridge
```

Part 1 files:

| What | File |
|---|---|
| Warehouse world | `group22_ws/src/warehouse_sim/worlds/warehouse_a.sdf` |
| Robot model | `group22_ws/src/warehouse_sim/models/warehouse_robot/model.sdf` |
| One-command launch | `group22_ws/src/warehouse_sim/launch/warehouse_robot.launch.py` |

---

## 2. How to run

Tested on Ubuntu 26.04, ROS 2 Lyrical, Gazebo Jetty. The files avoid absolute paths and version-specific plugin names, so they should also run on the course setup (ROS 2 Jazzy, Gazebo Harmonic).

Requirements: `ros-<distro>-ros-gz` (Gazebo + ROS bridge) and, for driving by keyboard, `ros-<distro>-teleop-twist-keyboard`.

**Build** (from the repository root):

```bash
cd group22_ws
colcon build
source install/setup.bash
```

Re-run `colcon build` after changing any file in `src/`. Files are used from `install/`, not `src/`.

**Start the world and the robot:**

```bash
ros2 launch warehouse_sim warehouse_robot.launch.py
```

**Drive the robot** (in a second terminal, after `source install/setup.bash`):

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Keep the teleop terminal focused. `i` forward, `j`/`l` turn, `k` stop.

**Launch arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `world` | `warehouse_a.sdf` | World file name inside `worlds/` |
| `x` | `-4.0` | Robot start x (m) |
| `y` | `0.0` | Robot start y (m) |
| `yaw` | `0.0` | Robot start heading (rad), 0 = facing +x |

Example, different start pose:

```bash
ros2 launch warehouse_sim warehouse_robot.launch.py x:=-5.0 y:=-1.0 yaw:=1.57
```

The launch file does four things:

1. Starts Gazebo with the chosen world (`-r` = start running immediately).
2. Spawns `models/warehouse_robot/model.sdf` at (`x`, `y`, `yaw`) with `ros_gz_sim create`.
3. Runs `ros_gz_bridge` to connect Gazebo topics to ROS 2 (list below).
4. Publishes static TFs from `base_link` to each sensor frame.

---

## 3. ROS 2 interface

All topics use simulation time (`/clock`), so ROS nodes should set `use_sim_time:=true`.

| Topic | Type | Direction | Rate | `header.frame_id` | Notes |
|---|---|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | ROS → Gazebo | – | – | Drive command (`linear.x`, `angular.z`) |
| `/odom` | `nav_msgs/msg/Odometry` | Gazebo → ROS | 20 Hz | `odom` (child `base_link`) | Wheel odometry |
| `/tf` | `tf2_msgs/msg/TFMessage` | Gazebo → ROS | 20 Hz | `odom` | `odom` → `base_link` |
| `/scan` | `sensor_msgs/msg/LaserScan` | Gazebo → ROS | 10 Hz | `lidar_link` | 2D lidar |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Gazebo → ROS | 15 Hz | `camera_link` | RGB camera |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo → ROS | 15 Hz | `camera_link` | |
| `/thermal/image_raw` | `sensor_msgs/msg/Image` | Gazebo → ROS | 10 Hz | `thermal_link` | Thermal (infrared) camera |
| `/thermal/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo → ROS | 10 Hz | `thermal_link` | |
| `/clock` | `rosgraph_msgs/msg/Clock` | Gazebo → ROS | ~1000 Hz | – | Simulation time (1 ms physics step) |

Rates were measured with `ros2 topic hz` on a running simulation.

**TF tree:**

```
odom ──> base_link ──> lidar_link
                  ├──> camera_link
                  └──> thermal_link
```

- `odom` → `base_link` comes from the DiffDrive plugin (via the bridge).
- `base_link` → sensors are static TFs published by the launch file.
- `base_link` is on the ground, centred between the wheels. There is no `base_footprint`, so for slam_toolbox / Nav2 set `base_frame: base_link`.
- No `map` frame yet: SLAM (Part 2) should publish `map` → `odom`.

---

## 4. Robot parameters

Round differential-drive robot. All values are in `models/warehouse_robot/model.sdf`.

Coordinates below are relative to `base_link` (x forward, y left, z up), in metres.

**Overall**

| Item | Value |
|---|---|
| Type | Differential drive: 2 driven wheels + 1 rear ball caster |
| Footprint | about 0.38 m wide (outer wheel faces) × 0.38 m long (x −0.18 … +0.195, incl. camera lenses) |
| Footprint radius (for Nav2) | about 0.2 m |
| Height | 0.25 m (top of the lidar) |
| Ground clearance | 0.06 m (bottom of the chassis) |
| Total mass | 2.9 kg (chassis 2.0, wheels 2 × 0.3, caster 0.1, lidar 0.1, cameras 2 × 0.05) |
| `base_link` | Model origin, on the ground at the centre of the chassis. The wheel axle is 0.04 m in front of it. |

**Body and wheels**

| Part | Shape | Size | Position (x, y, z) | Mass |
|---|---|---|---|---|
| Chassis | Cylinder | radius 0.18, height 0.11 (0.06–0.17 m above ground) | (0, 0, 0.115) | 2.0 kg |
| Left wheel | Cylinder | radius 0.08, width 0.04 | (0.04, 0.17, 0.08) | 0.3 kg |
| Right wheel | Cylinder | radius 0.08, width 0.04 | (0.04, -0.17, 0.08) | 0.3 kg |
| Rear caster | Sphere, frictionless | radius 0.04 | (-0.13, 0, 0.04) | 0.1 kg |

Overall footprint: about 0.38 m wide, 0.38 m long. The narrowest aisle in world A is 1.4 m.

**Drive (DiffDrive plugin)**

| Parameter | Value |
|---|---|
| Wheel separation | 0.34 m |
| Wheel radius | 0.08 m |
| Max linear speed | ±0.5 m/s |
| Max angular speed | ±1.0 rad/s |
| Odometry rate | 20 Hz |
| Acceleration limits | None set |
| Command topic / odometry topic | `cmd_vel` / `odom` |
| Odometry frames | `odom` → `base_link` |

If you change the wheel size or position, update `wheel_separation` / `wheel_radius` in the DiffDrive plugin as well, or the odometry will be wrong.

---

## 5. Sensors

All sensors are currently **noise-free**. Gaussian noise can be added in each `<sensor>` block if needed.

| Sensor | Frame | Mounted at (x, y, z) | Topic | Output | Rate |
|---|---|---|---|---|---|
| 2D LiDAR | `lidar_link` | (0, 0, 0.21) | `/scan` | 360 beams, 360°, 0.12–10 m | 10 Hz |
| RGB camera | `camera_link` | (0.185, 0.04, 0.13) | `/camera/image_raw` | 640 × 480 RGB, 60° FOV | 15 Hz |
| Thermal camera | `thermal_link` | (0.185, −0.04, 0.13) | `/thermal/image_raw` | 320 × 240 8-bit grey, 60° FOV | 10 Hz |

Both cameras look straight ahead (+x) with no tilt. All sensors are `always_on` and shown in the Gazebo GUI (`visualize`).

### 2D LiDAR (`lidar_link`, topic `/scan`)

| Parameter | Value |
|---|---|
| Type | `gpu_lidar` |
| Position | (0, 0, 0.21), top centre of the robot |
| Field of view | 360° |
| Samples | 360 (1° resolution) |
| Range | 0.12 – 10.0 m |
| Update rate | 10 Hz |

The scan plane is at 0.21 m on purpose: the racks in world A have a solid base 0.3 m high, so a lidar mounted below 0.3 m sees them as walls. If you raise the lidar above 0.3 m it will see through the racks.

### RGB camera (`camera_link`, topic `/camera/image_raw`)

| Parameter | Value |
|---|---|
| Type | `camera` |
| Position | (0.185, 0.04, 0.13), front, left of centre |
| Resolution | 640 × 480, `R8G8B8` |
| Horizontal FOV | 1.047 rad (60°) |
| Clip range | 0.05 – 20 m |
| Update rate | 15 Hz |

### Thermal camera (`thermal_link`, topic `/thermal/image_raw`)

| Parameter | Value |
|---|---|
| Type | `thermal` |
| Position | (0.185, -0.04, 0.13), front, right of centre |
| Resolution | 320 × 240, 8-bit greyscale (`L8`), brighter = hotter |
| Horizontal FOV | 1.047 rad (60°) |
| Temperature range | 253.15 – 673.15 K, 3 K per grey level |
| Clip range | 0.1 – 20 m |
| Update rate | 10 Hz |

In Gazebo, every object is at ambient temperature unless it is given a temperature. To make an object hot, add this inside its `<visual>`:

```xml
<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">
  <temperature>350.0</temperature>
</plugin>
```

---

## 6. World A (`worlds/warehouse_a.sdf`)

World origin is the centre of the warehouse, x to the east (towards the racks), y to the north. All objects are static. The numbers below are taken from the `<collision>` blocks, which is what the lidar and the physics see.

**General**

| Item | Value |
|---|---|
| World name | `warehouse_a` |
| Ground plane | 20 × 20 m, grey |
| Inside of the walls | 14 × 10 m (x −7 … 7, y −5 … 5), 140 m² |
| Lighting | One directional light (`sun`) with shadows |
| Physics | Gazebo defaults: 1 ms step, real-time factor 1 |
| World plugins | `Physics`, `UserCommands`, `SceneBroadcaster`, `Sensors` (ogre2) |
| Default robot start | (−4, 0), facing +x, in the open area west of the racks |

**Walls** (thickness 0.2 m, height 2 m)

| Model | Centre (x, y) | Size x × y × z |
|---|---|---|
| `wall_north` | (0, 5.1) | 14 × 0.2 × 2 |
| `wall_south` | (0, −5.1) | 14 × 0.2 × 2 |
| `wall_east` | (7.1, 0) | 0.2 × 10.4 × 2 |
| `wall_west` | (−7.1, 0) | 0.2 × 10.4 × 2 |

**Racks** (3 identical yellow racks, long side along y)

| Model | Centre (x, y) | Occupies |
|---|---|---|
| `rack_1` | (1.5, 0) | x 1.2 … 1.8, y −3 … 3 |
| `rack_2` | (3.5, 0) | x 3.2 … 3.8, y −3 … 3 |
| `rack_3` | (5.5, 0) | x 5.2 … 5.8, y −3 … 3 |

Each rack is 6 m long, 0.6 m deep and 1.8 m high:

| Part | Size | Height above floor |
|---|---|---|
| Solid base | 6 × 0.6 × 0.3 m | 0 – 0.3 m |
| Shelf boards (3) | 6 × 0.6 × 0.04 m | centred at 0.31, 0.95, 1.6 m |
| Posts (8) | 0.06 × 0.06 × 1.8 m | 0 – 1.8 m, every ~2 m along the rack |

The boxes on the shelves are visual only (no collision), so the lidar and physics do not see them. The cameras do.

**Pallet stacks** (each 1.2 × 1.0 m on the floor)

| Model | Centre (x, y) | Occupies | Height |
|---|---|---|---|
| `pallet_stack_1` | (−5.5, 2.8) | x −6.1 … −4.9, y 2.3 … 3.3 | 1.35 m |
| `pallet_stack_2` | (−3.0, 2.8) | x −3.6 … −2.4, y 2.3 … 3.3 | 0.95 m |
| `pallet_stack_3` | (−5.5, −2.8) | x −6.1 … −4.9, y −3.3 … −2.3 | 1.75 m |
| `pallet_stack_4` | (5.0, 4.2) | x 4.4 … 5.6, y 3.7 … 4.7 | 0.95 m |

**Free space and passages**

| Passage | Width |
|---|---|
| Aisle between `rack_1` and `rack_2` (x 1.8 … 3.2) | 1.4 m |
| Aisle between `rack_2` and `rack_3` (x 3.8 … 5.2) | 1.4 m |
| Between `rack_3` and the east wall (x 5.8 … 7.0) | 1.2 m |
| Cross passages at the rack ends (y 3 … 5 and y −5 … −3) | 2.0 m |
| Narrowest gap: `rack_3` north end ↔ `pallet_stack_4` (y 3.0 … 3.7) | 0.7 m |
| `pallet_stack_4` ↔ north wall (y 4.7 … 5.0) | 0.3 m (too narrow for the robot) |
| Open area west of the racks (x −7 … 1.2) | 8.2 × 10 m, with 3 pallet stacks |

The robot is about 0.38 m wide, so it fits through every passage except the 0.3 m gap behind `pallet_stack_4`.

**Floor markings** (`floor_markings`): yellow/black striped bays and green lane lines, 2 mm high, visual only (no collision).

The world includes the `gz-sim-sensors-system` plugin. **Any new world must include it too**, or the lidar and cameras will publish nothing:

```xml
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>
```

---

## 7. How to change things

| I want to... | Edit | Also update |
|---|---|---|
| Move or resize a sensor | `<pose>` of that link in `model.sdf` | The matching `static_tf(...)` line in `warehouse_robot.launch.py` |
| Change sensor resolution / range / rate | That `<sensor>` block in `model.sdf` | Nothing |
| Change wheel size or spacing | Wheel links in `model.sdf` | `wheel_radius` / `wheel_separation` in the DiffDrive plugin |
| Change speed limits | DiffDrive plugin in `model.sdf` | Nothing |
| Add a new ROS topic from Gazebo | `bridge` arguments in `warehouse_robot.launch.py` | Nothing |
| Add a new world | Put the `.sdf` in `worlds/` (with the Sensors plugin) | Run with `world:=<file>.sdf` |

After any change: `colcon build`, then `source install/setup.bash`.

Portability rules (so the files run on both Jazzy/Harmonic and Lyrical/Jetty):

- No absolute paths such as `/home/<user>/...`. Find files with `get_package_share_directory('warehouse_sim')`.
- `<sdf version="1.9">`.
- Plugin file names without version numbers, e.g. `gz-sim-diff-drive-system`.
- Every `<pose>` needs 6 numbers: `x y z roll pitch yaw`.
- Each sensor has both `<frame_id>` and `<gz_frame_id>`. Harmonic only reads `gz_frame_id`, while Jetty reads `frame_id`. Keep both, or the sensor messages get the wrong `frame_id` on one of the two versions. On Jetty this prints a harmless warning: `XML Element[gz_frame_id], child of element[sensor], not defined in SDF`.
