# Frontier exploration

Start Warehouse A or B with live SLAM, hazards disabled, and the warehouse
Nav2 launch. Use the same ROS environment and domain for the explorer.
Finish any manual goal and stop teleop before giving the explorer control.

```bash
ros2 run warehouse_sim frontier_explorer
ros2 run warehouse_sim frontier_explorer --navigate --max-goals 1
ros2 run warehouse_sim frontier_explorer --navigate
```

The first command reads the latest map and prints a proposed pose. The second
attempts one goal. The third runs successive goals until the available
approaches are exhausted. Ctrl+C requests cancellation of the current goal.
Each process starts its own attempt history.

A frontier is a known-free cell next to an unknown cell. The selector groups
frontier cells, discards clusters smaller than 0.20 m measured in cell widths,
and searches for the nearest reachable viewpoint on the known side. It uses
the warehouse navigation radius of 0.22 m. Nav2 performs the actual planning,
control and obstacle avoidance. After each result, the explorer waits for the
next map publication before choosing again. It skips approaches within 0.50 m
of a previous attempt in that session.

For a partial-map smoke test, leave some accessible floor unexplored and
inspect the first proposed pose in the map frame. For an end-to-end coverage
trial, start a fresh simulation with live SLAM, run continuous exploration
from the initial map, and record the final map for evaluation. Exhausting the
selector's eligible viewpoints is a stopping condition. Measuring the proposal's
90% reachable-area target requires an independent reachable-area reference.
