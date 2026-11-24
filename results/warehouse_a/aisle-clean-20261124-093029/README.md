# Warehouse A aisle reroute

On 24 November 2026, the clear and blocked runs used the same map-frame goal
at (6.642 m, −2.475 m). Their starting estimates differed by 0.019 m and
0.201 rad. The trial requested the box after Nav2 published its first plan.

| Measurement | Clear aisle | Blocked aisle |
| --- | ---: | ---: |
| First planned path length | 5.11 m | 5.06 m |
| Estimated distance travelled | 5.10 m | 13.42 m |
| Time from setup to result, simulation seconds | 17.80 | 46.65 |
| NavigateToPose status | SUCCEEDED | SUCCEEDED |

Gazebo confirmed the box at 618.070 s. At 619.375 s Nav2 published a 13.12 m
path around the rack. The interval from spawn confirmation to the first detour
plan was 1.305 s. The exact first LiDAR detection time is unknown because the
trial recorded only the centre beam. Distances travelled sum consecutive
map-frame robot positions from action feedback.

`clear.jsonl` and `blocked.jsonl` contain the two original runs. The box
service and confirmation belong to the scenario runner. Navigation used
the robot's sensor map and Nav2. This pair demonstrates one successful
reroute. Repeatability and reachable-area coverage require separate trials.
