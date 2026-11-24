"""Run the fixed box event for warehouse A from a sourced ROS 2 workspace."""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock


WORLD = "warehouse_a"
NAME = "trial_aisle_box"
# Gazebo world coordinates. The model's collision bottom sits 0.075 m below its origin.
POSITION = (2.5, 0.0, 0.075)


def service(endpoint, request_type, response_type, request):
    result = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD}/{endpoint}",
         "--reqtype", f"gz.msgs.{request_type}", "--reptype", f"gz.msgs.{response_type}",
         "--timeout", "3000", "--req", request],
        capture_output=True, text=True, timeout=6, check=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(f"Empty response from {endpoint}: {result.stderr.strip()}")
    return result.stdout


def box_present():
    scene = service("scene/info", "Empty", "Scene", "")
    if not re.search(r"(?m)^\s*model\s*\{", scene):
        raise RuntimeError("No models in scene response. Check that warehouse A is running.")
    return bool(re.search(rf'(?m)^\s*name:\s*"{NAME}"\s*$', scene))


def run(node, action, delay):
    stamp = None

    def on_clock(msg):
        nonlocal stamp
        stamp = msg.clock.sec + msg.clock.nanosec * 1e-9

    node.create_subscription(Clock, "/clock", on_clock, qos_profile_sensor_data)
    present = box_present()
    if present == (action == "spawn"):
        raise RuntimeError(f"{NAME} is already {'present' if present else 'absent'}.")

    def next_stamp():
        previous = stamp
        deadline = time.monotonic() + 10
        while stamp == previous:
            if time.monotonic() >= deadline:
                raise RuntimeError("No advancing /clock for 10 seconds. Check Gazebo and the bridge.")
            rclpy.spin_once(node, timeout_sec=0.1)
        if previous is not None and stamp < previous:
            raise RuntimeError("Simulation clock reset during the trial.")
        return stamp

    if action == "spawn":
        model = Path(get_package_share_directory("warehouse_sim")) / "models/box/model.sdf"
        if not model.is_file():
            raise RuntimeError(f"Missing installed model: {model}")
        x, y, z = POSITION
        request = (f'name: "{NAME}" sdf_filename: {json.dumps(str(model))} '
                   f'pose: {{position: {{x: {x} y: {y} z: {z}}} orientation: {{w: 1}}}}')
        endpoint, kind = "create", "EntityFactory"
    else:
        request = f'name: "{NAME}" type: MODEL'
        endpoint, kind = "remove", "Entity"

    armed = next_stamp()
    scheduled = armed + delay
    print(f"{action} scheduled at {scheduled:.3f} sim seconds", file=sys.stderr, flush=True)
    while stamp < scheduled:
        next_stamp()
    requested = stamp
    reply = service(endpoint, kind, "Boolean", request)
    if "data: true" not in reply:
        raise RuntimeError(f"{endpoint} was not acknowledged: {reply.strip()}")
    deadline = time.monotonic() + 8
    while box_present() != (action == "spawn"):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Gazebo acknowledged {endpoint}, but scene verification timed out.")
        time.sleep(0.1)
    # Drain queued /clock samples, then take a fresh clock tick after scene verification.
    for _ in range(qos_profile_sensor_data.depth):
        rclpy.spin_once(node, timeout_sec=0.0)
    print(json.dumps(dict(world=WORLD, entity=NAME, action=action, frame="world",
                          position=POSITION, armed_sim_s=armed, scheduled_sim_s=scheduled,
                          requested_sim_s=requested, confirmed_sim_s=next_stamp())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("spawn", "clear"))
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds of simulation time after arming")
    args = parser.parse_args()
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be finite and nonnegative")
    rclpy.init(args=[])
    node = rclpy.create_node("aisle_box_trial")
    try:
        run(node, args.action, args.delay)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
