#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Headless Unitree Go2 navigation control for Jetson.

This script runs the native Go2 navigation stack without the web UI and lets you
control goal navigation / frontier exploration from the terminal.

Usage:
    export ROBOT_IP=10.2.80.101
    python scripts/go2_headless_nav.py --robot-ip "$ROBOT_IP" \
        --navigation-voxel-size 0.2 --planner-robot-speed 0.7
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from typing import Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from dimos.core.blueprints import autoconnect
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.voxels import voxel_mapper
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
    wavefront_frontier_explorer,
)
from dimos.navigation.replanning_a_star.module import (
    ReplanningAStarPlanner,
    replanning_a_star_planner,
)
from dimos.protocol import pubsub
from dimos.robot.unitree.go2.connection import GO2Connection, go2_connection
from unitree_webrtc_connect.constants import RTC_TOPIC


def _resolve_voxel_device(requested_device: str) -> str:
    if not requested_device.upper().startswith("CUDA"):
        return requested_device

    try:
        import open3d.core as o3c  # type: ignore[import-untyped]
    except Exception:
        print("[warn] Open3D import failed. Falling back to CPU:0 for voxel mapper.")
        return "CPU:0"

    if not o3c.cuda.is_available():
        print(
            "[warn] Open3D CUDA backend is not available. "
            "VoxelGridMapper will run on CPU:0."
        )
        return "CPU:0"

    return requested_device


def _pose_from_xy_yaw(x: float, y: float, yaw_deg: float) -> PoseStamped:
    yaw_rad = math.radians(yaw_deg)
    orientation = Quaternion.from_euler(Vector3(0.0, 0.0, yaw_rad))
    return PoseStamped(
        frame_id="world",
        position=(x, y, 0.0),
        orientation=orientation,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless Go2 navigation terminal control")
    parser.add_argument("--robot-ip", required=True, help="Unitree Go2 IP address")
    parser.add_argument("--n-dask-workers", type=int, default=6, help="Dask workers")
    parser.add_argument("--planner-robot-speed", type=float, default=None)
    parser.add_argument("--navigation-voxel-size", type=float, default=0.2)
    parser.add_argument("--goal-timeout", type=float, default=15.0)
    parser.add_argument("--voxel-device", default="CUDA:0", help="Voxel mapper device")
    parser.add_argument("--skip-recovery", action="store_true", help="Skip recovery stand on startup")
    parser.add_argument("--mqtt-broker", type=str, default=None, help="MQTT broker host (enables MQTT mode)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--robot-id", type=str, default="go2_02", help="Robot ID for MQTT topics")
    return parser


def _print_help() -> None:
    print("\nCommands:")
    print("  goal <x> <y> [yaw_deg]     # send navigation goal (default yaw=0)")
    print("  explore start              # start frontier exploration")
    print("  explore stop               # stop frontier exploration")
    print("  explore status             # print exploration status")
    print("  state                      # print planner state")
    print("  cancel                     # cancel current goal")
    print("  recovery                   # run recovery stand (unlock robot)")
    print("  standup                    # make robot stand up")
    print("  liedown                    # make robot lie down")
    print("  help                       # show commands")
    print("  quit                       # stop and exit\n")


def _handle_goal_command(parts: list[str], navigator: ReplanningAStarPlanner) -> None:
    if len(parts) not in {3, 4}:
        print("usage: goal <x> <y> [yaw_deg]")
        return

    x = float(parts[1])
    y = float(parts[2])
    yaw_deg = float(parts[3]) if len(parts) == 4 else 0.0
    goal = _pose_from_xy_yaw(x, y, yaw_deg)
    accepted = navigator.set_goal(goal)
    print(f"goal accepted={accepted} -> ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}deg)")


def _handle_explore_command(parts: list[str], explorer: WavefrontFrontierExplorer) -> None:
    if len(parts) != 2:
        print("usage: explore <start|stop|status>")
        return

    action = parts[1].lower()
    if action == "start":
        print(f"explore started={explorer.explore()}")
        return
    if action == "stop":
        print(f"explore stopped={explorer.stop_exploration()}")
        return
    if action == "status":
        print(f"explore active={explorer.is_exploration_active()}")
        return

    print("usage: explore <start|stop|status>")


def _handle_robot_command(parts: list[str], connection: GO2Connection) -> None:
    if len(parts) != 1:
        print(f"usage: {parts[0]}")
        return

    cmd = parts[0].lower()
    if cmd == "recovery":
        print("[INFO] Running recovery stand to unlock robot...")
        result = connection.publish_request(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": 1006}  # RecoveryStand
        )
        print(f"recovery result={result}")
        time.sleep(1.0)
        return

    if cmd == "standup":
        print("[INFO] Making robot stand up...")
        result = connection.standup()
        print(f"standup result={result}")
        return

    if cmd == "liedown":
        print("[INFO] Making robot lie down...")
        result = connection.liedown()
        print(f"liedown result={result}")
        return


def _mqtt_listener(
    broker_host: str,
    broker_port: int,
    robot_id: str,
    navigator: ReplanningAStarPlanner,
    explorer: WavefrontFrontierExplorer,
    connection: GO2Connection,
) -> None:
    """Background thread to listen for MQTT commands"""
    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if reason_code == 0:
            topic_move = f"bb/v1/robot/{robot_id}/cmd/move"
            topic_stop = f"bb/v1/robot/{robot_id}/cmd/stop"
            client.subscribe(topic_move)
            client.subscribe(topic_stop)
            print(f"[MQTT] Connected and subscribed to {topic_move}, {topic_stop}")
        else:
            print(f"[MQTT] Connection failed: {reason_code}")

    def on_message(client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            print(f"[MQTT] Received: {msg.topic} -> {payload}")
            
            if "cmd/move" in msg.topic:
                params = payload.get("params", payload)
                x = float(params["x"])
                y = float(params["y"])
                yaw_deg = float(params.get("yaw", 0.0))
                
                goal = _pose_from_xy_yaw(x, y, yaw_deg)
                accepted = navigator.set_goal(goal)
                print(f"[MQTT] Goal set: ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}deg) accepted={accepted}")
            
            elif "cmd/stop" in msg.topic:
                cancelled = navigator.cancel_goal()
                print(f"[MQTT] Goal cancelled: {cancelled}")
        
        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"go2_nav_{robot_id}")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker_host, broker_port, 60)
        print(f"[MQTT] Connecting to {broker_host}:{broker_port}...")
        client.loop_forever()
    except Exception as e:
        print(f"[MQTT] Connection error: {e}")


def _dispatch_command(
    parts: list[str],
    navigator: ReplanningAStarPlanner,
    explorer: WavefrontFrontierExplorer,
    connection: GO2Connection,
) -> bool:
    cmd = parts[0].lower()
    if cmd in {"quit", "exit"}:
        return False

    if cmd == "help":
        _print_help()
        return True

    if cmd == "goal":
        _handle_goal_command(parts, navigator)
        return True

    if cmd == "explore":
        _handle_explore_command(parts, explorer)
        return True

    if cmd in {"recovery", "standup", "liedown"}:
        _handle_robot_command(parts, connection)
        return True

    if cmd == "state":
        print(f"planner state={navigator.get_state()}")
        print(f"goal reached={navigator.is_goal_reached()}")
        return True

    if cmd == "cancel":
        print(f"cancelled={navigator.cancel_goal()}")
        return True

    print(f"unknown command: {cmd}")
    _print_help()
    return True


def main() -> None:
    args = _build_parser().parse_args()

    voxel_device = _resolve_voxel_device(args.voxel_device)

    # Same stack as `dimos run unitree-go2`, but headless (no UI module).
    blueprint = autoconnect(
        go2_connection(),
        voxel_mapper(voxel_size=args.navigation_voxel_size, device=voxel_device),
        cost_mapper(),
        replanning_a_star_planner(),
        wavefront_frontier_explorer(goal_timeout=args.goal_timeout),
    ).global_config(
        robot_ip=args.robot_ip,
        viewer_backend="none",
        n_dask_workers=args.n_dask_workers,
        planner_robot_speed=args.planner_robot_speed,
        robot_model="unitree_go2",
    )

    pubsub.lcm.autoconf()  # type: ignore[attr-defined]
    coordinator = blueprint.build()

    navigator = coordinator.get_instance(ReplanningAStarPlanner)
    explorer = coordinator.get_instance(WavefrontFrontierExplorer)
    connection = coordinator.get_instance(GO2Connection)

    assert navigator is not None
    assert explorer is not None
    assert connection is not None

    print("Headless Go2 navigation is running.")
    print(f"robot_ip={args.robot_ip}, voxel_size={args.navigation_voxel_size}, voxel_device={voxel_device}")
    
    # Start MQTT listener if broker is specified
    mqtt_thread = None
    if args.mqtt_broker:
        if not MQTT_AVAILABLE:
            print("[ERROR] MQTT requested but paho-mqtt is not installed.")
            print("Install with: uv pip install paho-mqtt")
            coordinator.stop()
            return
        
        print(f"[MQTT] Starting MQTT listener for robot_id={args.robot_id}")
        mqtt_thread = threading.Thread(
            target=_mqtt_listener,
            args=(args.mqtt_broker, args.mqtt_port, args.robot_id, navigator, explorer, connection),
            daemon=True,
            name="MQTTListener"
        )
        mqtt_thread.start()
        print("[MQTT] MQTT listener started. Commands will be received from MQTT broker.")

    # Run recovery stand to unlock robot after standup
    if not args.skip_recovery:
        print("\n[INFO] Running recovery stand to unlock robot after standup...")
        try:
            time.sleep(2.0)  # Wait for standup to complete
            result = connection.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": 1006}  # RecoveryStand
            )
            print(f"[INFO] Recovery stand result: {result}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] Failed to run recovery stand: {e}")
            print("[WARN] You may need to manually unlock the robot with the controller")

    _print_help()

    try:
        while True:
            raw = input("go2-nav> ").strip()
            if not raw:
                continue

            parts = raw.split()
            if not _dispatch_command(parts, navigator, explorer, connection):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            explorer.stop_exploration()
        except Exception:
            pass
        coordinator.stop()


if __name__ == "__main__":
    main()
