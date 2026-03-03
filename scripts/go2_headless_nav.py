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

Supports two modes:
  - Terminal REPL: interactive goal/explore commands (default)
  - MQTT mode: receive move/patrol/stop commands from broker (--mqtt-broker)

Usage:
    export ROBOT_IP=10.2.80.101
    # Terminal REPL:
    python scripts/go2_headless_nav.py --robot-ip "$ROBOT_IP" \\
        --map-file /path/to/lit5f_field.yaml --planner-robot-speed 0.8

    # MQTT mode:
    python scripts/go2_headless_nav.py --robot-ip "$ROBOT_IP" \\
        --map-file /path/to/lit5f_field.yaml \\
        --mqtt-broker 10.2.80.46 --robot-id go2_02
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
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
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis
from unitree_webrtc_connect.constants import RTC_TOPIC


# =============================================================================
# Helpers
# =============================================================================

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


# =============================================================================
# MQTT Bridge (move / patrol / stop)
# =============================================================================

class MQTTNavigationBridge:
    """Bridge between MQTT commands and dimos navigation."""

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        robot_id: str,
        navigator: ReplanningAStarPlanner,
        explorer: WavefrontFrontierExplorer,
        connection: GO2Connection,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.robot_id = robot_id
        self.navigator = navigator
        self.explorer = explorer
        self.connection = connection
        self.running = False

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"go2_nav_{robot_id}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def start(self) -> None:
        try:
            print(f"[MQTT] Connecting to {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.running = True
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            raise

    def stop(self) -> None:
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        print("[MQTT] Disconnected")

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        if reason_code == 0:
            topic_move   = f"bb/v1/robot/{self.robot_id}/cmd/move"
            topic_patrol = f"bb/v1/robot/{self.robot_id}/cmd/patrol"
            topic_stop   = f"bb/v1/robot/{self.robot_id}/cmd/stop"
            client.subscribe(topic_move)
            client.subscribe(topic_patrol)
            client.subscribe(topic_stop)
            print(f"[MQTT] Subscribed to {topic_move}, {topic_patrol}, {topic_stop}")
        else:
            print(f"[MQTT] Connection failed: {reason_code}")

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            print(f"[MQTT] Received: {msg.topic} -> {payload}")

            if "cmd/move" in msg.topic:
                self._handle_move(payload)
            elif "cmd/patrol" in msg.topic:
                self._handle_patrol(payload)
            elif "cmd/stop" in msg.topic:
                cancelled = self.navigator.cancel_goal()
                print(f"[MQTT] Goal cancelled: {cancelled}")

        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")

    def _handle_move(self, payload: dict) -> None:
        x       = float(payload.get("x", 0.0))
        y       = float(payload.get("y", 0.0))
        yaw_deg = float(payload.get("yaw", 0.0))
        goal    = _pose_from_xy_yaw(x, y, yaw_deg)
        accepted = self.navigator.set_goal(goal)
        print(f"[MQTT] Move goal: ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}°) accepted={accepted}")

    def _handle_patrol(self, payload: dict) -> None:
        points = payload.get("points", [])
        if not points:
            print("[MQTT] Empty patrol path")
            return
        print(f"[MQTT] Patrol with {len(points)} waypoints")
        # Navigate to first waypoint; TODO: full waypoint sequencing
        p = points[0]
        x, y = float(p[0]), float(p[1])
        yaw_deg = float(p[2]) if len(p) > 2 else 0.0
        goal = _pose_from_xy_yaw(x, y, yaw_deg)
        accepted = self.navigator.set_goal(goal)
        print(f"[MQTT] Patrol wp[0]: ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}°) accepted={accepted}")


# =============================================================================
# Terminal REPL helpers
# =============================================================================

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
    x       = float(parts[1])
    y       = float(parts[2])
    yaw_deg = float(parts[3]) if len(parts) == 4 else 0.0
    goal    = _pose_from_xy_yaw(x, y, yaw_deg)
    accepted = navigator.set_goal(goal)
    print(f"goal accepted={accepted} -> ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}deg)")


def _handle_explore_command(
    parts: list[str], explorer: WavefrontFrontierExplorer
) -> None:
    if len(parts) != 2:
        print("usage: explore <start|stop|status>")
        return
    action = parts[1].lower()
    if action == "start":
        print(f"explore started={explorer.explore()}")
    elif action == "stop":
        print(f"explore stopped={explorer.stop_exploration()}")
    elif action == "status":
        print(f"explore active={explorer.is_exploration_active()}")
    else:
        print("usage: explore <start|stop|status>")


def _handle_robot_command(parts: list[str], connection: GO2Connection) -> None:
    cmd = parts[0].lower()
    if cmd == "recovery":
        print("[INFO] Running recovery stand...")
        result = connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": 1006})
        print(f"recovery result={result}")
        time.sleep(1.0)
    elif cmd == "standup":
        print(f"standup result={connection.standup()}")
    elif cmd == "liedown":
        print(f"liedown result={connection.liedown()}")


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
    elif cmd == "goal":
        _handle_goal_command(parts, navigator)
    elif cmd == "explore":
        _handle_explore_command(parts, explorer)
    elif cmd in {"recovery", "standup", "liedown"}:
        _handle_robot_command(parts, connection)
    elif cmd == "state":
        print(f"planner state={navigator.get_state()}")
        print(f"goal reached={navigator.is_goal_reached()}")
    elif cmd == "cancel":
        print(f"cancelled={navigator.cancel_goal()}")
    else:
        print(f"unknown command: {cmd}")
        _print_help()
    return True


# =============================================================================
# Argument parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless Go2 navigation — terminal REPL or MQTT mode"
    )
    parser.add_argument("--robot-ip", required=True, help="Unitree Go2 IP address")
    # Map
    parser.add_argument(
        "--map-file", type=str, default=None,
        help="Path to nav2-format YAML map file (enables static global map)",
    )
    # Planner
    parser.add_argument("--n-dask-workers", type=int, default=6)
    parser.add_argument("--planner-robot-speed", type=float, default=None)
    parser.add_argument("--navigation-voxel-size", type=float, default=0.2)
    parser.add_argument("--goal-timeout", type=float, default=15.0)
    parser.add_argument("--voxel-device", default="CUDA:0")
    # Robot
    parser.add_argument("--skip-recovery", action="store_true",
                        help="Skip recovery stand on startup")
    # MQTT (optional — enables MQTT mode)
    parser.add_argument("--mqtt-broker", type=str, default=None,
                        help="MQTT broker host (enables MQTT mode, overrides terminal REPL)")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--robot-id", type=str, default="go2_02")
    # Viewer
    parser.add_argument("--viewer", type=str, default="none",
                        choices=["none", "rerun", "rerun-web", "foxglove"],
                        help="Viewer backend (default: none)")
    parser.add_argument("--viewer-port", type=int, default=7779,
                        help="WebSocket vis port (default: 7779)")
    return parser


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _build_parser().parse_args()
    voxel_device = _resolve_voxel_device(args.voxel_device)

    print("=" * 60)
    print("Go2 dimos Navigation (headless)")
    print("=" * 60)
    print(f"Robot IP:        {args.robot_ip}")
    print(f"Robot ID:        {args.robot_id}")
    print(f"Voxel size:      {args.navigation_voxel_size}m  device={voxel_device}")
    print(f"Static map:      {args.map_file or 'None (LiDAR-only mode)'}")
    if args.mqtt_broker:
        print(f"Mode:            MQTT  ({args.mqtt_broker}:{args.mqtt_port})")
    else:
        print("Mode:            Terminal REPL")
    print("=" * 60)

    # Build global config
    global_config_kwargs: dict[str, Any] = dict(
        robot_ip=args.robot_ip,
        viewer_backend="rerun-web",
        n_dask_workers=args.n_dask_workers,
        planner_robot_speed=args.planner_robot_speed,
        robot_model="unitree_go2",
    )
    if args.map_file:
        global_config_kwargs["mujoco_global_costmap_from_occupancy"] = args.map_file
    global_config_kwargs["viewer_backend"] = args.viewer

    blueprint = autoconnect(
        go2_connection(),
        voxel_mapper(voxel_size=args.navigation_voxel_size, device=voxel_device),
        cost_mapper(),
        replanning_a_star_planner(),
        wavefront_frontier_explorer(goal_timeout=args.goal_timeout),
        websocket_vis(port=args.viewer_port),
    ).global_config(**global_config_kwargs)

    pubsub.lcm.autoconf()  # type: ignore[attr-defined]
    coordinator = blueprint.build()

    navigator  = coordinator.get_instance(ReplanningAStarPlanner)
    explorer   = coordinator.get_instance(WavefrontFrontierExplorer)
    connection = coordinator.get_instance(GO2Connection)

    assert navigator  is not None
    assert explorer   is not None
    assert connection is not None

    # Recovery stand on startup
    if not args.skip_recovery:
        print("\n[INFO] Running recovery stand...")
        try:
            time.sleep(2.0)
            result = connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": 1006})
            print(f"[INFO] Recovery stand result: {result}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] Recovery stand failed: {e}")

    # -------------------------------------------------------------------------
    # MQTT mode
    # -------------------------------------------------------------------------
    if args.mqtt_broker:
        if not MQTT_AVAILABLE:
            print("[ERROR] paho-mqtt is not installed. Run: pip install paho-mqtt")
            coordinator.stop()
            sys.exit(1)

        mqtt_bridge = MQTTNavigationBridge(
            args.mqtt_broker, args.mqtt_port, args.robot_id,
            navigator, explorer, connection,
        )

        def _signal_handler(sig: int, frame: object) -> None:
            print("\n[INFO] Shutting down...")
            mqtt_bridge.stop()
            coordinator.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        mqtt_bridge.start()
        print("[INFO] MQTT mode running. Waiting for commands... (Ctrl+C to stop)\n")
        try:
            while mqtt_bridge.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")
        finally:
            mqtt_bridge.stop()
            coordinator.stop()

    # -------------------------------------------------------------------------
    # Terminal REPL mode
    # -------------------------------------------------------------------------
    else:
        print("[INFO] Navigation running. Type 'help' for commands.\n")
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
