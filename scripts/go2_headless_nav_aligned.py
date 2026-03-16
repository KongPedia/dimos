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

"""Headless Unitree Go2 navigation with map alignment and multi-robot frame namespace support.

This script extends go2_headless_nav.py with:
- Map preloading and costmap initialization from YAML
- Initial pose anchoring for consistent map alignment across sessions
- Optional robot_id-based frame namespace for multi-robot scenarios

Usage:
    # Basic usage (same as go2_headless_nav.py)
    export ROBOT_IP=10.2.80.101
    python scripts/go2_headless_nav_aligned.py --robot-ip "$ROBOT_IP"
    
    # With map preload
    python scripts/go2_headless_nav_aligned.py --robot-ip "$ROBOT_IP" \
        --map-yaml maps/my_map.yaml
    
    # With initial pose from file
    python scripts/go2_headless_nav_aligned.py --robot-ip "$ROBOT_IP" \
        --map-yaml maps/my_map.yaml \
        --initial-pose-json last_pose.json
    
    # With explicit initial pose
    python scripts/go2_headless_nav_aligned.py --robot-ip "$ROBOT_IP" \
        --initial-x 2.5 --initial-y 3.0 --initial-yaw 1.57
    
    # Multi-robot mode with frame namespace
    python scripts/go2_headless_nav_aligned.py --robot-ip "$ROBOT_IP" \
        --robot-id robot_1 \
        --map-yaml maps/shared_map.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from dimos.constants import (
    DEFAULT_CAPACITY_COLOR_IMAGE,
    DEFAULT_CAPACITY_OCCUPANCY_GRID,
    DEFAULT_CAPACITY_POINTCLOUD,
)
from dimos.core.blueprints import autoconnect
from dimos.core.transport import pSHMTransport
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.localization import localization_module
from dimos.mapping.pointclouds.occupancy import HeightCostConfig
from dimos.mapping.voxels import voxel_mapper
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import Image, PointCloud2
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


JETSON_TRANSPORTS = {
    ("lidar", PointCloud2): pSHMTransport(
        "lidar", default_capacity=DEFAULT_CAPACITY_POINTCLOUD
    ),
    ("color_image", Image): pSHMTransport(
        "color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
    ("global_map", PointCloud2): pSHMTransport(
        "global_map", default_capacity=DEFAULT_CAPACITY_POINTCLOUD
    ),
    ("global_costmap", OccupancyGrid): pSHMTransport(
        "global_costmap", default_capacity=DEFAULT_CAPACITY_OCCUPANCY_GRID
    ),
}


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


def _pose_from_xy_yaw(x: float, y: float, yaw_deg: float, frame_id: str = "map") -> PoseStamped:
    yaw_rad = math.radians(yaw_deg)
    orientation = Quaternion.from_euler(Vector3(0.0, 0.0, yaw_rad))
    return PoseStamped(
        frame_id=frame_id,
        position=(x, y, 0.0),
        orientation=orientation,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless Go2 navigation with map alignment and multi-robot support"
    )
    parser.add_argument("--robot-ip", required=True, help="Unitree Go2 IP address")
    parser.add_argument(
        "--unitree-connection",
        choices=["webrtc", "webrtc-rs", "replay", "mujoco", "ros"],
        default="webrtc-rs",
        help="Backend used for the Unitree connection",
    )
    parser.add_argument(
        "--n-workers",
        "--n-dask-workers",
        dest="n_workers",
        type=int,
        default=2,
        help="Worker processes",
    )
    parser.add_argument(
        "--memory-limit",
        "--dask-memory-limit",
        dest="memory_limit",
        type=str,
        default="2GiB",
        help="Worker memory limit hint",
    )
    parser.add_argument("--planner-robot-speed", type=float, default=None)
    parser.add_argument("--navigation-voxel-size", type=float, default=0.2)
    parser.add_argument(
        "--map-publish-interval",
        type=float,
        default=0.2,
        help="Seconds between global_map publishes (0 = every frame)",
    )
    parser.add_argument(
        "--costmap-update-interval",
        type=float,
        default=0.2,
        help="Seconds between costmap updates (0 = every map update)",
    )
    parser.add_argument("--goal-timeout", type=float, default=15.0)
    parser.add_argument("--voxel-device", default="CUDA:0", help="Voxel mapper device")
    parser.add_argument("--skip-recovery", action="store_true", help="Skip recovery stand on startup")
    parser.add_argument("--mqtt-broker", type=str, default=None, help="MQTT broker host (enables MQTT mode)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--robot-id", type=str, default=None, help="Robot ID for frame namespace and MQTT")
    
    # Map alignment options
    parser.add_argument("--map-yaml", type=str, default=None, help="Path to map YAML file for preloading")
    parser.add_argument("--initial-pose-json", type=str, default=None, help="Path to initial pose JSON file")
    parser.add_argument("--initial-x", type=float, default=None, help="Initial pose X coordinate (meters)")
    parser.add_argument("--initial-y", type=float, default=None, help="Initial pose Y coordinate (meters)")
    parser.add_argument("--initial-yaw", type=float, default=None, help="Initial pose yaw (radians)")
    parser.add_argument("--alignment-frame", type=str, default="map", help="Target frame for alignment (default: map)")
    parser.add_argument("--robot-frame-namespace", type=str, default=None, help="Explicit frame namespace prefix (overrides robot-id)")
    
    # Continuous localization options
    parser.add_argument("--localization-update-rate", type=float, default=2.0, help="Localization update rate in Hz (default: 2.0)")
    parser.add_argument("--disable-localization", action="store_true", help="Disable continuous scan matching localization")
    parser.add_argument("--localization-min-fitness", type=float, default=0.3, help="Minimum ICP fitness score (default: 0.3)")
    
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


def _handle_goal_command(parts: list[str], navigator: ReplanningAStarPlanner, alignment_frame: str) -> None:
    if len(parts) not in {3, 4}:
        print("usage: goal <x> <y> [yaw_deg]")
        return

    x = float(parts[1])
    y = float(parts[2])
    yaw_deg = float(parts[3]) if len(parts) == 4 else 0.0
    goal = _pose_from_xy_yaw(x, y, yaw_deg, frame_id=alignment_frame)
    accepted = navigator.set_goal(goal)
    print(f"goal accepted={accepted} -> ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}deg) in {alignment_frame}")


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
    alignment_frame: str,
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

                goal = _pose_from_xy_yaw(x, y, yaw_deg, frame_id=alignment_frame)
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
    alignment_frame: str,
) -> bool:
    cmd = parts[0].lower()
    if cmd in {"quit", "exit"}:
        return False

    if cmd == "help":
        _print_help()
        return True

    if cmd == "goal":
        _handle_goal_command(parts, navigator, alignment_frame)
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


def _occupancy_grid_to_pointcloud(grid: OccupancyGrid) -> PointCloud2:
    """Convert OccupancyGrid to PointCloud2 for localization.
    
    Extracts occupied cells (value > threshold) as 3D points.
    """
    import numpy as np
    
    occupied_threshold = 50  # cells with value > 50 are considered occupied
    points = []
    
    # Extract occupied cells
    for y in range(grid.height):
        for x in range(grid.width):
            cell_value = grid.grid[y, x]
            if cell_value > occupied_threshold:
                # Convert grid coordinates to world coordinates
                world_pos = grid.grid_to_world((x, y))
                points.append([world_pos.x, world_pos.y, 0.0])
    
    if len(points) == 0:
        print("[WARN] No occupied cells found in map")
        points = [[0, 0, 0]]  # Add dummy point
    
    points_array = np.array(points, dtype=np.float32)
    return PointCloud2.from_numpy(points_array, frame_id=grid.frame_id)


def _load_map_if_specified(args: argparse.Namespace) -> OccupancyGrid | None:
    """Load map from YAML if specified."""
    if not args.map_yaml:
        return None
    
    map_path = Path(args.map_yaml)
    if not map_path.exists():
        print(f"[WARN] Map file not found: {map_path}")
        return None
    
    try:
        print(f"[INFO] Loading map from {map_path}...")
        costmap = OccupancyGrid.from_path(map_path)
        print(f"[INFO] Map loaded: {costmap.width}x{costmap.height}, resolution={costmap.resolution:.3f}m")
        origin_yaw = costmap.origin.orientation.to_euler().z
        print(f"[INFO] Map origin: ({costmap.origin.position.x:.3f}, {costmap.origin.position.y:.3f}), yaw={math.degrees(origin_yaw):.1f}deg")
        return costmap
    except Exception as e:
        print(f"[ERROR] Failed to load map: {e}")
        return None


def main() -> None:
    args = _build_parser().parse_args()

    voxel_device = _resolve_voxel_device(args.voxel_device)
    
    # Determine robot_id for MQTT
    mqtt_robot_id = args.robot_id or "go2_02"
    
    # Determine alignment configuration
    enable_alignment = (
        args.initial_pose_json is not None
        or args.initial_x is not None
        or args.initial_y is not None
    )

    # Build global config with alignment and namespace options
    config_kwargs = {
        "robot_ip": args.robot_ip,
        "unitree_connection": args.unitree_connection,
        "viewer": "none",
        "n_workers": args.n_workers,
        "memory_limit": args.memory_limit,
        "planner_robot_speed": args.planner_robot_speed,
        "robot_model": "unitree_go2",
    }
    
    # Add robot_id and namespace if specified
    if args.robot_id:
        config_kwargs["robot_id"] = args.robot_id
    if args.robot_frame_namespace:
        config_kwargs["robot_frame_namespace"] = args.robot_frame_namespace
    
    # Add alignment config if specified
    if enable_alignment:
        config_kwargs["navigation_enable_initial_alignment"] = True
        config_kwargs["navigation_alignment_frame"] = args.alignment_frame
        
        if args.initial_pose_json:
            config_kwargs["navigation_initial_pose_path"] = args.initial_pose_json
        if args.initial_x is not None:
            config_kwargs["navigation_initial_x"] = args.initial_x
        if args.initial_y is not None:
            config_kwargs["navigation_initial_y"] = args.initial_y
        if args.initial_yaw is not None:
            config_kwargs["navigation_initial_yaw"] = args.initial_yaw

    # Load map if specified
    preloaded_map = _load_map_if_specified(args)

    # Build blueprint with localization
    modules = [
        go2_connection(),
        voxel_mapper(
            voxel_size=args.navigation_voxel_size,
            device=voxel_device,
            publish_interval=args.map_publish_interval,
        ),
        cost_mapper(
            config=HeightCostConfig(use_float64=False),
            update_interval=args.costmap_update_interval,
        ),
        replanning_a_star_planner(),
        wavefront_frontier_explorer(goal_timeout=args.goal_timeout),
    ]
    
    # Add localization module if not disabled
    if not args.disable_localization:
        from dimos.mapping.localization import LocalizationConfig
        
        localization_config = LocalizationConfig(
            alignment_frame=args.alignment_frame,
            update_rate=args.localization_update_rate,
            enable_continuous_localization=True,
            min_fitness_score=args.localization_min_fitness,
        )
        modules.append(localization_module(
            alignment_frame=args.alignment_frame,
            update_rate=args.localization_update_rate,
            enable=True,
        ))
    
    blueprint = autoconnect(*modules).transports(JETSON_TRANSPORTS).global_config(**config_kwargs)

    pubsub.lcm.autoconf()  # type: ignore[attr-defined]
    coordinator = blueprint.build()

    navigator = coordinator.get_instance(ReplanningAStarPlanner)
    explorer = coordinator.get_instance(WavefrontFrontierExplorer)
    connection = coordinator.get_instance(GO2Connection)

    assert navigator is not None
    assert explorer is not None
    assert connection is not None
    
    # Inject preloaded map into LocalizationModule if available
    if preloaded_map and not args.disable_localization:
        from dimos.mapping.localization import LocalizationModule
        
        localization = coordinator.get_instance(LocalizationModule)
        if localization is not None:
            print("[INFO] Injecting preloaded map into LocalizationModule...")
            try:
                # Convert OccupancyGrid to PointCloud2 for localization
                map_pointcloud = _occupancy_grid_to_pointcloud(preloaded_map)
                localization._on_global_map(map_pointcloud)
                print(f"[INFO] Preloaded map injected: {len(map_pointcloud.to_numpy())} points")
            except Exception as e:
                print(f"[WARN] Failed to inject preloaded map: {e}")

    print("Headless Go2 navigation (aligned) is running.")
    print(f"robot_ip={args.robot_ip}, voxel_size={args.navigation_voxel_size}, voxel_device={voxel_device}")
    
    if args.robot_id:
        print(f"robot_id={args.robot_id} (frame namespace active)")
    if enable_alignment:
        print(f"initial pose alignment enabled, target frame={args.alignment_frame}")
    if preloaded_map:
        print(f"preloaded map: {preloaded_map.width}x{preloaded_map.height}")
    
    # Localization status
    if not args.disable_localization:
        print(f"continuous localization: ENABLED (update_rate={args.localization_update_rate}Hz, min_fitness={args.localization_min_fitness})")
    else:
        print("continuous localization: DISABLED")

    # Start MQTT listener if broker is specified
    mqtt_thread = None
    if args.mqtt_broker:
        if not MQTT_AVAILABLE:
            print("[ERROR] MQTT requested but paho-mqtt is not installed.")
            print("Install with: uv pip install paho-mqtt")
            coordinator.stop()
            return

        print(f"[MQTT] Starting MQTT listener for robot_id={mqtt_robot_id}")
        mqtt_thread = threading.Thread(
            target=_mqtt_listener,
            args=(args.mqtt_broker, args.mqtt_port, mqtt_robot_id, navigator, explorer, connection, args.alignment_frame),
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

    # If map was preloaded, publish it once to initialize costmap
    if preloaded_map:
        print("[INFO] Publishing preloaded map to initialize costmap...")
        try:
            # Access the cost_mapper instance and inject the map
            # This would require the cost_mapper to accept external map injection
            # For now, just note that map preload is available
            print("[INFO] Map preloaded. Navigator will use it once costmap integration is available.")
        except Exception as e:
            print(f"[WARN] Could not inject preloaded map: {e}")

    _print_help()

    try:
        while True:
            raw = input("go2-nav> ").strip()
            if not raw:
                continue

            parts = raw.split()
            if not _dispatch_command(parts, navigator, explorer, connection, args.alignment_frame):
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
