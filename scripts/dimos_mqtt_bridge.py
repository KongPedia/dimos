#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Direct MQTT Bridge for dimos Navigation.

This script runs in the dimos container. It directly connects to the 
Command Center's MQTT broker and subscribes to navigation commands.
It bypasses ROS2 and battlebang completely for navigation execution.

Requirements:
  uv pip install paho-mqtt
"""

import argparse
import json
import logging
import math
import sys
import threading
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.duration import Duration
    from rclpy.time import Time
    try:
        from tf2_ros import Buffer, TransformListener
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    TF_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Error: paho-mqtt is not installed.")
    print("Please run: uv pip install paho-mqtt")
    sys.exit(1)

from dimos import core as dimos_core
from dimos.msgs.geometry_msgs import PoseStamped as DimosPoseStamped
from dimos.msgs.geometry_msgs import Quaternion, Vector3
from dimos.navigation import rosnav as rosnav_module
from dimos.protocol import pubsub
from dimos.utils.logging_config import setup_logger

logger = setup_logger(level=logging.INFO)


def _convert_to_dimos_pose(x: float, y: float, yaw: float | None = 0.0) -> DimosPoseStamped:
    """Convert x, y, yaw to a dimos PoseStamped."""
    if yaw is None:
        yaw = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    
    return DimosPoseStamped(
        ts=time.time(),
        frame_id="odom",  # Modified for dimos native internal frame
        position=Vector3(float(x), float(y), 0.0),
        orientation=Quaternion(0.0, 0.0, float(qz), float(qw)),
    )

def _transform_pose_manual(x: float, y: float, yaw: float, transform_stamped: object) -> tuple[float, float, float]:
    """Manually apply tf2 transform (map -> odom) to x, y, yaw to avoid geometry_msgs dependency."""
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    
    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w
    
    # 1. Convert Quaternion to Euler Yaw
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_tf = math.atan2(siny_cosp, cosy_cosp)
    
    # 2. 2D Rotation + Translation
    new_x = tx + x * math.cos(yaw_tf) - y * math.sin(yaw_tf)
    new_y = ty + x * math.sin(yaw_tf) + y * math.cos(yaw_tf)
    new_yaw = yaw + yaw_tf
    
    return new_x, new_y, new_yaw

class TfListenerNode(Node):
    """Background ROS 2 Node to track tf2 transforms."""
    def __init__(self) -> None:
        super().__init__('dimos_mqtt_bridge_tf_node')
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )



class DimosMqttBridge:
    def __init__(self, rosnav: object, robot_id: str, broker_host: str, broker_port: int, tf_buffer=None) -> None:
        self._rosnav = rosnav
        self._robot_id = robot_id
        self._host = broker_host
        self._port = broker_port
        self._tf_buffer = tf_buffer

        self._topic_move = f"bb/v1/robot/{robot_id}/cmd/move"
        self._topic_patrol = f"bb/v1/robot/{robot_id}/cmd/patrol"
        self._topic_stop = f"bb/v1/robot/{robot_id}/cmd/stop"

        # Setup MQTT Client (v2 API for paho-mqtt 2.x)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"dimos_nav_{robot_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        # Navigation state management
        self._patrol_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    def start(self) -> None:
        logger.info(f"Connecting to MQTT Broker at {self._host}:{self._port}...")
        try:
            self.client.connect(self._host, self._port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
            sys.exit(1)

    def stop(self) -> None:
        logger.info("Stopping MQTT Bridge...")
        self._cancel_event.set()
        if self._patrol_thread and self._patrol_thread.is_alive():
            self._patrol_thread.join(timeout=1.0)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            logger.info("Successfully connected to MQTT Broker.")
            self.client.subscribe(self._topic_move)
            self.client.subscribe(self._topic_patrol)
            self.client.subscribe(self._topic_stop)
            logger.info(f"Subscribed to:\n  - {self._topic_move}\n  - {self._topic_patrol}\n  - {self._topic_stop}")
        else:
            logger.error(f"Failed to connect. Reason code: {reason_code}")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received on {msg.topic}")
            return

        logger.info(f"Received command on {msg.topic}: {payload}")

        if msg.topic == self._topic_move:
            self._handle_move(payload)
        elif msg.topic == self._topic_patrol:
            self._handle_patrol(payload)
        elif msg.topic == self._topic_stop:
            self._handle_stop()

    # -------------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------------

    def _handle_move(self, payload: dict) -> None:
        self._stop_current_navigation()
        
        # Support both flat and nested 'params' structures
        params = payload.get("params", payload)
        
        try:
            x = params["x"]
            y = params["y"]
            yaw = params.get("yaw", 0.0)
        except KeyError as e:
            logger.error(f"Missing required field in MoveCmd: {e}")
            return

        # TF 변환 (map -> odom)
        if self._tf_buffer:
            try:
                if self._tf_buffer.can_transform(
                    'odom',
                    'map',
                    Time(seconds=0),
                    timeout=Duration(seconds=2.0)
                ):
                    t = self._tf_buffer.lookup_transform(
                        'odom',
                        'map',
                        Time(seconds=0),
                        timeout=Duration(seconds=2.0)
                    )
                    x, y, yaw = _transform_pose_manual(x, y, yaw, t)
                    logger.info(f"[TF] map -> odom transformed: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
                else:
                    logger.warning("[TF] Transform not available yet. Passing raw coordinates.")
            except Exception as e:
                logger.warning(f"[TF] Transform failed: {e}. Passing raw coordinates.")

        dimos_pose = _convert_to_dimos_pose(x, y, yaw)
        logger.info(f"Setting dimos goal (odom frame): x={x}, y={y}, yaw={yaw}")
        
        try:
            self._rosnav.set_goal(dimos_pose)  # type: ignore[union-attr]
        except Exception as e:
            logger.error(f"ROSNav set_goal failed: {e}")

    def _handle_patrol(self, payload: dict) -> None:
        self._stop_current_navigation()
        
        # Support both flat and nested 'params' structures
        params = payload.get("params", payload)
        
        try:
            points = params["points"]
            loop = params.get("loop", True)
        except KeyError as e:
            logger.error(f"Missing required field in PatrolCmd: {e}")
            return

        if not points:
            logger.warning("PatrolCmd received with empty points list.")
            return

        self._cancel_event.clear()
        self._patrol_thread = threading.Thread(
            target=self._run_patrol,
            args=(points, loop),
            daemon=True,
            name="DimosPatrolThread"
        )
        self._patrol_thread.start()

    def _handle_stop(self) -> None:
        self._stop_current_navigation()
        logger.info("Navigation stopped via MQTT command.")

    # -------------------------------------------------------------------------
    # Execution Logic
    # -------------------------------------------------------------------------

    def _stop_current_navigation(self) -> None:
        self._cancel_event.set()
        try:
            self._rosnav.cancel_goal()  # type: ignore[union-attr]
        except Exception:
            pass
        
        if self._patrol_thread and self._patrol_thread.is_alive():
            # We don't block heavily to avoid deadlocking the MQTT thread
            pass

    def _run_patrol(self, points: list, loop: bool) -> None:
        logger.info(f"Starting patrol with {len(points)} points. loop={loop}")
        
        while not self._cancel_event.is_set():
            for idx, pt in enumerate(points):
                if self._cancel_event.is_set():
                    logger.info("Patrol cancelled.")
                    return

                x = float(pt[0])
                y = float(pt[1])
                yaw = float(pt[2]) if len(pt) > 2 else 0.0
                
                # TF 변환 (map -> odom)
                if self._tf_buffer:
                    try:
                        if self._tf_buffer.can_transform(
                            'odom',
                            'map',
                            Time(seconds=0),
                            timeout=Duration(seconds=2.0)
                        ):
                            t = self._tf_buffer.lookup_transform(
                                'odom',
                                'map',
                                Time(seconds=0),
                                timeout=Duration(seconds=2.0)
                            )
                            x, y, yaw = _transform_pose_manual(x, y, yaw, t)
                        else:
                            logger.warning("Patrol TF transform not ready.")
                    except Exception as e:
                        logger.warning(f"Patrol TF Transform failed: {e}")

                logger.info(f"Patrol waypoint {idx+1}/{len(points)}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
                dimos_pose = _convert_to_dimos_pose(x, y, yaw)
                
                try:
                    self._rosnav.set_goal(dimos_pose)  # type: ignore[union-attr]
                except Exception as e:
                    logger.error(f"ROSNav set_goal failed during patrol: {e}")
                    return

                # Wait until reached or cancelled
                while not self._cancel_event.is_set():
                    if self._rosnav.is_goal_reached():  # type: ignore[union-attr]
                        break
                    time.sleep(0.1)

            if not loop:
                logger.info("Patrol complete (loop=False).")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct MQTT Bridge for dimos ROSNav")
    parser.add_argument("--workers", type=int, default=2, help="Dask worker count")
    parser.add_argument("--robot_id", type=str, default="go2_02")
    parser.add_argument("--broker_host", type=str, default="localhost", help="MQTT Broker Host")
    parser.add_argument("--broker_port", type=int, default=1883, help="MQTT Broker Port")
    args = parser.parse_args()

    # 0. Initialize ROS2 if available (required by ROSNav and TF)
    tf_node = None
    tf_buffer = None
    executor = None
    if RCLPY_AVAILABLE:
        rclpy.init()
        logger.info("ROS2 (rclpy) initialized successfully.")
        
        if TF_AVAILABLE:
            tf_node = TfListenerNode()
            tf_buffer = tf_node.tf_buffer
            
            # Start TF Node in a separate executor thread
            executor = rclpy.executors.MultiThreadedExecutor()
            executor.add_node(tf_node)
            executor_thread = threading.Thread(target=executor.spin, daemon=True, name="TFExecutorThread")
            executor_thread.start()
            logger.info("Background TF Listener thread started.")
            
            logger.info("Waiting for TF frames to populate...")
            time.sleep(2.0)
            logger.info(f"TF Frames in buffer:\n{tf_buffer.all_frames_as_yaml()}")
        else:
            logger.warning("tf2_ros not available. Frame transforms will not work.")
    else:
        logger.warning("rclpy not found. ROSNav will likely crash on deploy.")

    # 1. Start dimos backend
    logger.info("Starting dimos cluster...")
    pubsub.lcm.autoconf()  # type: ignore[attr-defined]
    dimos_cluster = dimos_core.start(args.workers)

    logger.info("Deploying dimos ROSNav module...")
    rosnav = rosnav_module.deploy(dimos_cluster)

    # 2. Start MQTT Bridge
    bridge = DimosMqttBridge(
        rosnav=rosnav,
        robot_id=args.robot_id,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        tf_buffer=tf_buffer,
    )
    bridge.start()

    logger.info("dimos_mqtt_bridge is ready. Waiting for MQTT Commands...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        bridge.stop()
        try:
            rosnav.stop()  # type: ignore[union-attr]
        except Exception:
            pass
        if RCLPY_AVAILABLE:
            if executor:
                executor.shutdown()
            if tf_node:
                tf_node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
