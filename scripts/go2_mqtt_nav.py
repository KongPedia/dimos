#!/usr/bin/env python3
"""
Go2 Navigation with MQTT Command Integration

This script runs dimos Go2 navigation and listens to MQTT commands
from BattleBang Command Center to set navigation goals.

Usage:
    python go2_mqtt_nav.py --robot-ip 192.168.12.1 --broker-host 10.2.80.46 --robot-id go2_02
"""

import argparse
import json
import logging
import threading
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Error: paho-mqtt is not installed.")
    print("Please run: uv pip install paho-mqtt")
    exit(1)

from dimos import core as dimos_core
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial
from dimos.utils.logging_config import setup_logger

logger = setup_logger(level=logging.INFO)


class Go2MqttNavigator:
    def __init__(self, robot_ip: str, robot_id: str, broker_host: str, broker_port: int):
        self.robot_ip = robot_ip
        self.robot_id = robot_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        
        self.topic_move = f"bb/v1/robot/{robot_id}/cmd/move"
        self.topic_patrol = f"bb/v1/robot/{robot_id}/cmd/patrol"
        self.topic_stop = f"bb/v1/robot/{robot_id}/cmd/stop"
        
        # MQTT Client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"go2_nav_{robot_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # dimos robot instance
        self.robot = None
        
        # Navigation state
        self._cancel_event = threading.Event()
        self._patrol_thread = None

    def start(self):
        """Start dimos robot and MQTT connection"""
        logger.info("Starting dimos Go2 robot...")
        
        # Deploy dimos robot with spatial navigation
        # Blueprint is called directly with robot_ip parameter
        self.robot = unitree_go2_spatial(robot_ip=self.robot_ip)
        
        logger.info("dimos Go2 robot deployed successfully")
        
        # Connect to MQTT
        logger.info(f"Connecting to MQTT Broker at {self.broker_host}:{self.broker_port}...")
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
            raise

    def stop(self):
        """Stop MQTT and robot"""
        logger.info("Stopping Go2 MQTT Navigator...")
        self._cancel_event.set()
        
        if self._patrol_thread and self._patrol_thread.is_alive():
            self._patrol_thread.join(timeout=1.0)
        
        self.client.loop_stop()
        self.client.disconnect()
        
        if self.robot:
            try:
                self.robot.stop()
            except Exception:
                pass

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info("Successfully connected to MQTT Broker.")
            self.client.subscribe(self.topic_move)
            self.client.subscribe(self.topic_patrol)
            self.client.subscribe(self.topic_stop)
            logger.info(f"Subscribed to:\n  - {self.topic_move}\n  - {self.topic_patrol}\n  - {self.topic_stop}")
        else:
            logger.error(f"Failed to connect. Reason code: {reason_code}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received on {msg.topic}")
            return

        logger.info(f"Received command on {msg.topic}: {payload}")

        if msg.topic == self.topic_move:
            self._handle_move(payload)
        elif msg.topic == self.topic_patrol:
            self._handle_patrol(payload)
        elif msg.topic == self.topic_stop:
            self._handle_stop()

    def _handle_move(self, payload: dict):
        """Handle move command"""
        self._stop_current_navigation()
        
        # Support both flat and nested 'params' structures
        params = payload.get("params", payload)
        
        try:
            x = float(params["x"])
            y = float(params["y"])
            yaw = float(params.get("yaw", 0.0))
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid move command: {e}")
            return

        logger.info(f"Navigating to: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        
        try:
            # Use dimos robot's goto_global method
            # Coordinates are in map frame (odom frame for Go2)
            self.robot.goto_global(x, y)
            logger.info("Navigation goal sent successfully")
        except Exception as e:
            logger.error(f"Navigation failed: {e}")

    def _handle_patrol(self, payload: dict):
        """Handle patrol command"""
        self._stop_current_navigation()
        
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
            name="Go2PatrolThread"
        )
        self._patrol_thread.start()

    def _handle_stop(self):
        """Handle stop command"""
        self._stop_current_navigation()
        logger.info("Navigation stopped via MQTT command.")

    def _stop_current_navigation(self):
        """Stop any ongoing navigation"""
        self._cancel_event.set()
        
        if self._patrol_thread and self._patrol_thread.is_alive():
            pass  # Thread will check cancel_event

    def _run_patrol(self, points: list, loop: bool):
        """Run patrol through waypoints"""
        logger.info(f"Starting patrol with {len(points)} points. loop={loop}")
        
        while not self._cancel_event.is_set():
            for idx, pt in enumerate(points):
                if self._cancel_event.is_set():
                    logger.info("Patrol cancelled.")
                    return

                x = float(pt[0])
                y = float(pt[1])
                
                logger.info(f"Patrol waypoint {idx+1}/{len(points)}: x={x:.2f}, y={y:.2f}")
                
                try:
                    self.robot.goto_global(x, y)
                except Exception as e:
                    logger.error(f"Navigation failed during patrol: {e}")
                    return

                # Wait for goal to be reached (simple timeout-based)
                time.sleep(5.0)  # TODO: implement proper goal reached check

            if not loop:
                logger.info("Patrol complete (loop=False).")
                break


def main():
    parser = argparse.ArgumentParser(description="Go2 Navigation with MQTT Integration")
    parser.add_argument("--robot-ip", type=str, default="192.168.12.1", help="Unitree Go2 IP address")
    parser.add_argument("--robot-id", type=str, default="go2_02", help="Robot ID for MQTT topics")
    parser.add_argument("--broker-host", type=str, default="localhost", help="MQTT Broker Host")
    parser.add_argument("--broker-port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--navigation-voxel-size", type=float, default=0.2, help="Navigation voxel size")
    parser.add_argument("--planner-robot-speed", type=float, default=0.8, help="Planner robot speed")
    parser.add_argument("--n-dask-workers", type=int, default=4, help="Number of Dask workers")
    args = parser.parse_args()

    navigator = Go2MqttNavigator(
        robot_ip=args.robot_ip,
        robot_id=args.robot_id,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
    )

    try:
        navigator.start()
        logger.info("Go2 MQTT Navigator is ready. Waiting for MQTT Commands...")
        
        # Keep running
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        navigator.stop()


if __name__ == "__main__":
    main()
