#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import ROSTransport
from dimos.msgs.geometry_msgs import Twist
from dimos.msgs.sensor_msgs import CameraInfo, PointCloud2
from dimos.msgs.tf2_msgs import TFMessage
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.perception.detection.detectors.person.yolo import YoloPersonDetector
from dimos.perception.detection.module2D import Detection2DModule
from dimos.robot.unitree.go2.battlebang_webrtc_bridge import battlebang_webrtc_bridge
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule


def _robot_ns_topic(path: str) -> str:
    """Build a ROS topic under ros_robot_namespace (e.g. /robot0/point_cloud2)."""
    base = path if path.startswith("/") else f"/{path}"
    ns = (global_config.ros_robot_namespace or "").strip("/")
    if not ns:
        return base
    return f"/{ns}{base}"


# Battlebang compatibility blueprint:
# - listens to /webrtc_req and forwards requests to GO2Connection.publish_request
# - runs native DimOS YOLO person detection and publishes only detections to ROS
unitree_go2_battlebang_bridge = (
    autoconnect(
        unitree_go2,
        battlebang_webrtc_bridge(
            webrtc_req_topic=_robot_ns_topic("webrtc_req"),
        ),
        Detection2DModule.blueprint(
            detector=YoloPersonDetector,
            publish_annotations=False,
            publish_detection_images=False,
        ),
    )
    .remappings(
        [
            # Prevent direct in-process move commands to GO2Connection.
            # Route internal navigation/teleop cmd_vel to a dedicated stream for ROS twist_mux.
            (ReplanningAStarPlanner, "cmd_vel", "cmd_vel_dimos"),
            (WebsocketVisModule, "cmd_vel", "cmd_vel_dimos"),
            # Feed only twist_mux output into GO2Connection move input.
            (GO2Connection, "cmd_vel", "cmd_vel_out"),
        ]
    )
    .transports(
        {
            # Internal DimOS movement output -> battlebang twist_mux input.
            ("cmd_vel_dimos", Twist): ROSTransport(_robot_ns_topic("cmd_vel_dimos"), Twist),
            # battlebang twist_mux output -> DimOS GO2 move input.
            ("cmd_vel_out", Twist): ROSTransport(_robot_ns_topic("cmd_vel_out"), Twist),
            # Feed battlebang PerceptionNode with lightweight detections.
            ("detections", Detection2DArray): ROSTransport(
                _robot_ns_topic("detected_persons"), Detection2DArray
            ),
            # Feed battlebang PerceptionNode with point cloud and camera intrinsics.
            ("lidar", PointCloud2): ROSTransport(_robot_ns_topic("point_cloud2"), PointCloud2),
            ("camera_info", CameraInfo): ROSTransport(
                _robot_ns_topic("camera/camera_info"), CameraInfo
            ),
            # Publish DimOS TF into ROS tf2 buffer for cloud->camera transforms.
            ("tf_msg", TFMessage): ROSTransport(_robot_ns_topic("tf"), TFMessage),
        }
    )
)

__all__ = ["unitree_go2_battlebang_bridge"]
