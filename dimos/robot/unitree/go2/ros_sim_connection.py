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

from __future__ import annotations

import threading
from typing import Any

from reactivex.observable import Observable
from reactivex.subject import Subject

from dimos.core.resource import Resource
from dimos.msgs.geometry_msgs import PoseStamped, Transform, Twist, Vector3
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.protocol.pubsub.impl.rospubsub import RawROS, RawROSTopic
from dimos.protocol.pubsub.impl.rospubsub_conversion import (
    derive_ros_type,
    dimos_to_ros,
    ros_to_dimos,
)
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.reactive import backpressure


def _resolve_topic(topic: str | None, namespace: str, leaf: str) -> str:
    if topic is not None:
        return topic if topic.startswith("/") else f"/{topic}"

    ns = namespace.strip("/")
    if not ns:
        return f"/{leaf}"
    return f"/{ns}/{leaf}"


def _get_sensor_qos() -> Any:
    """Return ROS sensor-data QoS profile when rclpy is available."""
    try:
        from rclpy.qos import qos_profile_sensor_data  # type: ignore[import-not-found]

        return qos_profile_sensor_data
    except Exception:
        return None


def _get_cmd_qos() -> Any:
    """Return a reliable QoS profile for command/control topics."""
    try:
        from rclpy.qos import (  # type: ignore[import-not-found]
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        return QoSProfile(  # type: ignore[no-untyped-call]
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
    except Exception:
        return None


class ROSSimConnection(Resource):
    """ROS-topic-backed GO2 connection for simulators.

    Reads sensor topics from ROS 2 and publishes velocity commands back to ROS 2.
    """

    def __init__(
        self,
        ros_env_namespace: str = "env0",
        ros_robot_namespace: str = "robot0",
        lidar_topic: str | None = None,
        odom_topic: str | None = None,
        image_topic: str | None = None,
        cmd_vel_topic: str | None = None,
        node_name: str | None = None,
    ) -> None:
        self.lidar_topic = _resolve_topic(lidar_topic, ros_env_namespace, "point_cloud2")
        self.odom_topic = _resolve_topic(odom_topic, ros_env_namespace, "odom")
        self.image_topic = _resolve_topic(image_topic, ros_robot_namespace, "camera/image_raw")
        self.cmd_vel_topic = _resolve_topic(cmd_vel_topic, ros_robot_namespace, "cmd_vel_out")

        self._ros = RawROS(node_name=node_name, qos=_get_sensor_qos())
        self._sensor_qos = _get_sensor_qos()
        self._cmd_qos = _get_cmd_qos()
        self._started = False
        self._stopping = False
        self._unsubs: list[Any] = []

        # Resolve ROS message classes once.
        self._ros_pc_type = derive_ros_type(PointCloud2)
        self._ros_img_type = derive_ros_type(Image)
        self._ros_twist_type = derive_ros_type(Twist)
        # nav_msgs/Odometry ROS type (raw ROS type, not dimos.msgs.nav_msgs.Odometry).
        from nav_msgs.msg import Odometry as RosOdometry  # type: ignore[import-not-found]

        self._ros_odom_type = RosOdometry

        self._lidar_subject: Subject[PointCloud2] = Subject()
        self._odom_subject: Subject[PoseStamped] = Subject()
        self._video_subject: Subject[Image] = Subject()
        self._latest_odom: PoseStamped | None = None

        self._stop_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

    @simple_mcache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return backpressure(self._lidar_subject)

    @simple_mcache
    def odom_stream(self) -> Observable[PoseStamped]:
        return backpressure(self._odom_subject)

    @simple_mcache
    def video_stream(self) -> Observable[Image]:
        return backpressure(self._video_subject)

    def start(self) -> None:
        if self._started:
            return
        self._stopping = False

        self._ros.start()

        self._unsubs.append(
            self._ros.subscribe(
                RawROSTopic(self.lidar_topic, self._ros_pc_type, qos=self._sensor_qos),
                self._on_lidar_raw,
            )
        )
        self._unsubs.append(
            self._ros.subscribe(
                RawROSTopic(self.image_topic, self._ros_img_type, qos=self._sensor_qos),
                self._on_image_raw,
            )
        )
        self._unsubs.append(
            self._ros.subscribe(
                RawROSTopic(self.odom_topic, self._ros_odom_type, qos=self._sensor_qos),
                self._on_odom_raw,
            )
        )

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        self._started = False

        with self._timer_lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None

        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()

        self._ros.stop()
        self._stopping = False

    def _on_lidar_raw(self, msg: Any, _topic: RawROSTopic) -> None:
        try:
            pc = ros_to_dimos(msg, PointCloud2)
            source_frame = (pc.frame_id or "").strip().lstrip("/")
            # Unitree WebRTC feeds global-frame lidar; emulate that behavior for ROS sim input.
            if (
                self._latest_odom is not None
                and source_frame
                and source_frame not in {"world", "map", "odom"}
            ):
                odom = self._latest_odom
                pc = pc.transform(
                    Transform(
                        translation=odom.position,
                        rotation=odom.orientation,
                        frame_id="world",
                        child_frame_id=source_frame,
                        ts=pc.ts,
                    )
                )
                pc.frame_id = "world"

            self._lidar_subject.on_next(pc)
        except Exception:
            return

    def _on_image_raw(self, msg: Any, _topic: RawROSTopic) -> None:
        try:
            self._video_subject.on_next(ros_to_dimos(msg, Image))
        except Exception:
            return

    def _on_odom_raw(self, msg: Any, _topic: RawROSTopic) -> None:
        try:
            sec = msg.header.stamp.sec
            nsec = msg.header.stamp.nanosec
            ts = float(sec) + float(nsec) / 1_000_000_000.0
            odom = PoseStamped(
                ts=ts,
                # Normalize ROS odom frame to dimos world frame for downstream TF consistency.
                frame_id="world",
                position=Vector3(
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ),
                orientation=[
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ],
            )
            self._latest_odom = odom
            self._odom_subject.on_next(odom)
        except Exception:
            return

    @staticmethod
    def _zero_twist() -> Twist:
        return Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

    def _publish_stop(self) -> None:
        if not self._started or self._stopping:
            return
        try:
            self._ros.publish(
                RawROSTopic(self.cmd_vel_topic, self._ros_twist_type, qos=self._cmd_qos),
                dimos_to_ros(self._zero_twist(), self._ros_twist_type),
            )
        except Exception:
            return

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        if self._stopping:
            return False
        if not self._started:
            self.start()
        if not self._started:
            return False

        try:
            self._ros.publish(
                RawROSTopic(self.cmd_vel_topic, self._ros_twist_type, qos=self._cmd_qos),
                dimos_to_ros(twist, self._ros_twist_type),
            )
        except Exception:
            return False

        if duration > 0.0:
            with self._timer_lock:
                if self._stop_timer is not None:
                    self._stop_timer.cancel()
                self._stop_timer = threading.Timer(duration, self._publish_stop)
                self._stop_timer.daemon = True
                self._stop_timer.start()

        return True

    def standup(self) -> bool:
        return True

    def liedown(self) -> bool:
        self._publish_stop()
        return True

    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "topic": topic,
            "data": data,
            "detail": "publish_request is not available in ros simulator mode",
        }
