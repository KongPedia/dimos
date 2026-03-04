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

import logging
from threading import Lock, Thread
import time
from typing import Any, Protocol

from reactivex.disposable import Disposable
from reactivex.observable import Observable
import rerun.blueprint as rrb

from dimos import spec
from dimos.agents.annotation import skill
from dimos.core import DimosCluster, In, LCMTransport, Module, Out, pSHMTransport, rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.msgs.geometry_msgs import (
    PoseStamped,
    Quaternion,
    Transform,
    Twist,
    Vector3,
)
from dimos.msgs.sensor_msgs import CameraInfo, Image, PointCloud2
from dimos.msgs.sensor_msgs.Image import ImageFormat
from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from dimos.robot.unitree.go2.ros_sim_connection import ROSSimConnection
from dimos.utils.data import get_data
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.testing.replay import TimedSensorReplay, TimedSensorStorage

logger = logging.getLogger(__name__)


class Go2ConnectionProtocol(Protocol):
    """Protocol defining the interface for Go2 robot connections."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def lidar_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def odom_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def video_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def move(self, twist: Twist, duration: float = 0.0) -> bool: ...
    def standup(self) -> bool: ...
    def liedown(self) -> bool: ...
    def publish_request(self, topic: str, data: dict) -> dict: ...  # type: ignore[type-arg]


def _camera_info_static() -> CameraInfo:
    fx, fy, cx, cy = (819.553492, 820.646595, 625.284099, 336.808987)
    width, height = (1280, 720)

    return CameraInfo(
        frame_id="camera_optical",
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
    )


class ReplayConnection(UnitreeWebRTCConnection):
    dir_name = "unitree_go2_bigoffice"

    # we don't want UnitreeWebRTCConnection to init
    def __init__(  # type: ignore[no-untyped-def]
        self,
        **kwargs,
    ) -> None:
        get_data(self.dir_name)
        self.replay_config = {
            "loop": kwargs.get("loop"),
            "seek": kwargs.get("seek"),
            "duration": kwargs.get("duration"),
        }

    def connect(self) -> None:
        pass

    def start(self) -> None:
        pass

    def standup(self) -> bool:
        return True

    def liedown(self) -> bool:
        return True

    @simple_mcache
    def lidar_stream(self):  # type: ignore[no-untyped-def]
        lidar_store = TimedSensorReplay(f"{self.dir_name}/lidar")  # type: ignore[var-annotated]
        return lidar_store.stream(**self.replay_config)  # type: ignore[arg-type]

    @simple_mcache
    def odom_stream(self):  # type: ignore[no-untyped-def]
        odom_store = TimedSensorReplay(f"{self.dir_name}/odom")  # type: ignore[var-annotated]
        return odom_store.stream(**self.replay_config)  # type: ignore[arg-type]

    # we don't have raw video stream in the data set
    @simple_mcache
    def video_stream(self):  # type: ignore[no-untyped-def]
        # Legacy Unitree recordings can have RGB bytes that were tagged/assumed as BGR.
        # Fix at replay-time by coercing everything to RGB before publishing/logging.
        def _autocast_video(x):  # type: ignore[no-untyped-def]
            # If the old recording tagged it as BGR, relabel to RGB (do NOT channel-swap again).
            if isinstance(x, Image):
                if x.format == ImageFormat.BGR:
                    x.format = ImageFormat.RGB
                if not x.frame_id:
                    x.frame_id = "camera_optical"
                return x

            # Some recordings may store raw arrays or frame wrappers.
            arr = x.to_ndarray(format="rgb24") if hasattr(x, "to_ndarray") else x
            return Image.from_numpy(arr, format=ImageFormat.RGB, frame_id="camera_optical")

        video_store = TimedSensorReplay(f"{self.dir_name}/video", autocast=_autocast_video)  # type: ignore[var-annotated]
        return video_store.stream(**self.replay_config)  # type: ignore[arg-type]

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        return True

    def publish_request(self, topic: str, data: dict):  # type: ignore[no-untyped-def, type-arg]
        """Fake publish request for testing."""
        return {"status": "ok", "message": "Fake publish"}


class GO2Connection(Module, spec.Camera, spec.Pointcloud):
    cmd_vel: In[Twist]
    # [BattleBang local] Viewer set_pose → reset_origin, avoids pickle over RPC
    origin_request: In[PoseStamped]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    connection: Go2ConnectionProtocol
    camera_info_static: CameraInfo = _camera_info_static()
    _global_config: GlobalConfig
    _camera_info_thread: Thread | None = None
    _latest_video_frame: Image | None = None

    # [BattleBang local] Origin offset for map-frame alignment
    # Set via reset_origin() to shift all published odom by a fixed amount
    _origin_offset: PoseStamped | None = None  # subtract from raw odom
    _pending_reset: bool = False               # capture next odom as origin
    _origin_lock: Lock

    @classmethod
    def rerun_views(cls):  # type: ignore[no-untyped-def]
        """Return Rerun view blueprints for GO2 camera visualization."""
        return [
            rrb.Spatial2DView(
                name="Camera",
                origin="world/robot/camera/rgb",
            ),
        ]

    def __init__(  # type: ignore[no-untyped-def]
        self,
        ip: str | None = None,
        cfg: GlobalConfig = global_config,
        *args,
        **kwargs,
    ) -> None:
        self._global_config = cfg
        self._origin_offset = None
        self._pending_reset = False
        self._origin_lock = Lock()

        ros_env_namespace = kwargs.pop("ros_env_namespace", None)
        ros_robot_namespace = kwargs.pop("ros_robot_namespace", None)
        ros_pointcloud_topic = kwargs.pop("ros_pointcloud_topic", None)
        if ros_pointcloud_topic is None:
            # Backward compatibility with older keyword argument.
            ros_pointcloud_topic = kwargs.pop("ros_lidar_topic", None)
        ros_odom_topic = kwargs.pop("ros_odom_topic", None)
        ros_image_topic = kwargs.pop("ros_image_topic", None)
        ros_cmd_vel_topic = kwargs.pop("ros_cmd_vel_topic", None)

        ip = ip if ip is not None else self._global_config.robot_ip

        connection_type = self._global_config.unitree_connection_type

        if ip in ["fake", "mock", "replay"] or connection_type == "replay":
            self.connection = ReplayConnection()
        elif ip == "mujoco" or connection_type == "mujoco":
            from dimos.robot.unitree.mujoco_connection import MujocoConnection

            self.connection = MujocoConnection(self._global_config)
        elif ip in ["ros", "ros2"] or connection_type == "ros":
            ros_connection_kwargs: dict[str, str] = {}
            if ros_env_namespace is not None:
                ros_connection_kwargs["ros_env_namespace"] = ros_env_namespace
            if ros_robot_namespace is not None:
                ros_connection_kwargs["ros_robot_namespace"] = ros_robot_namespace
            if ros_pointcloud_topic is not None:
                ros_connection_kwargs["lidar_topic"] = ros_pointcloud_topic
            if ros_odom_topic is not None:
                ros_connection_kwargs["odom_topic"] = ros_odom_topic
            if ros_image_topic is not None:
                ros_connection_kwargs["image_topic"] = ros_image_topic
            if ros_cmd_vel_topic is not None:
                ros_connection_kwargs["cmd_vel_topic"] = ros_cmd_vel_topic
            self.connection = ROSSimConnection(**ros_connection_kwargs)
        else:
            assert ip is not None, "IP address must be provided"
            self.connection = UnitreeWebRTCConnection(ip)

        Module.__init__(self, *args, **kwargs) 

    @rpc
    def record(self, recording_name: str) -> None:
        lidar_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/lidar")  # type: ignore[type-arg]
        lidar_store.consume_stream(self.connection.lidar_stream())

        odom_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/odom")  # type: ignore[type-arg]
        odom_store.consume_stream(self.connection.odom_stream())

        video_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/video")  # type: ignore[type-arg]
        video_store.consume_stream(self.connection.video_stream())

    @rpc
    def start(self) -> None:
        super().start()

        self.connection.start()

        def onimage(image: Image) -> None:
            self.color_image.publish(image)
            self._latest_video_frame = image

        self._disposables.add(self.connection.lidar_stream().subscribe(self.lidar.publish))
        self._disposables.add(self.connection.odom_stream().subscribe(self._publish_tf))
        self._disposables.add(self.connection.video_stream().subscribe(onimage))
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self.move)))
        # [BattleBang local] Viewer 📍 Set Pose → reset_origin via LCM (avoids pickle)
        self._disposables.add(Disposable(self.origin_request.subscribe(self._on_origin_request)))

        self._camera_info_thread = Thread(
            target=self.publish_camera_info,
            daemon=True,
        )
        self._camera_info_thread.start()

        self.standup()
        # self.record("go2_bigoffice")

    @rpc
    def stop(self) -> None:
        self.liedown()

        if self.connection:
            self.connection.stop()

        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=1.0)

        super().stop()

    @classmethod
    def _odom_to_tf(cls, odom: PoseStamped) -> list[Transform]:
        camera_link = Transform(
            translation=Vector3(0.3, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
            child_frame_id="camera_link",
            ts=odom.ts,
        )

        camera_optical = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
            frame_id="camera_link",
            child_frame_id="camera_optical",
            ts=odom.ts,
        )

        return [
            Transform.from_pose("base_link", odom),
            camera_link,
            camera_optical,
        ]

    def _apply_origin_offset(self, msg: PoseStamped) -> PoseStamped:
        """[BattleBang local] Apply the origin offset to a raw odom pose.

        If _pending_reset is True, the current msg becomes the new origin (0, 0).
        If _origin_offset is set, subtract it so the published pose is relative.
        """
        with self._origin_lock:
            if self._pending_reset:
                self._origin_offset = msg
                self._pending_reset = False
                logger.info(
                    f"[origin] Reset: raw odom ({msg.x:.3f}, {msg.y:.3f}) is now (0, 0)"
                )

            if self._origin_offset is None:
                return msg

            # Compute relative pose in the *origin* robot frame
            # delta pos = global pos - origin pos, rotated into world
            delta = msg - self._origin_offset  # uses Pose.__sub__
            adjusted = PoseStamped(
                frame_id=msg.frame_id,
                ts=msg.ts,
                position=delta.position,
                orientation=delta.orientation,
            )
            return adjusted

    @rpc
    def reset_origin(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw_deg: float | None = None,
    ) -> None:
        """[BattleBang local] Set the map-frame origin for odom publishing.

        Call with no args to mark the *next* odom reading as (0, 0).
        Call with x/y/yaw_deg to set a specific map-frame offset:
          - The robot's current raw odom position will be reported as (x, y, yaw)
          - i.e., origin_offset = raw_odom - (x, y, yaw)
        
        Args:
            x: Desired map-frame X position of current robot location (m)
            y: Desired map-frame Y position of current robot location (m)
            yaw_deg: Desired map-frame yaw of current robot location (deg)
        """
        with self._origin_lock:
            if x is None and y is None:
                # Capture next odom as the new origin
                self._pending_reset = True
                self._origin_offset = None
                logger.info("[origin] Will reset on next odom message")
            else:
                # Will be resolved on next odom tick: store a "virtual" offset
                # We store sentinel values; actual subtraction happens in _apply_origin_offset
                # when the next odom arrives.
                # For now just store desired map-frame position as the pending "where am I" marker.
                self._pending_reset = False
                # Store desired map position; resolved against raw odom on next tick
                self._desired_map_pose = PoseStamped(
                    frame_id="world",
                    position=(x or 0.0, y or 0.0, 0.0),
                    orientation=Quaternion.from_euler(Vector3(0.0, 0.0, float(yaw_deg or 0.0) * 3.14159265 / 180.0)),
                )
                self._pending_map_pose = True
                logger.info(f"[origin] Will set map pose to ({x}, {y}, yaw={yaw_deg}deg) on next odom")

    def _on_origin_request(self, msg: PoseStamped) -> None:
        """[BattleBang local] LCM handler for viewer 📍 Set Pose → reset_origin.

        WebsocketVisModule publishes a PoseStamped to origin_request when the user
        clicks 'Set Pose' in the browser. This avoids RPC+pickle issues.
        """
        import math as _math
        x = float(msg.x)
        y = float(msg.y)
        # Extract yaw from quaternion
        q = msg.orientation
        yaw_rad = _math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        yaw_deg = _math.degrees(yaw_rad)
        logger.info(f"[origin] origin_request received: ({x:.3f}, {y:.3f}, yaw={yaw_deg:.1f}deg)")
        self.reset_origin(x=x, y=y, yaw_deg=yaw_deg)


    def _publish_tf(self, msg: PoseStamped) -> None:
        # [BattleBang local] Resolve pending map-pose-based offset
        with self._origin_lock:
            if getattr(self, "_pending_map_pose", False):
                desired = self._desired_map_pose
                # origin_offset = raw_odom - desired_map_pose
                # so that: raw_odom - origin_offset = desired_map_pose
                offset_pos = msg.position - desired.position
                offset_ori = msg.orientation * desired.orientation.inverse()
                self._origin_offset = PoseStamped(
                    frame_id="world",
                    position=offset_pos,
                    orientation=offset_ori,
                )
                self._pending_map_pose = False
                logger.info(
                    f"[origin] Map pose set: raw ({msg.x:.3f}, {msg.y:.3f}) "
                    f"→ map ({desired.x:.3f}, {desired.y:.3f})"
                )

        adjusted = self._apply_origin_offset(msg)
        # [BattleBang local] Keep latest map-frame pose for MQTT pose broadcaster
        self._latest_adjusted_odom = adjusted
        transforms = self._odom_to_tf(adjusted)
        self.tf.publish(*transforms)
        if self.odom.transport:
            self.odom.publish(adjusted)

    def publish_camera_info(self) -> None:
        while True:
            self.camera_info.publish(_camera_info_static())
            time.sleep(1.0)

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send movement command to robot."""
        return self.connection.move(twist, duration)

    @rpc
    def standup(self) -> bool:
        """Make the robot stand up."""
        return self.connection.standup()

    @rpc
    def liedown(self) -> bool:
        """Make the robot lie down."""
        return self.connection.liedown()

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        """Publish a request to the WebRTC connection.
        Args:
            topic: The RTC topic to publish to
            data: The data dictionary to publish
        Returns:
            The result of the publish request
        """
        return self.connection.publish_request(topic, data)

    @skill
    def observe(self) -> Image | None:
        """Returns the latest video frame from the robot camera. Use this skill for any visual world queries.

        This skill provides the current camera view for perception tasks.
        Returns None if no frame has been captured yet.
        """
        return self._latest_video_frame


go2_connection = GO2Connection.blueprint


def deploy(dimos: DimosCluster, ip: str, prefix: str = "") -> GO2Connection:
    from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE

    connection = dimos.deploy(GO2Connection, ip)  # type: ignore[attr-defined]

    connection.pointcloud.transport = pSHMTransport(
        f"{prefix}/lidar", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )
    connection.color_image.transport = pSHMTransport(
        f"{prefix}/image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )

    connection.cmd_vel.transport = LCMTransport(f"{prefix}/cmd_vel", Twist)

    connection.camera_info.transport = LCMTransport(f"{prefix}/camera_info", CameraInfo)
    connection.start()

    return connection  # type: ignore[no-any-return]


__all__ = ["GO2Connection", "deploy", "go2_connection"]
