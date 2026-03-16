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
from threading import Event, Thread
import time
from typing import TYPE_CHECKING, Any, Protocol

from reactivex.disposable import Disposable
from reactivex.observable import Observable
import rerun.blueprint as rrb

from dimos import spec
from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module
from dimos.core.module_coordinator import ModuleCoordinator
from dimos.core.stream import In, Out
from dimos.core.transport import LCMTransport, pSHMTransport

if TYPE_CHECKING:
    from dimos.core.rpc_client import ModuleProxy
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
    def balance_stand(self) -> bool: ...
    def set_obstacle_avoidance(self, enabled: bool = True) -> None: ...
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


def make_connection(ip: str | None, cfg: GlobalConfig) -> Go2ConnectionProtocol:
    raw_connection = _make_connection(ip, cfg)
    
    # Wrap with AlignedGo2Connection if namespace or alignment is configured
    if cfg.robot_id or cfg.robot_frame_namespace or cfg.navigation_enable_initial_alignment:
        return AlignedGo2Connection(raw_connection, cfg)
    
    return raw_connection


def _make_connection(
    ip: str | None,
    cfg: GlobalConfig,
    *,
    ros_env_namespace: str | None = None,
    ros_robot_namespace: str | None = None,
    ros_pointcloud_topic: str | None = None,
    ros_odom_topic: str | None = None,
    ros_image_topic: str | None = None,
    ros_cmd_vel_topic: str | None = None,
) -> Go2ConnectionProtocol:
    connection_type = cfg.unitree_connection_type

    if ip in ("fake", "mock", "replay") or connection_type == "replay":
        dataset = cfg.replay_dir
        return ReplayConnection(dataset=dataset)
    if ip == "mujoco" or connection_type == "mujoco":
        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        return MujocoConnection(cfg)
    if ip in ("ros", "ros2") or connection_type == "ros":
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
        return ROSSimConnection(**ros_connection_kwargs)
    if connection_type == "webrtc-rs":
        try:
            from dimos.robot.unitree.connection_rs import UnitreeWebRTCRSConnection
        except ImportError as error:
            raise ImportError(
                "unitree-webrtc-rs backend selected, but the package is not installed. "
                "Install `dimos[unitree]` or `unitree-webrtc-rs`."
            ) from error

        assert ip is not None, "IP address must be provided"
        return UnitreeWebRTCRSConnection(ip)

    assert ip is not None, "IP address must be provided"
    return UnitreeWebRTCConnection(ip)


class AlignedGo2Connection:
    """Wrapper for Go2 connection that adds optional frame namespace and initial pose alignment.
    
    This wrapper:
    - Applies robot_id-based frame prefixing to robot-local frames (odom, base_link, camera_*)
    - Keeps global frames (map, world) unprefixed
    - Optionally anchors initial odom/lidar to a specified map pose
    """

    def __init__(
        self,
        raw_connection: Go2ConnectionProtocol,
        cfg: GlobalConfig,
    ) -> None:
        self._raw = raw_connection
        self._cfg = cfg
        
        # Compute frame namespace prefix
        if cfg.robot_frame_namespace:
            self._frame_prefix = cfg.robot_frame_namespace
        elif cfg.robot_id:
            self._frame_prefix = cfg.robot_id
        else:
            self._frame_prefix = None
        
        # Initial pose alignment state
        self._computed_init_tf: Transform | None = None
        self._last_map_pose: Transform | None = None
        
        if cfg.navigation_enable_initial_alignment:
            self._setup_initial_alignment()

    def _setup_initial_alignment(self) -> None:
        """Load initial pose from config or file."""
        cfg = self._cfg
        
        # Priority 1: Explicit x, y, yaw from config
        if cfg.navigation_initial_x is not None and cfg.navigation_initial_y is not None:
            yaw = cfg.navigation_initial_yaw or 0.0
            self._last_map_pose = Transform(
                translation=Vector3(cfg.navigation_initial_x, cfg.navigation_initial_y, 0.0),
                rotation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
                frame_id=cfg.navigation_alignment_frame,
                child_frame_id=self._robot_frame("base_link"),
            )
            return
        
        # Priority 2: Load from JSON file
        if cfg.navigation_initial_pose_path:
            import json
            from pathlib import Path
            
            pose_path = Path(cfg.navigation_initial_pose_path)
            if pose_path.exists():
                try:
                    with open(pose_path) as f:
                        data = json.load(f)
                    
                    orientation = data.get("orientation", {})
                    self._last_map_pose = Transform(
                        translation=Vector3(
                            float(data.get("x", 0.0)),
                            float(data.get("y", 0.0)),
                            float(data.get("z", 0.0)),
                        ),
                        rotation=Quaternion(
                            float(orientation.get("x", 0.0)),
                            float(orientation.get("y", 0.0)),
                            float(orientation.get("z", 0.0)),
                            float(orientation.get("w", 1.0)),
                        ),
                        frame_id=cfg.navigation_alignment_frame,
                        child_frame_id=self._robot_frame("base_link"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to load initial pose from {pose_path}: {e}")

    def _robot_frame(self, base_name: str) -> str:
        """Apply namespace prefix to robot-local frame names."""
        if self._frame_prefix and base_name not in ("map", "world"):
            return f"{self._frame_prefix}/{base_name}"
        return base_name

    def start(self) -> None:
        self._raw.start()

    def stop(self) -> None:
        self._raw.stop()

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        return self._raw.move(twist, duration)

    def standup(self) -> bool:
        return self._raw.standup()

    def liedown(self) -> bool:
        return self._raw.liedown()

    def balance_stand(self) -> bool:
        return self._raw.balance_stand()

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self._raw.set_obstacle_avoidance(enabled)

    def publish_request(self, topic: str, data: dict) -> dict:  # type: ignore[type-arg]
        return self._raw.publish_request(topic, data)

    @simple_mcache
    def lidar_stream(self) -> Observable:  # type: ignore[type-arg]
        def apply_alignment_and_frame(pointcloud: PointCloud2) -> PointCloud2:
            # Apply initial pose alignment if configured
            if self._computed_init_tf is not None:
                pointcloud = pointcloud.transform(self._computed_init_tf)
            
            # Apply frame namespace
            pointcloud.frame_id = self._robot_frame(pointcloud.frame_id)
            return pointcloud

        return self._raw.lidar_stream().pipe(ops.map(apply_alignment_and_frame))

    @simple_mcache
    def odom_stream(self) -> Observable:  # type: ignore[type-arg]
        def apply_alignment_and_frame(odom: PoseStamped) -> PoseStamped:
            # Compute initial alignment on first odom
            if self._cfg.navigation_enable_initial_alignment:
                if self._last_map_pose is not None and self._computed_init_tf is None:
                    odom_tf = Transform(
                        translation=odom.position,
                        rotation=odom.orientation,
                        frame_id="odom_start",
                        child_frame_id=self._robot_frame("base_link"),
                    )
                    self._computed_init_tf = self._last_map_pose + odom_tf.inverse()
                    logger.info(
                        f"Anchored odom to {self._cfg.navigation_alignment_frame}: "
                        f"x={self._computed_init_tf.translation.x:.3f}, "
                        f"y={self._computed_init_tf.translation.y:.3f}"
                    )
                
                # Apply alignment transform
                if self._computed_init_tf is not None:
                    odom_tf = Transform(
                        translation=odom.position,
                        rotation=odom.orientation,
                        frame_id="odom_start",
                        child_frame_id=self._robot_frame("base_link"),
                    )
                    combined = self._computed_init_tf + odom_tf
                    odom.position = combined.translation
                    odom.orientation = combined.rotation
                    odom.frame_id = self._cfg.navigation_alignment_frame
                    return odom
            
            # Just apply frame namespace without alignment
            odom.frame_id = self._robot_frame(odom.frame_id)
            return odom

        return self._raw.odom_stream().pipe(ops.map(apply_alignment_and_frame))

    @simple_mcache
    def video_stream(self) -> Observable:  # type: ignore[type-arg]
        def apply_frame(image: Image) -> Image:
            image.frame_id = self._robot_frame(image.frame_id)
            return image

        return self._raw.video_stream().pipe(ops.map(apply_frame))


class ReplayConnection(UnitreeWebRTCConnection):
    # we don't want UnitreeWebRTCConnection to init
    def __init__(  # type: ignore[no-untyped-def]
        self,
        dataset: str = "go2_sf_office",
        **kwargs,
    ) -> None:
        self.dir_name = dataset
        get_data(self.dir_name)
        self.replay_config = {
            "loop": kwargs.get("loop", True),
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

    def balance_stand(self) -> bool:
        return True

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        pass

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
            return Image.from_numpy(
                arr, format=ImageFormat.RGB, frame_id="camera_optical"
            )

        video_store = TimedSensorReplay(f"{self.dir_name}/video", autocast=_autocast_video)  # type: ignore[var-annotated]
        return video_store.stream(**self.replay_config)  # type: ignore[arg-type]

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        return True

    def publish_request(self, topic: str, data: dict):  # type: ignore[no-untyped-def, type-arg]
        """Fake publish request for testing."""
        return {"status": "ok", "message": "Fake publish"}


class GO2Connection(Module, spec.Camera, spec.Pointcloud):
    cmd_vel: In[Twist]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]

    connection: Go2ConnectionProtocol
    camera_info_static: CameraInfo
    _global_config: GlobalConfig
    _camera_info_thread: Thread | None = None
    _camera_info_stop: Event
    _latest_video_frame: Image | None = None
    _frame_prefix: str | None = None

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
        
        # Compute frame namespace prefix
        if cfg.robot_frame_namespace:
            self._frame_prefix = cfg.robot_frame_namespace
        elif cfg.robot_id:
            self._frame_prefix = cfg.robot_id
        else:
            self._frame_prefix = None

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
        self.connection = _make_connection(
            ip,
            self._global_config,
            ros_env_namespace=ros_env_namespace,
            ros_robot_namespace=ros_robot_namespace,
            ros_pointcloud_topic=ros_pointcloud_topic,
            ros_odom_topic=ros_odom_topic,
            ros_image_topic=ros_image_topic,
            ros_cmd_vel_topic=ros_cmd_vel_topic,
        )
        self._camera_info_stop = Event()
        
        # Create camera_info with appropriate frame_id
        self.camera_info_static = _camera_info_static()
        self.camera_info_static.frame_id = self._robot_frame("camera_optical")

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

        self._disposables.add(
            self.connection.lidar_stream().subscribe(self.lidar.publish)
        )
        self._disposables.add(self.connection.odom_stream().subscribe(self._publish_tf))
        self._disposables.add(self.connection.video_stream().subscribe(onimage))
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self.move)))

        self._camera_info_stop.clear()
        self._camera_info_thread = Thread(
            target=self.publish_camera_info,
            daemon=True,
        )
        self._camera_info_thread.start()

        self.standup()
        time.sleep(3)
        self.connection.balance_stand()
        self.connection.set_obstacle_avoidance(self._global_config.obstacle_avoidance)

        # self.record("go2_bigoffice")

    @rpc
    def stop(self) -> None:
        self.liedown()
        self._camera_info_stop.set()

        if self.connection:
            self.connection.stop()

        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=1.0)

        super().stop()

    def _robot_frame(self, base_name: str) -> str:
        """Apply namespace prefix to robot-local frame names."""
        if self._frame_prefix and base_name not in ("map", "world"):
            return f"{self._frame_prefix}/{base_name}"
        return base_name

    @classmethod
    def _odom_to_tf(self, odom: PoseStamped) -> list[Transform]:
        camera_link = Transform(
            translation=Vector3(0.3, 0.0, 0.0),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id=self._robot_frame("base_link"),
            child_frame_id=self._robot_frame("camera_link"),
            ts=odom.ts,
        )

        camera_optical = Transform(
            translation=Vector3(0.0, 0.0, 0.0),
            rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
            frame_id=self._robot_frame("camera_link"),
            child_frame_id=self._robot_frame("camera_optical"),
            ts=odom.ts,
        )

        return [
            Transform.from_pose(self._robot_frame("base_link"), odom),
            camera_link,
            camera_optical,
        ]

    def _publish_tf(self, msg: PoseStamped) -> None:
        transforms = self._odom_to_tf(msg)
        self.tf.publish(*transforms)
        if self.odom.transport:
            self.odom.publish(msg)

    def publish_camera_info(self) -> None:
        while not self._camera_info_stop.is_set():
            self.camera_info.publish(self.camera_info_static)
            self._camera_info_stop.wait(1.0)

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


def deploy(dimos: ModuleCoordinator, ip: str, prefix: str = "") -> "ModuleProxy":
    from dimos.constants import (
        DEFAULT_CAPACITY_COLOR_IMAGE,
        DEFAULT_CAPACITY_POINTCLOUD,
    )

    connection = dimos.deploy(GO2Connection, ip)  # type: ignore[attr-defined]

    connection.lidar.transport = pSHMTransport(
        f"{prefix}/lidar", default_capacity=DEFAULT_CAPACITY_POINTCLOUD
    )
    connection.color_image.transport = pSHMTransport(
        f"{prefix}/image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )

    connection.cmd_vel.transport = LCMTransport(f"{prefix}/cmd_vel", Twist)

    connection.camera_info.transport = LCMTransport(f"{prefix}/camera_info", CameraInfo)
    connection.start()

    return connection


__all__ = ["GO2Connection", "deploy", "go2_connection", "make_connection"]
