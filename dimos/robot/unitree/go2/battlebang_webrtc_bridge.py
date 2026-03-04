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

from dataclasses import dataclass
import json
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.msgs.geometry_msgs import Twist
from dimos.protocol.pubsub.impl.rospubsub import RawROS, RawROSTopic
from dimos.protocol.pubsub.impl.rospubsub_conversion import derive_ros_type, ros_to_dimos
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class Config(ModuleConfig):
    """Configuration for battlebang ROS->DimOS bridge topics."""

    webrtc_req_topic: str = "/webrtc_req"
    cmd_vel_topic: str | None = None
    ros_node_name: str = "dimos_battlebang_bridge"


class BattlebangWebrtcBridge(Module):
    """Bridge legacy battlebang ROS topics into GO2Connection RPC calls.

    - `/webrtc_req` (go2_interfaces/WebRtcReq) -> `GO2Connection.publish_request(...)`
    - Optional `/cmd_vel_out` (geometry_msgs/Twist) -> `GO2Connection.move(...)`
    """

    default_config = Config
    config: Config

    rpc_calls: list[str] = [
        "GO2Connection.publish_request",
        "GO2Connection.move",
    ]

    _ros: RawROS | None = None
    _unsubs: list[Any]
    _publish_request_rpc: Any
    _move_rpc: Any

    @rpc
    def start(self) -> None:
        super().start()

        webrtc_req_type = self._resolve_webrtc_req_type()
        self._unsubs = []
        self._publish_request_rpc = None
        self._move_rpc = None

        subscribe_webrtc = webrtc_req_type is not None
        subscribe_cmd_vel = bool(self.config.cmd_vel_topic)

        if not subscribe_webrtc:
            logger.warning(
                "[BattlebangWebrtcBridge] go2_interfaces.msg.WebRtcReq not found; "
                "skipping webrtc request bridge"
            )

        if not subscribe_webrtc and not subscribe_cmd_vel:
            logger.warning(
                "[BattlebangWebrtcBridge] no active ROS subscriptions configured; bridge is idle"
            )
            return

        self._ros = RawROS(node_name=self.config.ros_node_name)
        self._ros.start()

        if subscribe_webrtc and webrtc_req_type is not None:
            self._publish_request_rpc = self.get_rpc_calls("GO2Connection.publish_request")
            self._unsubs.append(
                self._ros.subscribe(
                    RawROSTopic(self.config.webrtc_req_topic, webrtc_req_type),
                    self._on_webrtc_req_raw,
                )
            )
            logger.info(f"[BattlebangWebrtcBridge] subscribed to {self.config.webrtc_req_topic}")

        if subscribe_cmd_vel:
            self._move_rpc = self.get_rpc_calls("GO2Connection.move")
            ros_twist_type = derive_ros_type(Twist)
            self._unsubs.append(
                self._ros.subscribe(
                    RawROSTopic(self.config.cmd_vel_topic, ros_twist_type),
                    self._on_cmd_vel_raw,
                )
            )
            logger.info(f"[BattlebangWebrtcBridge] subscribed to {self.config.cmd_vel_topic}")

    @rpc
    def stop(self) -> None:
        for unsub in getattr(self, "_unsubs", []):
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

        if self._ros is not None:
            self._ros.stop()
            self._ros = None

        super().stop()

    @staticmethod
    def _resolve_webrtc_req_type() -> type[Any] | None:
        try:
            from go2_interfaces.msg import WebRtcReq  # type: ignore[import-not-found]

            return WebRtcReq
        except Exception:
            return None

    @staticmethod
    def _parse_parameter(raw_parameter: Any) -> Any:
        if raw_parameter is None:
            return None
        if not isinstance(raw_parameter, str):
            return raw_parameter

        parameter = raw_parameter.strip()
        if parameter == "":
            return None

        try:
            return json.loads(parameter)
        except Exception:
            return parameter

    def _on_webrtc_req_raw(self, msg: Any, _topic: RawROSTopic) -> None:
        if self._publish_request_rpc is None:
            return

        topic = str(getattr(msg, "topic", "")).strip()
        if not topic:
            logger.warning("[BattlebangWebrtcBridge] dropping webrtc request: empty topic")
            return

        api_id = int(getattr(msg, "api_id", -1))
        if api_id < 0:
            logger.warning("[BattlebangWebrtcBridge] dropping webrtc request: invalid api_id")
            return

        payload: dict[str, Any] = {"api_id": api_id}
        parameter = self._parse_parameter(getattr(msg, "parameter", None))
        if parameter is not None:
            payload["parameter"] = parameter

        try:
            self._publish_request_rpc(topic, payload)
        except Exception as e:
            logger.error(f"[BattlebangWebrtcBridge] publish_request failed: {e}")

    def _on_cmd_vel_raw(self, msg: Any, _topic: RawROSTopic) -> None:
        if self._move_rpc is None:
            return

        try:
            twist = ros_to_dimos(msg, Twist)
            self._move_rpc(twist)
        except Exception as e:
            logger.error(f"[BattlebangWebrtcBridge] move bridge failed: {e}")


battlebang_webrtc_bridge = BattlebangWebrtcBridge.blueprint

__all__ = ["BattlebangWebrtcBridge", "battlebang_webrtc_bridge"]
