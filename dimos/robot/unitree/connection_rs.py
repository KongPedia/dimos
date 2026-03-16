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

import asyncio
import functools
import threading
import time
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.observable import Observable

from unitree_webrtc_rs import (
    RTC_TOPIC,
    SPORT_CMD,
    VUI_COLOR,
    UnitreeWebRTCConnection as RustConnection,
    WebRTCConnectionMethod,
)

from dimos.core.resource import Resource
from dimos.msgs.geometry_msgs import Pose, Transform, Twist
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.msgs.sensor_msgs.Image import ImageFormat
from dimos.robot.unitree.type.lidar import RawLidarMsg, pointcloud2_from_webrtc_lidar
from dimos.robot.unitree.type.lowstate import LowStateMsg
from dimos.robot.unitree.type.odometry import Odometry
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.reactive import backpressure, callback_to_observable

VideoMessage: TypeAlias = NDArray[np.uint8]


class UnitreeWebRTCRSConnection(Resource):
    def __init__(self, ip: str, mode: str = "ai") -> None:
        self.ip = ip
        self.mode = mode
        self.cmd_vel_timeout = 0.2
        self.stop_timer: threading.Timer | None = None
        self.conn = RustConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
        self._connect_error: BaseException | None = None
        self._noop_video_callback = lambda _frame: None
        self.connect()

    def _run_coro(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def _run_async_call(self, fn: Callable[[], Any]) -> Any:
        async def runner() -> Any:
            return await fn()

        future = asyncio.run_coroutine_threadsafe(runner(), self.loop)
        return future.result()

    def _run_sync_in_loop(self, fn) -> None:  # type: ignore[no-untyped-def]
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(fn)

    def connect(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.connection_ready = threading.Event()

        async def async_connect() -> None:
            try:
                await self.conn.connect()
                await self.conn.datachannel.disableTrafficSaving(True)
                self.conn.datachannel.set_decoder(decoder_type="native")
                await self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"],
                    {"api_id": 1002, "parameter": {"name": self.mode}},
                )
            except BaseException as error:
                self._connect_error = error
            finally:
                self.connection_ready.set()

        def start_background_loop() -> None:
            asyncio.set_event_loop(self.loop)
            self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.thread = threading.Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        self.connection_ready.wait()

        if self._connect_error is not None:
            raise RuntimeError(f"Failed to connect to Unitree via webrtc-rs: {self._connect_error}")

    def start(self) -> None:
        needs_reconnect = (
            not hasattr(self, "loop")
            or not hasattr(self, "thread")
            or self.loop.is_closed()
            or
            not self.thread.is_alive()
        )
        if needs_reconnect:
            self.connect()

    def _publish_wireless_controller(self, *, lx: float, ly: float, rx: float, ry: float = 0.0) -> None:
        self.conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["WIRELESS_CONTROLLER"],
            data={"lx": lx, "ly": ly, "rx": rx, "ry": ry},
        )

    def _stop_motion(self) -> None:
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        async def async_stop() -> None:
            self._publish_wireless_controller(lx=0.0, ly=0.0, rx=0.0, ry=0.0)

        if self.loop.is_running():
            try:
                self._run_coro(async_stop())
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_motion()

        async def async_disconnect() -> None:
            try:
                self.conn.video.switchVideoChannel(False)
            except Exception:
                pass
            try:
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["ULIDAR_SWITCH"], "off"
                )
            except Exception:
                pass
            try:
                await self.conn.disconnect()
            except Exception:
                pass

        if self.loop.is_running():
            try:
                self._run_coro(async_disconnect())
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        async def async_move() -> None:
            self._publish_wireless_controller(lx=-y, ly=x, rx=-yaw, ry=0.0)

        async def async_move_duration() -> None:
            start_time = time.time()
            while time.time() - start_time < duration:
                self._publish_wireless_controller(lx=-y, ly=x, rx=-yaw, ry=0.0)
                await asyncio.sleep(0.01)

        if self.stop_timer:
            self.stop_timer.cancel()

        self.stop_timer = threading.Timer(self.cmd_vel_timeout, self._stop_motion)
        self.stop_timer.daemon = True
        self.stop_timer.start()

        try:
            if duration > 0:
                self._run_coro(async_move_duration())
                self._stop_motion()
            else:
                self._run_coro(async_move())
            return True
        except Exception:
            return False

    def unitree_sub_stream(
        self,
        topic_name: str,
        *,
        on_subscribe=None,  # type: ignore[no-untyped-def]
        on_unsubscribe=None,  # type: ignore[no-untyped-def]
    ):
        def subscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            def run_subscription() -> None:
                if on_subscribe is not None:
                    on_subscribe()
                self.conn.datachannel.pub_sub.subscribe(topic_name, cb)

            self._run_sync_in_loop(run_subscription)

        def unsubscribe_in_thread(_cb) -> None:  # type: ignore[no-untyped-def]
            def run_unsubscription() -> None:
                try:
                    self.conn.datachannel.pub_sub.unsubscribe(topic_name)
                finally:
                    if on_unsubscribe is not None:
                        on_unsubscribe()

            self._run_sync_in_loop(run_unsubscription)

        return callback_to_observable(start=subscribe_in_thread, stop=unsubscribe_in_thread)

    # Generic sync API call (we jump into the client thread)
    def publish_request(self, topic: str, data: dict[Any, Any]) -> Any:
        return self._run_async_call(
            lambda: self.conn.datachannel.pub_sub.publish_request_new(topic, data)
        )

    @simple_mcache
    def raw_lidar_stream(self) -> Observable[RawLidarMsg]:
        def enable_lidar() -> None:
            self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")

        def disable_lidar() -> None:
            self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "off")

        return backpressure(
            self.unitree_sub_stream(
                RTC_TOPIC["ULIDAR_ARRAY"],
                on_subscribe=enable_lidar,
                on_unsubscribe=disable_lidar,
            )
        )

    @simple_mcache
    def raw_odom_stream(self) -> Observable[Pose]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["ROBOTODOM"]))

    @simple_mcache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return backpressure(self.raw_lidar_stream().pipe(ops.map(pointcloud2_from_webrtc_lidar)))

    @simple_mcache
    def tf_stream(self) -> Observable[Transform]:
        base_link = functools.partial(Transform.from_pose, "base_link")
        return backpressure(self.odom_stream().pipe(ops.map(base_link)))

    @simple_mcache
    def odom_stream(self) -> Observable[Pose]:
        return backpressure(self.raw_odom_stream().pipe(ops.map(Odometry.from_msg)))

    @simple_mcache
    def raw_video_stream(self) -> Observable[VideoMessage]:
        def start_video(cb) -> None:  # type: ignore[no-untyped-def]
            def run_start() -> None:
                self.conn.video.on_frame(cb)
                self.conn.video.switchVideoChannel(True)

            self._run_sync_in_loop(run_start)

        def stop_video(_cb) -> None:  # type: ignore[no-untyped-def]
            def run_stop() -> None:
                self.conn.video.switchVideoChannel(False)
                self.conn.video.on_frame(self._noop_video_callback)

            self._run_sync_in_loop(run_stop)

        return backpressure(callback_to_observable(start=start_video, stop=stop_video))

    @simple_mcache
    def video_stream(self) -> Observable[Image]:
        return backpressure(
            self.raw_video_stream().pipe(
                ops.filter(lambda frame: frame is not None),
                ops.map(
                    lambda frame: Image.from_numpy(
                        np.ascontiguousarray(frame),
                        format=ImageFormat.BGR,
                        frame_id="camera_optical",
                    )
                ),
            )
        )

    @simple_mcache
    def lowstate_stream(self) -> Observable[LowStateMsg]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["LOW_STATE"]))

    def standup(self) -> bool:
        return bool(self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}))

    def balance_stand(self) -> bool:
        return bool(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        )

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self.publish_request(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"enable": int(enabled)}},
        )

    def liedown(self) -> bool:
        return bool(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]})
        )

    def color(self, color: VUI_COLOR = VUI_COLOR.RED, colortime: int = 60) -> bool:
        return self.publish_request(
            RTC_TOPIC["VUI"],
            {
                "api_id": 1001,
                "parameter": {
                    "color": color,
                    "time": colortime,
                },
            },
        )

    def get_video_stream(self, fps: int = 30) -> Observable[Image]:
        return self.video_stream()


__all__ = ["UnitreeWebRTCRSConnection", "VideoMessage"]
