#!/usr/bin/env python3

"""Expose suction detections through the ecommerce HTTP API."""

import threading
from typing import Dict, Iterator, List, Tuple

import cv2
import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray

from zed_suction_pose.http_conversion import image_message_to_bgr, make_http_item


class EcommerceItem(BaseModel):
    extent_x: float
    extent_y: float
    extent_z: float
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


class SuctionHttpNode(Node):
    def __init__(self) -> None:
        super().__init__("zed_suction_http_api")
        self.declare_parameter("detection_topic", "/suction_detections")
        self.declare_parameter("overlay_topic", "/suction_debug/overlay")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 4444)
        self.declare_parameter("jpeg_quality", 85)

        self.http_host = str(self.get_parameter("http_host").value)
        self.http_port = int(self.get_parameter("http_port").value)
        self._items: Tuple[Dict[str, float], ...] = ()
        self._items_lock = threading.Lock()
        self._image_condition = threading.Condition()
        self._latest_jpeg = b""
        self._image_sequence = 0
        self._jpeg_quality = max(1, min(100, int(self.get_parameter("jpeg_quality").value)))

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        detection_topic = str(self.get_parameter("detection_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.create_subscription(
            Detection3DArray,
            detection_topic,
            self._detections_callback,
            qos,
        )
        self.create_subscription(Image, overlay_topic, self._overlay_callback, qos)
        self.get_logger().info(f"HTTP API detection topic: {detection_topic}")
        self.get_logger().info(f"HTTP visualization topic: {overlay_topic}")

    def _detections_callback(self, message: Detection3DArray) -> None:
        ranked_items: List[Tuple[float, Dict[str, float]]] = []
        for detection in message.detections:
            pose = detection.bbox.center
            size = detection.bbox.size
            try:
                item = make_http_item(
                    (pose.position.x, pose.position.y, pose.position.z),
                    (
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ),
                    (size.x, size.y, size.z),
                )
            except ValueError as exc:
                self.get_logger().warning(f"Ignoring invalid suction detection: {exc}")
                continue

            score = max(
                (float(result.hypothesis.score) for result in detection.results),
                default=0.0,
            )
            ranked_items.append((score, item))

        ranked_items.sort(key=lambda entry: entry[0], reverse=True)
        with self._items_lock:
            self._items = tuple(item for _, item in ranked_items)

    def current_items(self) -> List[Dict[str, float]]:
        with self._items_lock:
            return [dict(item) for item in self._items]

    def _overlay_callback(self, message: Image) -> None:
        try:
            image = image_message_to_bgr(message)
            success, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not success:
                raise ValueError("OpenCV could not encode the overlay")
        except (ValueError, cv2.error) as exc:
            self.get_logger().warning(f"Ignoring invalid overlay image: {exc}")
            return

        with self._image_condition:
            self._latest_jpeg = encoded.tobytes()
            self._image_sequence += 1
            self._image_condition.notify_all()

    def mjpeg_frames(self) -> Iterator[bytes]:
        sequence = -1
        while rclpy.ok():
            with self._image_condition:
                self._image_condition.wait_for(
                    lambda: (
                        bool(self._latest_jpeg) and self._image_sequence != sequence
                    )
                    or not rclpy.ok(),
                    timeout=1.0,
                )
                if not rclpy.ok():
                    return
                if self._image_sequence == sequence or not self._latest_jpeg:
                    continue
                sequence = self._image_sequence
                jpeg = self._latest_jpeg

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )


def create_app(node: SuctionHttpNode) -> FastAPI:
    app = FastAPI(title="ZED Suction Pose API")

    @app.get("/cv/ecommerce/items", response_model=List[EcommerceItem])
    def get_items() -> List[Dict[str, float]]:
        return node.current_items()

    @app.get("/cv/ecommerce/items-vis")
    def get_items_visualization() -> StreamingResponse:
        return StreamingResponse(
            node.mjpeg_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SuctionHttpNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.get_logger().info(
            f"HTTP API listening on http://{node.http_host}:{node.http_port}/cv/ecommerce/items"
        )
        node.get_logger().info(
            "HTTP visualization listening on "
            f"http://{node.http_host}:{node.http_port}/cv/ecommerce/items-vis"
        )
        uvicorn.run(create_app(node), host=node.http_host, port=node.http_port, log_level="info")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
