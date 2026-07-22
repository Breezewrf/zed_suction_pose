#!/usr/bin/env python3

"""Expose suction detections through the ecommerce HTTP API."""

import threading
from typing import Dict, List, Tuple

import rclpy
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3DArray

from zed_suction_pose.http_conversion import make_http_item


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
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 4444)

        self.http_host = str(self.get_parameter("http_host").value)
        self.http_port = int(self.get_parameter("http_port").value)
        self._items: Tuple[Dict[str, float], ...] = ()
        self._items_lock = threading.Lock()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        topic = str(self.get_parameter("detection_topic").value)
        self.create_subscription(Detection3DArray, topic, self._detections_callback, qos)
        self.get_logger().info(f"HTTP API source topic: {topic}")

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


def create_app(node: SuctionHttpNode) -> FastAPI:
    app = FastAPI(title="ZED Suction Pose API")

    @app.get("/cv/ecommerce/items", response_model=List[EcommerceItem])
    def get_items() -> List[Dict[str, float]]:
        return node.current_items()

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
        uvicorn.run(create_app(node), host=node.http_host, port=node.http_port, log_level="info")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
