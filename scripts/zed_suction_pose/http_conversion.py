"""Conversion helpers for the ecommerce HTTP response."""

import math
from typing import Dict, Protocol, Tuple

import cv2
import numpy as np


Quaternion = Tuple[float, float, float, float]


class ImageMessage(Protocol):
    encoding: str
    height: int
    width: int
    step: int
    data: bytes


def _normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("orientation quaternion has zero length")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_to_euler_degrees(quaternion: Quaternion) -> Tuple[float, float, float]:
    """Return ROS-style roll, pitch, yaw angles in degrees."""
    x, y, z, w = _normalize_quaternion(quaternion)

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def make_http_item(
    position: Tuple[float, float, float],
    quaternion: Quaternion,
    extents: Tuple[float, float, float],
) -> Dict[str, float]:
    """Build one response item, keeping the longer surface edge on local X."""
    extent_x, extent_y, extent_z = (max(0.0, float(value)) for value in extents)
    quaternion = _normalize_quaternion(quaternion)

    # For a clearly rectangular surface, the default cluster-OBB pose estimator
    # already aligns local X with the longest edge. This is not a strict invariant:
    # nearly square surfaces (below the configured aspect-ratio threshold), sparse
    # or noisy point clouds, and orientation fallbacks may still produce Y > X.
    # Rotate the frame together with the dimensions so the HTTP contract always
    # keeps extent_x and the returned local X axis on the same, longest edge.
    if extent_y > extent_x:
        extent_x, extent_y = extent_y, extent_x
        half_turn = math.pi / 4.0
        local_quarter_turn = (0.0, 0.0, math.sin(half_turn), math.cos(half_turn))
        quaternion = _multiply_quaternions(quaternion, local_quarter_turn)

    rx, ry, rz = quaternion_to_euler_degrees(quaternion)
    x, y, z = (float(value) for value in position)
    values = (extent_x, extent_y, extent_z, x, y, z, rx, ry, rz)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("detection contains a non-finite value")

    return {
        "extent_x": extent_x,
        "extent_y": extent_y,
        "extent_z": extent_z,
        "x": x,
        "y": y,
        "z": z,
        "rx": rx,
        "ry": ry,
        "rz": rz,
    }


def image_message_to_bgr(message: ImageMessage) -> np.ndarray:
    """Decode the common 8-bit ROS Image encodings without cv_bridge."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_size = width * channels
    if height <= 0 or width <= 0 or step < row_size:
        raise ValueError("invalid image dimensions or row step")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_size = height * step
    if raw.size < required_size:
        raise ValueError("image data is shorter than height * step")
    rows = raw[:required_size].reshape(height, step)[:, :row_size]

    if channels == 1:
        return cv2.cvtColor(rows.reshape(height, width), cv2.COLOR_GRAY2BGR)

    image = rows.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(image)
