"""Conversion helpers for the ecommerce HTTP response."""

import math
from typing import Dict, List, Protocol, Tuple

import cv2
import numpy as np

from .constants import ITEM_COLORS_BGR


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


def draw_items_panel(image: np.ndarray, items: List[Dict[str, float]]) -> np.ndarray:
    """Draw the /items payload within the image's top-left quadrant."""
    output = image.copy()
    height, width = output.shape[:2]
    if height <= 0 or width <= 1:
        return output

    lines = [(f"items: {len(items)}", (255, 255, 255))]
    for index, item in enumerate(items):
        color = ITEM_COLORS_BGR[index % len(ITEM_COLORS_BGR)]
        lines.extend(
            [
                (
                    f"[{index}] extent_x={item['extent_x']:.4f} "
                    f"extent_y={item['extent_y']:.4f} extent_z={item['extent_z']:.4f}",
                    color,
                ),
                (
                    f"    x={item['x']:.4f} y={item['y']:.4f} z={item['z']:.4f}",
                    color,
                ),
                (
                    f"    rx={item['rx']:.2f} ry={item['ry']:.2f} rz={item['rz']:.2f}",
                    color,
                ),
            ]
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    margin = max(4, min(12, width // 100, height // 100))
    panel_width = max(1, width // 2)
    available_width = max(1, panel_width - 2 * margin)
    available_height = max(1, height - 2 * margin)

    base_sizes = [cv2.getTextSize(text, font, 1.0, thickness)[0] for text, _ in lines]
    max_base_width = max(size[0] for size in base_sizes)
    max_base_height = max(size[1] for size in base_sizes)
    width_scale = available_width / max(1.0, float(max_base_width))
    height_scale = available_height / max(
        1.0,
        float(len(lines) * (max_base_height + 6)),
    )
    font_scale = max(0.01, min(0.52, width_scale, height_scale))

    line_height = max(1, int(math.floor((max_base_height + 6) * font_scale)))
    panel_height = min(height, 2 * margin + line_height * len(lines))
    panel_region = output[:panel_height, :panel_width]
    black = np.zeros_like(panel_region)
    output[:panel_height, :panel_width] = cv2.addWeighted(
        panel_region,
        0.35,
        black,
        0.65,
        0.0,
    )

    baseline_y = margin + max(1, int(math.ceil(max_base_height * font_scale)))
    for line_index, (text, color) in enumerate(lines):
        y = min(panel_height - 1, baseline_y + line_index * line_height)
        cv2.putText(
            output,
            text,
            (margin, y),
            font,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (margin, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return output
