"""Conversion helpers for the ecommerce HTTP response."""

import math
from typing import Dict, Tuple


Quaternion = Tuple[float, float, float, float]


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
