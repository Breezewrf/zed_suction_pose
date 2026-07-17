"""General helpers with no ROS node state."""

import os
from typing import List, Optional, Sequence

import numpy as np
from ament_index_python.packages import get_package_share_directory


def default_model_path() -> str:
    """Return the model shipped with this ROS package."""
    try:
        return os.path.join(
            get_package_share_directory("zed_suction_pose"),
            "checkpoints",
            "best.pt",
        )
    except Exception:
        return ""


def parse_classes(value: str) -> Optional[List[int]]:
    value = value.strip()
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def as_unit_vector(values: Sequence[float], fallback: np.ndarray) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        return fallback.copy()
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return fallback.copy()
    return vec / norm


def make_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def make_odd_bounded(value: float, min_value: int, max_value: int) -> int:
    max_value = max(3, int(max_value))
    if max_value % 2 == 0:
        max_value -= 1
    max_value = max(3, max_value)

    min_value = max(3, int(min_value))
    if min_value % 2 == 0:
        min_value += 1
    if min_value > max_value:
        min_value = max_value

    window = max(3, int(round(value)))
    if window % 2 == 0:
        window += 1
    return int(np.clip(window, min_value, max_value))


def parse_axis(value: str) -> int:
    text = str(value).strip().lower()
    axes = {
        "x": 0,
        "0": 0,
        "forward": 0,
        "y": 1,
        "1": 1,
        "z": 2,
        "2": 2,
        "optical_z": 2,
    }
    if text not in axes:
        raise ValueError(f"Invalid depth_axis '{value}'. Expected x, y, or z.")
    return axes[text]

