"""Small data containers used across the suction pose pipeline."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class InstanceMask:
    index: int
    class_id: int
    class_name: str
    yolo_score: float
    mask: np.ndarray


@dataclass
class SuctionPose:
    object_id: int
    cluster_id: int
    class_id: int
    class_name: str
    yolo_score: float
    suction_score: float
    normal_alignment: float
    center_px: Tuple[int, int]
    position: np.ndarray
    normal: np.ndarray
    orientation_xyzw: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_size: np.ndarray
    valid_points: int

