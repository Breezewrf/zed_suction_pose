#!/usr/bin/env python3

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import open3d as o3d
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from ultralytics import YOLO
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


INSTANCE_COLORS_BGR = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 165, 0),
]

CLUSTER_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 165, 255),
    (128, 128, 0),
    (255, 128, 0),
]


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


def _default_model_path() -> str:
    try:
        return os.path.join(
            get_package_share_directory("zed_suction_pose"),
            "checkpoints",
            "best.pt",
        )
    except Exception:
        return ""


def _parse_classes(value: str) -> Optional[List[int]]:
    value = value.strip()
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _as_unit_vector(values: Sequence[float], fallback: np.ndarray) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        return fallback.copy()
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return fallback.copy()
    return vec / norm


def _make_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _make_odd_bounded(value: float, min_value: int, max_value: int) -> int:
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


def _parse_axis(value: str) -> int:
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


class ZedSuctionPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("zed_suction_pose")

        default_model = _default_model_path()
        self._declare_parameters(default_model)
        self._read_parameters(default_model)

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")

        self.model = YOLO(self.model_path, task="segment")
        self.latest_camera_info: Optional[CameraIntrinsics] = None
        self.frame_count = 0
        self.last_process_time = 0.0
        self.last_error_log_time = 0.0
        self.last_camera_info_warn_time = 0.0
        self.last_stats_log_time = time.monotonic()
        self.processed_frame_count = 0
        self.last_stats_processed_frame_count = 0
        self.timing_sums: Dict[str, float] = {}
        self.timing_sample_count = 0
        self.last_instance_point_counts: List[Tuple[int, int, int]] = []

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pose_pub = (
            self.create_publisher(PoseArray, self.pose_array_topic, pub_qos)
            if self.publish_pose_array
            else None
        )
        self.detection_pub = (
            self.create_publisher(Detection3DArray, self.detection_topic, pub_qos)
            if self.publish_detections
            else None
        )
        self.marker_pub = (
            self.create_publisher(MarkerArray, self.marker_topic, pub_qos)
            if self.publish_markers
            else None
        )
        self.overlay_pub = (
            self.create_publisher(Image, self.overlay_topic, pub_qos)
            if self.publish_overlay
            else None
        )
        self.heatmap_overlay_pub = (
            self.create_publisher(Image, self.heatmap_overlay_topic, pub_qos)
            if self.publish_heatmap_overlay
            else None
        )
        self.cluster_overlay_pub = (
            self.create_publisher(Image, self.cluster_overlay_topic, pub_qos)
            if self.publish_cluster_overlay
            else None
        )
        self.visualization_grid_pub = (
            self.create_publisher(Image, self.visualization_grid_topic, pub_qos)
            if self.publish_visualization_grid
            else None
        )
        self.masked_cloud_pub = (
            self.create_publisher(PointCloud2, self.masked_cloud_topic, pub_qos)
            if self.publish_masked_cloud
            else None
        )
        self.cluster_cloud_pub = (
            self.create_publisher(PointCloud2, self.cluster_cloud_topic, pub_qos)
            if self.publish_cluster_cloud
            else None
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_tf else None

        sub_qos = self._make_input_qos()
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sub_qos,
        )
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=sub_qos,
        )
        self.cloud_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            self.pointcloud_topic,
            qos_profile=sub_qos,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cloud_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"Loaded YOLO model: {self.model_path}")
        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")
        self.get_logger().info(f"Subscribed point cloud topic: {self.pointcloud_topic}")
        self.get_logger().info(f"Subscribed camera info topic: {self.camera_info_topic}")
        self.get_logger().info(f"Using point cloud {self.depth_axis_name.upper()} axis for range filtering")

    def _declare_parameters(self, default_model: str) -> None:
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("image_topic", "/zed/zed_node/rgb/color/rect/image")
        self.declare_parameter("pointcloud_topic", "/zed/zed_node/point_cloud/cloud_registered")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/rgb/color/rect/camera_info")
        self.declare_parameter("pose_array_topic", "/suction_poses")
        self.declare_parameter("detection_topic", "/suction_detections")
        self.declare_parameter("marker_topic", "/suction_markers")
        self.declare_parameter("overlay_topic", "/suction_debug/overlay")
        self.declare_parameter("masked_cloud_topic", "/suction_debug/masked_cloud")
        self.declare_parameter("heatmap_overlay_topic", "/suction_debug/heatmap_overlay")
        self.declare_parameter("cluster_overlay_topic", "/suction_debug/cluster_overlay")
        self.declare_parameter("visualization_grid_topic", "/suction_debug/visualization_grid")
        self.declare_parameter("cluster_cloud_topic", "/suction_debug/cluster_cloud")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.70)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_det", 50)
        self.declare_parameter("device", "")
        self.declare_parameter("classes", "")
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("verbose", False)
        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("sync_slop_sec", 0.20)
        self.declare_parameter("input_qos_reliability", "best_effort")
        self.declare_parameter("process_every_n_frames", 1)
        self.declare_parameter("target_fps", 0.0)
        self.declare_parameter("log_timing", True)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.5)
        self.declare_parameter("depth_axis", "x")
        self.declare_parameter("min_mask_area_px", 300)
        self.declare_parameter("mask_downsample_min_coverage", 0.01)
        self.declare_parameter("cloud_mask_dilate_px", 0)
        self.declare_parameter("min_valid_points", 80)
        self.declare_parameter("downsample_scale", 1)
        self.declare_parameter("knn", 80)
        self.declare_parameter("std_window", 25)
        self.declare_parameter("dynamic_std_window", True)
        self.declare_parameter("std_window_min", 5)
        self.declare_parameter("std_window_max", 41)
        self.declare_parameter("std_window_mask_ratio", 0.08)
        self.declare_parameter("std_window_max_mask_fraction", 0.33)
        self.declare_parameter("heatmap_threshold", 0.60)
        self.declare_parameter("min_cluster_area_px", 80)
        self.declare_parameter("min_pose_score", 0.50)
        self.declare_parameter("fallback_to_best_heatmap", False)
        self.declare_parameter("fallback_min_heatmap_score", 0.05)
        self.declare_parameter("max_objects", 20)
        self.declare_parameter("multi_pose_per_mask", True)
        self.declare_parameter("max_poses_per_mask", 3)
        self.declare_parameter("plane_merge_angle_deg", 10.0)
        self.declare_parameter("plane_merge_distance_m", 0.02)
        self.declare_parameter("suction_width_mm", 10.0)
        self.declare_parameter("suction_height_mm", 10.0)
        self.declare_parameter("fallback_patch_width_px", 10)
        self.declare_parameter("fallback_patch_height_px", 10)
        self.declare_parameter("weight_heatmap", 0.6)
        self.declare_parameter("weight_distance", 0.4)
        self.declare_parameter("normal_orientation", [-1.0, 0.0, 0.0])
        self.declare_parameter("invert_normal_for_pose", False)
        self.declare_parameter("pose_reference_axis", [0.0, 1.0, 0.0])
        self.declare_parameter("use_pca_pose_orientation", True)
        self.declare_parameter("pose_orientation_method", "cluster_obb")
        self.declare_parameter("pose_obb_min_aspect_ratio", 1.05)
        self.declare_parameter("pose_pca_patch_scale", 2.0)
        self.declare_parameter("pose_pca_min_points", 20)
        self.declare_parameter("pose_pca_min_axis_ratio", 1.05)
        self.declare_parameter("pose_axis_marker_length_m", 0.08)
        self.declare_parameter("pose_footprint_thickness_m", 0.003)
        self.declare_parameter("publish_pose_array", True)
        self.declare_parameter("publish_detections", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("publish_heatmap_overlay", True)
        self.declare_parameter("publish_cluster_overlay", True)
        self.declare_parameter("publish_visualization_grid", False)
        self.declare_parameter("publish_masked_cloud", True)
        self.declare_parameter("publish_cluster_cloud", True)
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("tf_child_prefix", "suction_object")
        self.declare_parameter("debug_cloud_max_points_per_object", 2500)

    def _read_parameters(self, default_model: str) -> None:
        self.model_path = self.get_parameter("model_path").value or default_model
        self.image_topic = self.get_parameter("image_topic").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pose_array_topic = self.get_parameter("pose_array_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.overlay_topic = self.get_parameter("overlay_topic").value
        self.masked_cloud_topic = self.get_parameter("masked_cloud_topic").value
        self.heatmap_overlay_topic = self.get_parameter("heatmap_overlay_topic").value
        self.cluster_overlay_topic = self.get_parameter("cluster_overlay_topic").value
        self.visualization_grid_topic = self.get_parameter("visualization_grid_topic").value
        self.cluster_cloud_topic = self.get_parameter("cluster_cloud_topic").value
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.device = self.get_parameter("device").value
        self.classes = _parse_classes(self.get_parameter("classes").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.verbose = bool(self.get_parameter("verbose").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.process_every_n_frames = max(1, int(self.get_parameter("process_every_n_frames").value))
        self.target_fps = float(self.get_parameter("target_fps").value)
        self.log_timing = bool(self.get_parameter("log_timing").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_axis_index = _parse_axis(self.get_parameter("depth_axis").value)
        self.depth_axis_name = "xyz"[self.depth_axis_index]
        self.min_mask_area_px = int(self.get_parameter("min_mask_area_px").value)
        self.mask_downsample_min_coverage = float(
            self.get_parameter("mask_downsample_min_coverage").value
        )
        self.mask_downsample_min_coverage = float(
            np.clip(self.mask_downsample_min_coverage, 0.0, 1.0)
        )
        self.cloud_mask_dilate_px = max(0, int(self.get_parameter("cloud_mask_dilate_px").value))
        self.min_valid_points = int(self.get_parameter("min_valid_points").value)
        self.downsample_scale = max(1, int(self.get_parameter("downsample_scale").value))
        self.knn = max(3, int(self.get_parameter("knn").value))
        self.std_window = _make_odd(int(self.get_parameter("std_window").value))
        self.dynamic_std_window = bool(self.get_parameter("dynamic_std_window").value)
        self.std_window_min = _make_odd(int(self.get_parameter("std_window_min").value))
        self.std_window_max = _make_odd(int(self.get_parameter("std_window_max").value))
        if self.std_window_max < self.std_window_min:
            self.std_window_max = self.std_window_min
        self.std_window_mask_ratio = max(0.0, float(self.get_parameter("std_window_mask_ratio").value))
        self.std_window_max_mask_fraction = float(
            np.clip(float(self.get_parameter("std_window_max_mask_fraction").value), 0.05, 1.0)
        )
        self.heatmap_threshold = float(self.get_parameter("heatmap_threshold").value)
        self.min_cluster_area_px = int(self.get_parameter("min_cluster_area_px").value)
        self.min_pose_score = float(self.get_parameter("min_pose_score").value)
        self.fallback_to_best_heatmap = bool(self.get_parameter("fallback_to_best_heatmap").value)
        self.fallback_min_heatmap_score = float(self.get_parameter("fallback_min_heatmap_score").value)
        self.max_objects = int(self.get_parameter("max_objects").value)
        self.multi_pose_per_mask = bool(self.get_parameter("multi_pose_per_mask").value)
        self.max_poses_per_mask = max(1, int(self.get_parameter("max_poses_per_mask").value))
        self.plane_merge_angle_deg = float(self.get_parameter("plane_merge_angle_deg").value)
        self.plane_merge_angle_cos = float(
            np.cos(np.deg2rad(np.clip(self.plane_merge_angle_deg, 0.0, 90.0)))
        )
        self.plane_merge_distance_m = max(0.0, float(self.get_parameter("plane_merge_distance_m").value))
        self.suction_width_mm = float(self.get_parameter("suction_width_mm").value)
        self.suction_height_mm = float(self.get_parameter("suction_height_mm").value)
        self.fallback_patch_width_px = int(self.get_parameter("fallback_patch_width_px").value)
        self.fallback_patch_height_px = int(self.get_parameter("fallback_patch_height_px").value)
        self.weight_heatmap = float(self.get_parameter("weight_heatmap").value)
        self.weight_distance = float(self.get_parameter("weight_distance").value)
        self.normal_orientation = _as_unit_vector(
            self.get_parameter("normal_orientation").value,
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.invert_normal_for_pose = bool(self.get_parameter("invert_normal_for_pose").value)
        self.pose_reference_axis = _as_unit_vector(
            self.get_parameter("pose_reference_axis").value,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        self.use_pca_pose_orientation = bool(self.get_parameter("use_pca_pose_orientation").value)
        self.pose_orientation_method = str(self.get_parameter("pose_orientation_method").value).strip().lower()
        valid_pose_methods = {"cluster_obb", "pca", "reference"}
        if self.pose_orientation_method not in valid_pose_methods:
            self.get_logger().warn(
                f"Invalid pose_orientation_method '{self.pose_orientation_method}', using 'cluster_obb'"
            )
            self.pose_orientation_method = "cluster_obb"
        self.pose_obb_min_aspect_ratio = max(
            1.0, float(self.get_parameter("pose_obb_min_aspect_ratio").value)
        )
        self.pose_pca_patch_scale = max(0.1, float(self.get_parameter("pose_pca_patch_scale").value))
        self.pose_pca_min_points = max(3, int(self.get_parameter("pose_pca_min_points").value))
        self.pose_pca_min_axis_ratio = max(1.0, float(self.get_parameter("pose_pca_min_axis_ratio").value))
        self.pose_axis_marker_length_m = max(
            0.0, float(self.get_parameter("pose_axis_marker_length_m").value)
        )
        self.pose_footprint_thickness_m = max(
            0.001, float(self.get_parameter("pose_footprint_thickness_m").value)
        )
        self.publish_pose_array = bool(self.get_parameter("publish_pose_array").value)
        self.publish_detections = bool(self.get_parameter("publish_detections").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.publish_heatmap_overlay = bool(self.get_parameter("publish_heatmap_overlay").value)
        self.publish_cluster_overlay = bool(self.get_parameter("publish_cluster_overlay").value)
        self.publish_visualization_grid = bool(self.get_parameter("publish_visualization_grid").value)
        self.publish_masked_cloud = bool(self.get_parameter("publish_masked_cloud").value)
        self.publish_cluster_cloud = bool(self.get_parameter("publish_cluster_cloud").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.tf_child_prefix = self.get_parameter("tf_child_prefix").value
        self.debug_cloud_max_points_per_object = int(
            self.get_parameter("debug_cloud_max_points_per_object").value
        )

    def _make_input_qos(self) -> QoSProfile:
        reliability_param = self.get_parameter("input_qos_reliability").value.lower()
        reliability = (
            ReliabilityPolicy.RELIABLE
            if reliability_param == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = CameraIntrinsics(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(msg.k[0]),
            fy=float(msg.k[4]),
            cx=float(msg.k[2]),
            cy=float(msg.k[5]),
        )

    def synced_callback(self, image_msg: Image, cloud_msg: PointCloud2) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        now = time.monotonic()
        if self.target_fps > 0.0 and now - self.last_process_time < 1.0 / self.target_fps:
            return
        self.last_process_time = now

        try:
            callback_start = time.monotonic()
            stage_start = callback_start
            bgr_image = self._image_msg_to_bgr(image_msg)
            xyz_img = self._pointcloud_msg_to_xyz_image(cloud_msg)
            after_input = time.monotonic()
            result = self.model.predict(
                source=bgr_image,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz if self.imgsz > 0 else None,
                device=self.device if self.device else None,
                classes=self.classes,
                max_det=self.max_det,
                retina_masks=self.retina_masks,
                verbose=self.verbose,
            )[0]
            after_yolo = time.monotonic()

            instances = self._extract_instances(result, bgr_image.shape[:2], xyz_img.shape[:2])
            after_extract = time.monotonic()
            poses, debug_points, cluster_points, heatmaps, cluster_candidates = self._process_instances(
                instances,
                xyz_img,
            )
            after_suction = time.monotonic()

            header = Header()
            header.stamp = cloud_msg.header.stamp
            header.frame_id = cloud_msg.header.frame_id

            self._publish_results(header, poses, debug_points, cluster_points)
            after_core_publish = time.monotonic()
            if self.overlay_pub:
                overlay = self._make_overlay(result, bgr_image, poses, xyz_img.shape[:2])
                self._publish_bgr_image(self.overlay_pub, overlay, image_msg.header)
            if self.heatmap_overlay_pub:
                heatmap_overlay = self._make_heatmap_overlay(bgr_image, heatmaps, xyz_img.shape[:2])
                self._publish_bgr_image(self.heatmap_overlay_pub, heatmap_overlay, image_msg.header)
            if self.cluster_overlay_pub:
                cluster_overlay = self._make_cluster_overlay(
                    bgr_image,
                    cluster_candidates,
                    xyz_img.shape[:2],
                )
                self._publish_bgr_image(self.cluster_overlay_pub, cluster_overlay, image_msg.header)
            if self.visualization_grid_pub:
                grid = self._make_visualization_grid(
                    result,
                    bgr_image,
                    instances,
                    xyz_img,
                    heatmaps,
                    cluster_candidates,
                    poses,
                )
                self._publish_bgr_image(self.visualization_grid_pub, grid, image_msg.header)
            after_debug_publish = time.monotonic()

            self._record_timings(
                {
                    "input": after_input - stage_start,
                    "yolo": after_yolo - after_input,
                    "extract": after_extract - after_yolo,
                    "suction": after_suction - after_extract,
                    "publish": after_core_publish - after_suction,
                    "debug": after_debug_publish - after_core_publish,
                    "total": after_debug_publish - callback_start,
                }
            )
            self.processed_frame_count += 1
            self._log_stats(bgr_image.shape[:2], xyz_img.shape[:2], len(instances), len(poses))

        except Exception as exc:
            self._log_callback_error(exc)

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        image = self._image_msg_to_numpy_u8(msg, encoding)

        if encoding in ("bgr8", "8uc3"):
            return np.ascontiguousarray(image)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding in ("bgra8", "8uc4"):
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    @staticmethod
    def _image_msg_to_numpy_u8(msg: Image, encoding: str) -> np.ndarray:
        channels_by_encoding = {
            "mono8": 1,
            "8uc1": 1,
            "bgr8": 3,
            "rgb8": 3,
            "8uc3": 3,
            "bgra8": 4,
            "rgba8": 4,
            "8uc4": 4,
        }
        channels = channels_by_encoding.get(encoding)
        if channels is None:
            raise ValueError(f"Unsupported uint8 image encoding: {msg.encoding}")

        bytes_per_pixel = channels
        min_step = int(msg.width) * bytes_per_pixel
        if int(msg.step) < min_step:
            raise ValueError(
                f"Invalid image step {msg.step} for {msg.width}x{msg.height} {msg.encoding}"
            )

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((int(msg.height), int(msg.step)))
        image_bytes = rows[:, :min_step]
        if channels == 1:
            image = image_bytes.reshape((int(msg.height), int(msg.width)))
        else:
            image = image_bytes.reshape((int(msg.height), int(msg.width), channels))
        return np.ascontiguousarray(image)

    def _pointcloud_msg_to_xyz_image(self, msg: PointCloud2) -> np.ndarray:
        if msg.height <= 1:
            raise ValueError("ZED point cloud must be organized; received height <= 1")
        xyz = pc2.read_points_numpy(
            msg,
            field_names=["x", "y", "z"],
            skip_nans=False,
            reshape_organized_cloud=False,
        )
        xyz = xyz.reshape((msg.height, msg.width, 3))
        return np.ascontiguousarray(xyz.astype(np.float32, copy=False))

    def _extract_instances(
        self,
        result,
        image_shape: Tuple[int, int],
        cloud_shape: Tuple[int, int],
    ) -> List[InstanceMask]:
        if result.masks is None or result.masks.data is None:
            return []

        masks = result.masks.data.detach().cpu().numpy()
        if masks.ndim != 3:
            return []

        boxes = result.boxes
        names: Dict[int, str] = getattr(result, "names", {}) or {}
        instances: List[InstanceMask] = []
        cloud_h, cloud_w = cloud_shape

        for idx, raw_mask in enumerate(masks[: self.max_objects]):
            # Binarize the mask and resize the image to match the pointcloud shapes
            mask_img = (raw_mask > 0.5).astype(np.uint8)
            if mask_img.shape != image_shape:
                mask_img = cv2.resize(
                    mask_img,
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if mask_img.shape != cloud_shape:
                coverage = cv2.resize(
                    mask_img.astype(np.float32),
                    (cloud_w, cloud_h),
                    interpolation=cv2.INTER_AREA,
                )
                mask = coverage >= self.mask_downsample_min_coverage
            else:
                mask = mask_img.astype(bool)
            
            # Dilate the mask
            if self.cloud_mask_dilate_px > 0:
                kernel_size = 2 * self.cloud_mask_dilate_px + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            area = int(np.count_nonzero(mask))
            if area < self.min_mask_area_px:
                continue

            class_id = -1
            yolo_score = 1.0
            if boxes is not None and idx < len(boxes):
                class_id = int(boxes.cls[idx].detach().cpu().item())
                yolo_score = float(boxes.conf[idx].detach().cpu().item())
            class_name = names.get(class_id, str(class_id))
            instances.append(
                InstanceMask(
                    index=idx,
                    class_id=class_id,
                    class_name=class_name,
                    yolo_score=yolo_score,
                    mask=mask,
                )
            )

        return instances

    def _process_instances(
        self,
        instances: List[InstanceMask],
        xyz_img: np.ndarray,
    ) -> Tuple[
        List[SuctionPose],
        List[Tuple[np.ndarray, Tuple[int, int, int]]],
        List[Tuple[np.ndarray, Tuple[int, int, int]]],
        List[np.ndarray],
        List[Dict],
    ]:
        poses: List[SuctionPose] = []
        debug_points: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        cluster_points: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        heatmaps: List[np.ndarray] = []
        cluster_candidates: List[Dict] = []
        self.last_instance_point_counts = []

        for instance in instances:
            raw_valid = self._valid_mask_for_cloud(xyz_img, instance.mask)
            mask_area = int(np.count_nonzero(instance.mask))
            valid_count = int(np.count_nonzero(raw_valid))
            self.last_instance_point_counts.append((instance.index, mask_area, valid_count))
            if self.publish_masked_cloud:
                color = INSTANCE_COLORS_BGR[instance.index % len(INSTANCE_COLORS_BGR)]
                points = self._sample_debug_points(xyz_img, raw_valid)
                if points.size > 0:
                    debug_points.append((points, color))

            heatmap, normal_map, valid_idx = self._estimate_suction_map(xyz_img, instance.mask)
            if heatmap is None or normal_map is None or valid_idx is None:
                continue
            heatmaps.append(heatmap)

            candidates = self._find_cluster_candidates(instance, xyz_img, heatmap, normal_map, valid_idx)
            if not candidates and self.fallback_to_best_heatmap:
                fallback = self._fallback_best_heatmap_candidate(
                    xyz_img,
                    heatmap,
                    normal_map,
                    instance.mask & valid_idx,
                )
                if fallback is not None:
                    fallback["instance_index"] = instance.index
                    fallback["cluster_id"] = -1
                    fallback["patch_size"] = self.fallback_patch_width_px, self.fallback_patch_height_px
                    fallback["is_fallback"] = True
                    candidates.append(fallback)
            selected_candidates = self._select_pose_candidates(candidates)
            if not selected_candidates:
                continue
            selected_candidate_ids = {id(selected_candidate) for selected_candidate in selected_candidates}

            for cluster_candidate in candidates:
                color = CLUSTER_COLORS_BGR[len(cluster_candidates) % len(CLUSTER_COLORS_BGR)]
                cluster_candidate["debug_color_bgr"] = color
                cluster_candidate["is_best"] = id(cluster_candidate) in selected_candidate_ids
                cluster_candidates.append(cluster_candidate)

                if self.publish_cluster_cloud:
                    points = self._sample_debug_points(xyz_img, cluster_candidate["cluster_valid"])
                    if points.size > 0:
                        cluster_points.append((points, color))

            for candidate in selected_candidates:
                pose = self._candidate_to_pose(instance, candidate, xyz_img, normal_map, valid_idx)
                if pose is None:
                    continue
                poses.append(pose)

        return poses, debug_points, cluster_points, heatmaps, cluster_candidates

    def _valid_mask_for_cloud(self, xyz_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        finite = np.isfinite(xyz_img).all(axis=2)
        depth = xyz_img[..., self.depth_axis_index]
        return mask & finite & (depth > self.min_depth_m) & (depth < self.max_depth_m)

    def _estimate_suction_map(
        self,
        xyz_img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        scale = self.downsample_scale
        xyz_ds = xyz_img[::scale, ::scale, :]
        mask_ds = mask[::scale, ::scale]

        finite = np.isfinite(xyz_ds).all(axis=2)
        depth = xyz_ds[..., self.depth_axis_index]
        valid_mask = (
            mask_ds
            & finite
            & (depth > self.min_depth_m)
            & (depth < self.max_depth_m)
        )

        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < self.min_valid_points:
            return None, None, None

        points = xyz_ds[valid_mask].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        knn = min(self.knn, max(3, valid_count - 1))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn),
            fast_normal_computation=True,
        )
        pcd.orient_normals_to_align_with_direction(self.normal_orientation.astype(np.float64))
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals).astype(np.float32)
        normal_map_ds = np.zeros_like(xyz_ds, dtype=np.float32)
        normal_map_ds[valid_mask] = normals

        std_window = self._std_window_for_mask(valid_mask)
        mean_normal_std = np.mean(self._std_filter(normal_map_ds, std_window), axis=2)
        max_std = float(np.max(mean_normal_std[valid_mask])) if valid_count > 0 else 0.0
        if max_std > 1e-9:
            heatmap_ds = 1.0 - (mean_normal_std / max_std)
        else:
            heatmap_ds = np.zeros_like(mean_normal_std, dtype=np.float32)
            heatmap_ds[valid_mask] = 1.0
        heatmap_ds[~valid_mask] = 0.0

        height, width = xyz_img.shape[:2]
        heatmap = cv2.resize(heatmap_ds, (width, height), interpolation=cv2.INTER_LINEAR)
        normal_map = cv2.resize(normal_map_ds, (width, height), interpolation=cv2.INTER_LINEAR)
        valid_idx = cv2.resize(
            valid_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normal_map = np.divide(
            normal_map,
            np.maximum(norms, 1e-9),
            out=np.zeros_like(normal_map),
            where=norms > 1e-9,
        )
        heatmap[~valid_idx] = 0.0
        return heatmap.astype(np.float32), normal_map.astype(np.float32), valid_idx

    def _std_window_for_mask(self, valid_mask: np.ndarray) -> int:
        if not self.dynamic_std_window:
            return self.std_window

        ys, xs = np.where(valid_mask)
        if ys.size == 0:
            return self.std_window

        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        short_side = max(1, min(bbox_w, bbox_h))
        image_limit = max(3, min(valid_mask.shape[:2]))
        mask_limit = max(3, int(round(short_side * self.std_window_max_mask_fraction)))
        upper = min(self.std_window_max, mask_limit, image_limit)

        return _make_odd_bounded(
            short_side * self.std_window_mask_ratio,
            self.std_window_min,
            upper,
        )

    @staticmethod
    def _std_filter(image: np.ndarray, window_size: int) -> np.ndarray:
        wmean = cv2.boxFilter(image, -1, (window_size, window_size), borderType=cv2.BORDER_REFLECT)
        wsqrmean = cv2.boxFilter(image * image, -1, (window_size, window_size), borderType=cv2.BORDER_REFLECT)
        return np.sqrt(np.maximum(wsqrmean - wmean * wmean, 0.0))

    def _find_cluster_candidates(
        self,
        instance: InstanceMask,
        xyz_img: np.ndarray,
        heatmap: np.ndarray,
        normal_map: np.ndarray,
        valid_idx: np.ndarray,
    ) -> List[Dict]:
        object_valid = instance.mask & valid_idx
        if int(np.count_nonzero(object_valid)) < self.min_valid_points:
            return []

        heatmap_binary = ((heatmap >= self.heatmap_threshold) & object_valid).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(heatmap_binary, connectivity=8)
        if num_labels <= 1:
            return []

        candidates: List[Dict] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_cluster_area_px:
                continue

            cluster_valid = (labels == label) & object_valid
            valid_depths = xyz_img[..., self.depth_axis_index][cluster_valid]
            valid_depths = valid_depths[np.isfinite(valid_depths)]
            if valid_depths.size == 0:
                continue

            patch_w, patch_h = self._suction_patch_size_px(float(np.median(valid_depths)), xyz_img.shape[:2])
            kernel = np.ones((patch_h, patch_w), dtype=np.float32) / float(patch_h * patch_w)

            heatmap_valid = np.zeros_like(heatmap, dtype=np.float32)
            heatmap_valid[cluster_valid] = heatmap[cluster_valid]
            smoothed = cv2.filter2D(heatmap_valid, -1, kernel)
            smoothed[~cluster_valid] = 0.0

            moments = cv2.moments(smoothed)
            if abs(moments["m00"]) < 1e-9:
                continue
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            ys, xs = np.where(smoothed > 0)
            if ys.size == 0:
                continue
            dist = np.zeros_like(smoothed, dtype=np.float32)
            dist[ys, xs] = np.sqrt((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2)
            max_dist = float(np.max(dist))
            dist_score = np.zeros_like(smoothed, dtype=np.float32)
            if max_dist > 1e-9:
                dist_score[ys, xs] = 1.0 - dist[ys, xs] / max_dist

            weighted = self.weight_heatmap * smoothed + self.weight_distance * dist_score
            weighted[~cluster_valid] = 0.0
            score = float(np.max(weighted))
            if score < self.min_pose_score:
                continue

            center_y, center_x = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
            point, center_px = self._nearest_valid_point(xyz_img, cluster_valid, center_x, center_y)
            if point is None:
                continue
            center_x, center_y = center_px

            normal = normal_map[center_y, center_x].astype(np.float64)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-6:
                normal = self._mean_valid_normal(normal_map, cluster_valid)
                normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-6:
                continue
            normal = normal / normal_norm

            alignment = float(abs(np.dot(normal, self.normal_orientation)))
            candidate = {
                "instance_index": instance.index,
                "cluster_id": label,
                "score": score,
                "center_px": (int(center_x), int(center_y)),
                "position": point.astype(np.float64),
                "normal": normal.astype(np.float64),
                "normal_alignment": alignment,
                "cluster_valid": cluster_valid,
                "patch_size": (int(patch_w), int(patch_h)),
                "is_fallback": False,
            }

            candidates.append(candidate)

        return candidates

    @staticmethod
    def _select_best_candidate(candidates: List[Dict]) -> Optional[Dict]:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item["normal_alignment"],
                item["score"],
            ),
        )

    def _select_pose_candidates(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        if not self.multi_pose_per_mask:
            best = self._select_best_candidate(candidates)
            return [best] if best is not None else []

        ranked = sorted(
            candidates,
            key=lambda item: (
                item["normal_alignment"],
                item["score"],
            ),
            reverse=True,
        )

        selected: List[Dict] = []
        for candidate in ranked:
            if any(self._is_same_suction_plane(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= self.max_poses_per_mask:
                break
        return selected

    def _is_same_suction_plane(self, candidate: Dict, existing: Dict) -> bool:
        normal_a = candidate["normal"].astype(np.float64)
        normal_b = existing["normal"].astype(np.float64)
        norm_a = np.linalg.norm(normal_a)
        norm_b = np.linalg.norm(normal_b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return False

        normal_a = normal_a / norm_a
        normal_b = normal_b / norm_b
        normal_cos = abs(float(np.dot(normal_a, normal_b)))
        if normal_cos < self.plane_merge_angle_cos:
            return False

        delta = candidate["position"].astype(np.float64) - existing["position"].astype(np.float64)
        distance_a = abs(float(np.dot(normal_a, delta)))
        distance_b = abs(float(np.dot(normal_b, delta)))
        return max(distance_a, distance_b) <= self.plane_merge_distance_m

    def _fallback_best_heatmap_candidate(
        self,
        xyz_img: np.ndarray,
        heatmap: np.ndarray,
        normal_map: np.ndarray,
        object_valid: np.ndarray,
    ) -> Optional[Dict]:
        if int(np.count_nonzero(object_valid)) < self.min_valid_points:
            return None

        candidate_scores = np.where(object_valid, heatmap, -np.inf)
        max_score = float(np.max(candidate_scores))
        if not np.isfinite(max_score) or max_score < self.fallback_min_heatmap_score:
            return None

        center_y, center_x = np.unravel_index(int(np.argmax(candidate_scores)), candidate_scores.shape)
        point, center_px = self._nearest_valid_point(xyz_img, object_valid, center_x, center_y)
        if point is None:
            return None
        center_x, center_y = center_px

        normal = normal_map[center_y, center_x].astype(np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            normal = self._mean_valid_normal(normal_map, object_valid)
            normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return None
        normal = normal / normal_norm

        return {
            "score": max_score,
            "center_px": (int(center_x), int(center_y)),
            "position": point.astype(np.float64),
            "normal": normal.astype(np.float64),
            "normal_alignment": float(abs(np.dot(normal, self.normal_orientation))),
            "cluster_valid": object_valid,
        }

    def _suction_patch_size_px(self, depth_m: float, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        height, width = image_shape
        info = self.latest_camera_info
        if info is None or depth_m <= 1e-6:
            now = time.monotonic()
            if now - self.last_camera_info_warn_time > 2.0:
                self.get_logger().warn("No camera_info yet; using fallback suction patch size")
                self.last_camera_info_warn_time = now
            patch_w = self.fallback_patch_width_px
            patch_h = self.fallback_patch_height_px
        else:
            scale_x = float(width) / float(info.width) if info.width > 0 else 1.0
            scale_y = float(height) / float(info.height) if info.height > 0 else 1.0
            fx = info.fx * scale_x
            fy = info.fy * scale_y
            patch_w = int((self.suction_width_mm / 1000.0) * fx / depth_m)
            patch_h = int((self.suction_height_mm / 1000.0) * fy / depth_m)

        max_size = max(5, min(width, height) // 2)
        patch_w = int(np.clip(patch_w, 5, max_size))
        patch_h = int(np.clip(patch_h, 5, max_size))
        return patch_w, patch_h

    def _record_timings(self, timings: Dict[str, float]) -> None:
        if not self.log_timing:
            return
        for name, value in timings.items():
            self.timing_sums[name] = self.timing_sums.get(name, 0.0) + float(value)
        self.timing_sample_count += 1

    def _log_stats(
        self,
        image_shape: Tuple[int, int],
        cloud_shape: Tuple[int, int],
        num_instances: int,
        num_poses: int,
    ) -> None:
        now = time.monotonic()
        elapsed = now - self.last_stats_log_time
        if elapsed < 2.0:
            return
        processed_delta = self.processed_frame_count - self.last_stats_processed_frame_count
        processing_fps = float(processed_delta) / elapsed if elapsed > 1e-9 else 0.0
        count_summary = ""
        if self.last_instance_point_counts:
            entries = [
                f"{idx}:{valid}/{area}"
                for idx, area, valid in self.last_instance_point_counts[:12]
            ]
            if len(self.last_instance_point_counts) > 12:
                entries.append("...")
            count_summary = f" mask_valid_points={','.join(entries)}"
        timing_summary = ""
        if self.log_timing and self.timing_sample_count > 0:
            timing_order = ("total", "input", "yolo", "extract", "suction", "publish", "debug")
            timing_entries = []
            for name in timing_order:
                if name not in self.timing_sums:
                    continue
                avg_ms = 1000.0 * self.timing_sums[name] / float(self.timing_sample_count)
                timing_entries.append(f"{name}:{avg_ms:.1f}")
            if timing_entries:
                timing_summary = f" timing_avg_ms={','.join(timing_entries)}"
        self.get_logger().info(
            f"Processed frame: image={image_shape[1]}x{image_shape[0]} "
            f"cloud={cloud_shape[1]}x{cloud_shape[0]} "
            f"depth_axis={self.depth_axis_name} "
            f"processing_fps={processing_fps:.2f} "
            f"instances={num_instances} suction_poses={num_poses}"
            f"{count_summary}"
            f"{timing_summary}"
        )
        self.last_stats_log_time = now
        self.last_stats_processed_frame_count = self.processed_frame_count
        self.timing_sums.clear()
        self.timing_sample_count = 0

    def _nearest_valid_point(
        self,
        xyz_img: np.ndarray,
        valid_mask: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        height, width = valid_mask.shape
        if 0 <= center_x < width and 0 <= center_y < height and valid_mask[center_y, center_x]:
            point = xyz_img[center_y, center_x]
            depth = point[self.depth_axis_index]
            if np.isfinite(point).all() and self.min_depth_m < depth < self.max_depth_m:
                return point, (center_x, center_y)

        ys, xs = np.where(valid_mask)
        if ys.size == 0:
            return None, (center_x, center_y)
        dist2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
        nearest = int(np.argmin(dist2))
        x = int(xs[nearest])
        y = int(ys[nearest])
        point = xyz_img[y, x]
        depth = point[self.depth_axis_index]
        if not np.isfinite(point).all() or not (self.min_depth_m < depth < self.max_depth_m):
            return None, (center_x, center_y)
        return point, (x, y)

    @staticmethod
    def _mean_valid_normal(normal_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        normals = normal_map[valid_mask].astype(np.float64)
        norms = np.linalg.norm(normals, axis=1)
        normals = normals[norms > 1e-6]
        if normals.size == 0:
            return np.zeros(3, dtype=np.float64)
        normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-9 else normal

    def _candidate_to_pose(
        self,
        instance: InstanceMask,
        candidate: Dict,
        xyz_img: np.ndarray,
        normal_map: np.ndarray,
        valid_idx: np.ndarray,
    ) -> Optional[SuctionPose]:
        object_valid = candidate.get("cluster_valid", instance.mask & valid_idx)
        points = xyz_img[object_valid]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] < self.min_valid_points:
            return None

        pose_frame = self._estimate_candidate_pose_frame(xyz_img, object_valid, candidate)
        if pose_frame is None:
            normal = candidate["normal"].astype(np.float64)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-9:
                return None
            normal = normal / normal_norm
            if self.invert_normal_for_pose:
                normal = -normal
            quat = self._orientation_from_normal(normal)
            if quat is None:
                return None
        else:
            quat, normal = pose_frame

        bbox_size = self._oriented_bbox_size(points, quat)

        return SuctionPose(
            object_id=instance.index,
            cluster_id=int(candidate.get("cluster_id", -1)),
            class_id=instance.class_id,
            class_name=instance.class_name,
            yolo_score=instance.yolo_score,
            suction_score=float(candidate["score"]),
            normal_alignment=float(candidate["normal_alignment"]),
            center_px=candidate["center_px"],
            position=candidate["position"].astype(np.float64),
            normal=normal.astype(np.float64),
            orientation_xyzw=quat,
            bbox_min=np.min(points, axis=0).astype(np.float64),
            bbox_max=np.max(points, axis=0).astype(np.float64),
            bbox_size=bbox_size,
            valid_points=int(points.shape[0]),
        )

    def _estimate_candidate_pose_frame(
        self,
        xyz_img: np.ndarray,
        valid_mask: np.ndarray,
        candidate: Dict,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.use_pca_pose_orientation:
            return None

        candidate_normal = candidate["normal"].astype(np.float64)
        candidate_normal_norm = np.linalg.norm(candidate_normal)
        if candidate_normal_norm < 1e-9:
            return None
        candidate_normal = candidate_normal / candidate_normal_norm

        if self.pose_orientation_method == "reference":
            return None

        if self.pose_orientation_method == "cluster_obb":
            pose_frame = self._estimate_cluster_obb_pose_frame(xyz_img, valid_mask, candidate_normal)
            if pose_frame is not None:
                return pose_frame

        if self.pose_orientation_method not in ("cluster_obb", "pca"):
            return None

        return self._estimate_pca_pose_frame(xyz_img, valid_mask, candidate, candidate_normal)

    def _estimate_pca_pose_frame(
        self,
        xyz_img: np.ndarray,
        valid_mask: np.ndarray,
        candidate: Dict,
        candidate_normal: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        points = self._candidate_pca_points(xyz_img, valid_mask, candidate)
        if points.shape[0] < self.pose_pca_min_points:
            return None

        eigvals, eigvecs = self._pointcloud_eigendecomposition(points)
        if eigvals is None or eigvecs is None:
            return None

        surface_normal = eigvecs[:, 0].astype(np.float64)
        if float(np.dot(surface_normal, candidate_normal)) < 0.0:
            surface_normal = -surface_normal

        z_axis = surface_normal
        if self.invert_normal_for_pose:
            z_axis = -z_axis
        z_axis_norm = np.linalg.norm(z_axis)
        if z_axis_norm < 1e-9:
            return None
        z_axis = z_axis / z_axis_norm

        axis_ratio = float(eigvals[2] / max(eigvals[1], 1e-12))
        if axis_ratio < self.pose_pca_min_axis_ratio:
            quat = self._orientation_from_normal(z_axis)
        else:
            x_hint = eigvecs[:, 2].astype(np.float64)
            x_hint = self._orient_tangent_axis_sign(x_hint, z_axis)
            quat = self._orientation_from_axes(z_axis, x_hint)

        if quat is None:
            return None
        rotation = Rotation.from_quat(quat).as_matrix()
        return quat, rotation[:, 2].astype(np.float64)

    def _estimate_cluster_obb_pose_frame(
        self,
        xyz_img: np.ndarray,
        valid_mask: np.ndarray,
        candidate_normal: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        points = self._filtered_points(xyz_img, valid_mask)
        if points.shape[0] < self.pose_pca_min_points:
            return None

        eigvals, eigvecs = self._pointcloud_eigendecomposition(points)
        if eigvals is None or eigvecs is None:
            return None

        surface_normal = eigvecs[:, 0].astype(np.float64)
        if float(np.dot(surface_normal, candidate_normal)) < 0.0:
            surface_normal = -surface_normal

        z_axis = surface_normal
        if self.invert_normal_for_pose:
            z_axis = -z_axis
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            return None
        z_axis = z_axis / z_norm

        base_quat = self._orientation_from_normal(z_axis)
        if base_quat is None:
            return None
        base_rotation = Rotation.from_quat(base_quat).as_matrix()
        base_x = base_rotation[:, 0]
        base_y = base_rotation[:, 1]

        center = np.mean(points, axis=0)
        delta = points - center
        coords = np.column_stack((delta @ base_x, delta @ base_y)).astype(np.float32)
        if coords.shape[0] < self.pose_pca_min_points:
            return None

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect).astype(np.float64)
        edges = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(edges, axis=1)
        max_length = float(np.max(lengths))
        min_length = float(np.min(lengths))
        if max_length < 1e-9 or min_length < 1e-9:
            return None

        if max_length / min_length < self.pose_obb_min_aspect_ratio:
            quat = self._orientation_from_normal(z_axis)
            if quat is None:
                return None
            rotation = Rotation.from_quat(quat).as_matrix()
            return quat, rotation[:, 2].astype(np.float64)

        long_edge = edges[int(np.argmax(lengths))]
        long_edge = long_edge / max(np.linalg.norm(long_edge), 1e-9)
        x_hint = long_edge[0] * base_x + long_edge[1] * base_y
        x_hint = self._orient_tangent_axis_sign(x_hint, z_axis)

        quat = self._orientation_from_axes(z_axis, x_hint)
        if quat is None:
            return None
        rotation = Rotation.from_quat(quat).as_matrix()
        return quat, rotation[:, 2].astype(np.float64)

    @staticmethod
    def _pointcloud_eigendecomposition(
        points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        centered = points - np.mean(points, axis=0, keepdims=True)
        if not np.all(np.isfinite(centered)):
            return None, None

        cov = (centered.T @ centered) / float(points.shape[0])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None, None
        if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
            return None, None

        order = np.argsort(eigvals)
        return eigvals[order], eigvecs[:, order]

    def _candidate_pca_points(
        self,
        xyz_img: np.ndarray,
        valid_mask: np.ndarray,
        candidate: Dict,
    ) -> np.ndarray:
        patch_w, patch_h = candidate.get(
            "patch_size",
            (self.fallback_patch_width_px, self.fallback_patch_height_px),
        )
        center_x, center_y = candidate["center_px"]
        height, width = valid_mask.shape
        half_w = max(2, int(round(float(patch_w) * self.pose_pca_patch_scale * 0.5)))
        half_h = max(2, int(round(float(patch_h) * self.pose_pca_patch_scale * 0.5)))

        x0 = max(0, int(center_x) - half_w)
        x1 = min(width, int(center_x) + half_w + 1)
        y0 = max(0, int(center_y) - half_h)
        y1 = min(height, int(center_y) + half_h + 1)

        local_mask = np.zeros_like(valid_mask, dtype=bool)
        local_mask[y0:y1, x0:x1] = valid_mask[y0:y1, x0:x1]

        points = self._filtered_points(xyz_img, local_mask)
        if points.shape[0] < self.pose_pca_min_points:
            points = self._filtered_points(xyz_img, valid_mask)
        return points

    def _filtered_points(self, xyz_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        points = xyz_img[mask].astype(np.float64)
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        finite = np.isfinite(points).all(axis=1)
        depth = points[:, self.depth_axis_index]
        valid = finite & (depth > self.min_depth_m) & (depth < self.max_depth_m)
        return points[valid]

    @staticmethod
    def _oriented_bbox_size(points: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)
        rotation = Rotation.from_quat(orientation_xyzw).as_matrix()
        local = points.astype(np.float64) @ rotation
        return np.maximum(np.max(local, axis=0) - np.min(local, axis=0), 0.0).astype(np.float64)

    def _orient_tangent_axis_sign(self, axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
        x_axis = axis.astype(np.float64)
        x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            return x_axis
        x_axis = x_axis / x_norm

        ref = self.pose_reference_axis.astype(np.float64)
        ref = ref - float(np.dot(ref, z_axis)) * z_axis
        ref_norm = np.linalg.norm(ref)
        if ref_norm < 1e-9:
            return x_axis
        ref = ref / ref_norm
        return x_axis if float(np.dot(x_axis, ref)) >= 0.0 else -x_axis

    def _orientation_from_normal(self, normal: np.ndarray) -> Optional[np.ndarray]:
        z_axis = normal.astype(np.float64)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            return None
        z_axis = z_axis / z_norm

        ref = self.pose_reference_axis.copy()
        if abs(float(np.dot(ref, z_axis))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, z_axis))) > 0.95:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        x_hint = np.cross(ref, z_axis)
        return self._orientation_from_axes(z_axis, x_hint)

    def _orientation_from_axes(
        self,
        z_axis: np.ndarray,
        x_hint: np.ndarray,
    ) -> Optional[np.ndarray]:
        z_axis = z_axis.astype(np.float64)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-9:
            return None
        z_axis = z_axis / z_norm

        x_axis = x_hint.astype(np.float64)
        x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            for ref in (
                self.pose_reference_axis,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ):
                x_axis = ref.astype(np.float64) - float(np.dot(ref, z_axis)) * z_axis
                x_norm = np.linalg.norm(x_axis)
                if x_norm >= 1e-9:
                    break
            if x_norm < 1e-9:
                return None
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)

        rotation = np.column_stack((x_axis, y_axis, z_axis))
        if np.linalg.det(rotation) < 0.0:
            y_axis = -y_axis
            rotation = np.column_stack((x_axis, y_axis, z_axis))

        return Rotation.from_matrix(rotation).as_quat()

    def _publish_results(
        self,
        header: Header,
        poses: List[SuctionPose],
        debug_points: List[Tuple[np.ndarray, Tuple[int, int, int]]],
        cluster_points: List[Tuple[np.ndarray, Tuple[int, int, int]]],
    ) -> None:
        if self.pose_pub:
            self.pose_pub.publish(self._make_pose_array(header, poses))
        if self.detection_pub:
            self.detection_pub.publish(self._make_detection_array(header, poses))
        if self.marker_pub:
            self.marker_pub.publish(self._make_marker_array(header, poses))
        if self.masked_cloud_pub:
            self.masked_cloud_pub.publish(self._make_debug_cloud(header, debug_points))
        if self.cluster_cloud_pub:
            self.cluster_cloud_pub.publish(self._make_debug_cloud(header, cluster_points))
        if self.tf_broadcaster:
            self._publish_transforms(header, poses)

    @staticmethod
    def _pose_msg(pose: SuctionPose) -> Pose:
        msg = Pose()
        msg.position.x = float(pose.position[0])
        msg.position.y = float(pose.position[1])
        msg.position.z = float(pose.position[2])
        msg.orientation.x = float(pose.orientation_xyzw[0])
        msg.orientation.y = float(pose.orientation_xyzw[1])
        msg.orientation.z = float(pose.orientation_xyzw[2])
        msg.orientation.w = float(pose.orientation_xyzw[3])
        return msg

    def _make_pose_array(self, header: Header, poses: List[SuctionPose]) -> PoseArray:
        msg = PoseArray()
        msg.header = header
        msg.poses = [self._pose_msg(pose) for pose in poses]
        return msg

    def _make_detection_array(self, header: Header, poses: List[SuctionPose]) -> Detection3DArray:
        array = Detection3DArray()
        array.header = header
        for pose in poses:
            det = Detection3D()
            det.header = header
            det.id = f"{pose.object_id}:{pose.cluster_id}"

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = pose.class_name
            hyp.hypothesis.score = float(pose.suction_score * pose.yolo_score)
            hyp.pose.pose = self._pose_msg(pose)
            det.results.append(hyp)

            det.bbox.center = self._pose_msg(pose)
            size = np.maximum(pose.bbox_size, 0.0)
            det.bbox.size.x = float(size[0])
            det.bbox.size.y = float(size[1])
            det.bbox.size.z = float(size[2])
            array.detections.append(det)
        return array

    def _make_marker_array(self, header: Header, poses: List[SuctionPose]) -> MarkerArray:
        array = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for i, pose in enumerate(poses):
            color_bgr = INSTANCE_COLORS_BGR[pose.object_id % len(INSTANCE_COLORS_BGR)]
            color = ColorRGBA(
                r=float(color_bgr[2]) / 255.0,
                g=float(color_bgr[1]) / 255.0,
                b=float(color_bgr[0]) / 255.0,
                a=0.9,
            )

            footprint = Marker()
            footprint.header = header
            footprint.ns = "suction_footprint"
            footprint.id = i
            footprint.type = Marker.CUBE
            footprint.action = Marker.ADD
            footprint.pose = self._pose_msg(pose)
            footprint.scale.x = max(0.001, self.suction_width_mm / 1000.0)
            footprint.scale.y = max(0.001, self.suction_height_mm / 1000.0)
            footprint.scale.z = self.pose_footprint_thickness_m
            footprint.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.35)
            array.markers.append(footprint)

            rotation = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
            axis_specs = [
                ("suction_axis_x", 0, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95)),
                ("suction_axis_y", 1, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.95)),
                ("suction_axis_z", 2, ColorRGBA(r=0.0, g=0.35, b=1.0, a=0.95)),
            ]
            for ns, axis_index, axis_color in axis_specs:
                axis = rotation[:, axis_index]
                arrow = Marker()
                arrow.header = header
                arrow.ns = ns
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                start = Point()
                start.x = float(pose.position[0])
                start.y = float(pose.position[1])
                start.z = float(pose.position[2])
                end = Point()
                end.x = float(pose.position[0] + self.pose_axis_marker_length_m * axis[0])
                end.y = float(pose.position[1] + self.pose_axis_marker_length_m * axis[1])
                end.z = float(pose.position[2] + self.pose_axis_marker_length_m * axis[2])
                arrow.points = [start, end]
                arrow.scale.x = 0.006
                arrow.scale.y = 0.014
                arrow.scale.z = 0.022
                arrow.color = axis_color
                array.markers.append(arrow)

            sphere = Marker()
            sphere.header = header
            sphere.ns = "suction_point"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = self._pose_msg(pose)
            sphere.scale.x = 0.025
            sphere.scale.y = 0.025
            sphere.scale.z = 0.025
            sphere.color = color
            array.markers.append(sphere)

        return array

    def _publish_transforms(self, header: Header, poses: List[SuctionPose]) -> None:
        transforms = []
        for pose in poses:
            tf = TransformStamped()
            tf.header = header
            tf.child_frame_id = f"{self.tf_child_prefix}_{pose.object_id}_{pose.cluster_id}"
            tf.transform.translation.x = float(pose.position[0])
            tf.transform.translation.y = float(pose.position[1])
            tf.transform.translation.z = float(pose.position[2])
            tf.transform.rotation.x = float(pose.orientation_xyzw[0])
            tf.transform.rotation.y = float(pose.orientation_xyzw[1])
            tf.transform.rotation.z = float(pose.orientation_xyzw[2])
            tf.transform.rotation.w = float(pose.orientation_xyzw[3])
            transforms.append(tf)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def _sample_debug_points(self, xyz_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        points = xyz_img[mask]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] == 0:
            return points
        max_points = self.debug_cloud_max_points_per_object
        if max_points > 0 and points.shape[0] > max_points:
            indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
            points = points[indices]
        return points.astype(np.float32, copy=False)

    @staticmethod
    def _pack_rgb_float(color_bgr: Tuple[int, int, int]) -> np.float32:
        b, g, r = color_bgr
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return np.frombuffer(np.array([rgb_uint32], dtype=np.uint32).tobytes(), dtype=np.float32)[0]

    def _make_debug_cloud(
        self,
        header: Header,
        debug_points: List[Tuple[np.ndarray, Tuple[int, int, int]]],
    ) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        rows = []
        for points, color in debug_points:
            rgb = self._pack_rgb_float(color)
            rgb_col = np.full((points.shape[0], 1), rgb, dtype=np.float32)
            rows.append(np.hstack((points.astype(np.float32), rgb_col)))

        if rows:
            data = np.vstack(rows).astype(np.float32)
        else:
            data = np.zeros((0, 4), dtype=np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(data.shape[0])
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = data.tobytes()
        return msg

    @staticmethod
    def _scale_point(
        point_xy: Tuple[int, int],
        source_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        src_h, src_w = source_shape
        dst_h, dst_w = target_shape
        if src_w <= 0 or src_h <= 0:
            return point_xy
        x = int(round(float(point_xy[0]) * float(dst_w) / float(src_w)))
        y = int(round(float(point_xy[1]) * float(dst_h) / float(src_h)))
        return int(np.clip(x, 0, max(0, dst_w - 1))), int(np.clip(y, 0, max(0, dst_h - 1)))

    @staticmethod
    def _scale_size(
        size_wh: Tuple[int, int],
        source_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> Tuple[int, int]:
        src_h, src_w = source_shape
        dst_h, dst_w = target_shape
        if src_w <= 0 or src_h <= 0:
            return size_wh
        width = int(round(float(size_wh[0]) * float(dst_w) / float(src_w)))
        height = int(round(float(size_wh[1]) * float(dst_h) / float(src_h)))
        return max(1, width), max(1, height)

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        height, width = target_shape
        if mask.shape == target_shape:
            return mask.astype(bool)
        return cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    @staticmethod
    def _label_debug_image(image: np.ndarray, label: str) -> np.ndarray:
        output = image.copy()
        cv2.putText(
            output,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _make_heatmap_overlay(
        self,
        bgr_image: np.ndarray,
        heatmaps: List[np.ndarray],
        cloud_shape: Tuple[int, int],
    ) -> np.ndarray:
        if not heatmaps:
            return bgr_image.copy()

        combined = np.zeros(cloud_shape, dtype=np.float32)
        for heatmap in heatmaps:
            if heatmap.shape != cloud_shape:
                heatmap = cv2.resize(
                    heatmap.astype(np.float32),
                    (cloud_shape[1], cloud_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            combined = np.maximum(combined, heatmap.astype(np.float32))

        image_h, image_w = bgr_image.shape[:2]
        combined_img = cv2.resize(combined, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        heatmap_u8 = np.clip(combined_img * 255.0, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(bgr_image, 0.5, heatmap_color, 0.5, 0)

    def _make_cluster_overlay(
        self,
        bgr_image: np.ndarray,
        cluster_candidates: List[Dict],
        cloud_shape: Tuple[int, int],
    ) -> np.ndarray:
        image_shape = bgr_image.shape[:2]
        cluster_vis = np.zeros_like(bgr_image)

        for i, candidate in enumerate(cluster_candidates):
            color = candidate.get("debug_color_bgr", CLUSTER_COLORS_BGR[i % len(CLUSTER_COLORS_BGR)])
            cluster_mask = self._resize_mask(candidate["cluster_valid"], image_shape)
            cluster_vis[cluster_mask] = color

        for i, candidate in enumerate(cluster_candidates):
            color = candidate.get("debug_color_bgr", CLUSTER_COLORS_BGR[i % len(CLUSTER_COLORS_BGR)])
            center = self._scale_point(candidate["center_px"], cloud_shape, image_shape)
            patch_w, patch_h = self._scale_size(candidate.get("patch_size", (8, 8)), cloud_shape, image_shape)
            x0 = int(np.clip(center[0] - patch_w // 2, 0, image_shape[1] - 1))
            y0 = int(np.clip(center[1] - patch_h // 2, 0, image_shape[0] - 1))
            x1 = int(np.clip(center[0] + patch_w // 2, 0, image_shape[1] - 1))
            y1 = int(np.clip(center[1] + patch_h // 2, 0, image_shape[0] - 1))
            is_best = bool(candidate.get("is_best", False))
            border = (255, 255, 255) if is_best else color
            thickness = 3 if is_best else 1
            cv2.rectangle(cluster_vis, (x0, y0), (x1, y1), border, thickness)
            cv2.circle(cluster_vis, center, 5 if is_best else 3, border, -1)
            label = f"{candidate['instance_index']}:{candidate['cluster_id']} a={candidate['normal_alignment']:.2f}"
            cv2.putText(
                cluster_vis,
                label,
                (min(image_shape[1] - 1, center[0] + 6), max(14, center[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                border,
                1,
                cv2.LINE_AA,
            )

        return cluster_vis

    def _make_mask_visualization(
        self,
        bgr_image: np.ndarray,
        instances: List[InstanceMask],
        cloud_shape: Tuple[int, int],
    ) -> np.ndarray:
        mask_vis = np.zeros_like(bgr_image)
        image_shape = bgr_image.shape[:2]
        for i, instance in enumerate(instances):
            color = INSTANCE_COLORS_BGR[instance.index % len(INSTANCE_COLORS_BGR)]
            mask = self._resize_mask(instance.mask, image_shape)
            mask_vis[mask] = color
        return mask_vis

    def _make_depth_visualization(
        self,
        xyz_img: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        depth = xyz_img[..., self.depth_axis_index]
        finite = np.isfinite(xyz_img).all(axis=2)
        valid = finite & (depth > self.min_depth_m) & (depth < self.max_depth_m)
        depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            values = depth[valid]
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            denom = max(max_value - min_value, 1e-6)
            depth_u8[valid] = np.clip((depth[valid] - min_value) * 255.0 / denom, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_color[~valid] = 0
        return cv2.resize(depth_color, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _make_visualization_grid(
        self,
        result,
        bgr_image: np.ndarray,
        instances: List[InstanceMask],
        xyz_img: np.ndarray,
        heatmaps: List[np.ndarray],
        cluster_candidates: List[Dict],
        poses: List[SuctionPose],
    ) -> np.ndarray:
        cloud_shape = xyz_img.shape[:2]
        result_img = result.plot(
            img=bgr_image,
            conf=True,
            labels=True,
            boxes=True,
            masks=True,
            line_width=2,
        )
        heatmap_overlay = self._make_heatmap_overlay(bgr_image, heatmaps, cloud_shape)
        mask_vis = self._make_mask_visualization(bgr_image, instances, cloud_shape)
        depth_vis = self._make_depth_visualization(xyz_img, bgr_image.shape[:2])
        cluster_vis = self._make_cluster_overlay(bgr_image, cluster_candidates, cloud_shape)
        tag_loc = self._make_overlay(result, bgr_image, poses, cloud_shape)

        panels = [
            self._label_debug_image(result_img, "YOLO Detection"),
            self._label_debug_image(heatmap_overlay, "Heatmap Overlay"),
            self._label_debug_image(mask_vis, "Object Mask"),
            self._label_debug_image(depth_vis, "Depth Map"),
            self._label_debug_image(cluster_vis, "Cluster Visualization"),
            self._label_debug_image(tag_loc, "Tag Location"),
        ]
        top = np.hstack(panels[0:2])
        middle = np.hstack(panels[2:4])
        bottom = np.hstack(panels[4:6])
        grid = np.vstack([top, middle, bottom])

        max_height = 1200
        max_width = 2200
        height, width = grid.shape[:2]
        if height > max_height or width > max_width:
            scale = min(float(max_height) / float(height), float(max_width) / float(width))
            grid = cv2.resize(grid, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return grid

    @staticmethod
    def _apply_red_segmentation_masks(result, bgr_image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        overlay = bgr_image.copy()
        if result.masks is None or result.masks.data is None:
            return overlay

        masks = result.masks.data.detach().cpu().numpy()
        if masks.ndim != 3:
            return overlay

        image_shape = bgr_image.shape[:2]
        red_layer = np.zeros_like(overlay)
        red_layer[:, :] = (0, 0, 255)

        combined_mask = np.zeros(image_shape, dtype=bool)
        for raw_mask in masks:
            mask = raw_mask > 0.5
            if mask.shape != image_shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            combined_mask |= mask

        if np.any(combined_mask):
            blended = cv2.addWeighted(overlay, 1.0 - alpha, red_layer, alpha, 0)
            overlay[combined_mask] = blended[combined_mask]
        return overlay

    def _make_overlay(
        self,
        result,
        bgr_image: np.ndarray,
        poses: List[SuctionPose],
        cloud_shape: Tuple[int, int],
    ) -> np.ndarray:
        red_mask_overlay = self._apply_red_segmentation_masks(result, bgr_image)
        overlay = result.plot(
            img=red_mask_overlay,
            conf=True,
            labels=True,
            boxes=True,
            masks=False,
            line_width=2,
        )
        overlay = np.ascontiguousarray(overlay)

        for pose in poses:
            x, y = self._scale_point(pose.center_px, cloud_shape, bgr_image.shape[:2])
            color = INSTANCE_COLORS_BGR[pose.object_id % len(INSTANCE_COLORS_BGR)]
            cv2.circle(overlay, (x, y), 6, color, -1)
            cv2.circle(overlay, (x, y), 9, (255, 255, 255), 2)
            label = f"{pose.object_id}:{pose.cluster_id}:{pose.suction_score:.2f}"
            cv2.putText(
                overlay,
                label,
                (x + 8, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _publish_bgr_image(self, publisher, image: np.ndarray, header: Header) -> None:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = np.ascontiguousarray(image)
        msg = Image()
        msg.header = header
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(msg.width) * 3
        msg.data = image.tobytes()
        publisher.publish(msg)

    def _log_callback_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self.last_error_log_time > 2.0:
            self.get_logger().error(f"Failed to process synced ZED data: {type(exc).__name__}: {exc}")
            self.last_error_log_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ZedSuctionPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
