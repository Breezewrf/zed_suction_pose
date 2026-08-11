#!/usr/bin/env python3

"""ROS entry point for ZED image segmentation and suction pose estimation."""

import os
import time
from typing import Dict, List, Optional, Tuple

import message_filters
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from ultralytics import YOLO
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from zed_suction_pose.io_utils import ImageCloudMixin
from zed_suction_pose.models import CameraIntrinsics
from zed_suction_pose.pose_geometry import PoseGeometryMixin
from zed_suction_pose.publishing import PublishingMixin
from zed_suction_pose.segmentation import SegmentationMixin
from zed_suction_pose.suction_processing import SuctionProcessingMixin
from zed_suction_pose.utils import (
    as_unit_vector,
    default_model_path,
    make_odd,
    parse_axis,
    parse_classes,
)
from zed_suction_pose.visualization import VisualizationMixin


class ZedSuctionPoseNode(
    ImageCloudMixin,
    SegmentationMixin,
    SuctionProcessingMixin,
    PoseGeometryMixin,
    PublishingMixin,
    VisualizationMixin,
    Node,
):
    """Coordinate ROS I/O, YOLO segmentation, suction geometry, and visualization."""

    def __init__(self) -> None:
        super().__init__("zed_suction_pose")

        default_model = default_model_path()
        self._declare_parameters(default_model)
        self._read_parameters(default_model)

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.model_path}")

        self.model = YOLO(self.model_path, task="segment")
        self.latest_camera_info: Optional[CameraIntrinsics] = None
        self.frame_count = 0
        self.last_process_time = 0.0
        self.last_error_log_time = 0.0
        self.last_pose_tf_warn_time = 0.0
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
        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        self.declare_parameter("pose_array_frame", "zed_left_camera_frame_optical")
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
        self.declare_parameter("max_surface_tilt_deg", 30.0)
        self.declare_parameter("cluster_opening_px", 3)
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
        self.pose_array_frame = self.get_parameter("pose_array_frame").value
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
        self.classes = parse_classes(self.get_parameter("classes").value)
        self.retina_masks = bool(self.get_parameter("retina_masks").value)
        self.verbose = bool(self.get_parameter("verbose").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.process_every_n_frames = max(1, int(self.get_parameter("process_every_n_frames").value))
        self.target_fps = float(self.get_parameter("target_fps").value)
        self.log_timing = bool(self.get_parameter("log_timing").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_axis_index = parse_axis(self.get_parameter("depth_axis").value)
        self.depth_axis_name = "xyz"[self.depth_axis_index]
        self.min_mask_area_px = int(self.get_parameter("min_mask_area_px").value)
        self.mask_downsample_min_coverage = float(
            self.get_parameter("mask_downsample_min_coverage").value
        )
        self.mask_downsample_min_coverage = float(np.clip(self.mask_downsample_min_coverage, 0.0, 1.0))
        self.cloud_mask_dilate_px = max(0, int(self.get_parameter("cloud_mask_dilate_px").value))
        self.min_valid_points = int(self.get_parameter("min_valid_points").value)
        self.downsample_scale = max(1, int(self.get_parameter("downsample_scale").value))
        self.knn = max(3, int(self.get_parameter("knn").value))
        self.std_window = make_odd(int(self.get_parameter("std_window").value))
        self.dynamic_std_window = bool(self.get_parameter("dynamic_std_window").value)
        self.std_window_min = make_odd(int(self.get_parameter("std_window_min").value))
        self.std_window_max = make_odd(int(self.get_parameter("std_window_max").value))
        if self.std_window_max < self.std_window_min:
            self.std_window_max = self.std_window_min
        self.std_window_mask_ratio = max(0.0, float(self.get_parameter("std_window_mask_ratio").value))
        self.std_window_max_mask_fraction = float(
            np.clip(float(self.get_parameter("std_window_max_mask_fraction").value), 0.05, 1.0)
        )
        self.max_surface_tilt_deg = float(
            np.clip(float(self.get_parameter("max_surface_tilt_deg").value), 0.0, 90.0)
        )
        self.surface_normal_min_alignment = float(
            np.cos(np.deg2rad(self.max_surface_tilt_deg))
        )
        self.cluster_opening_px = max(0, int(self.get_parameter("cluster_opening_px").value))
        if self.cluster_opening_px > 1 and self.cluster_opening_px % 2 == 0:
            self.cluster_opening_px += 1
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
        self.normal_orientation = as_unit_vector(
            self.get_parameter("normal_orientation").value,
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.invert_normal_for_pose = bool(self.get_parameter("invert_normal_for_pose").value)
        self.pose_reference_axis = as_unit_vector(
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
        self.pose_axis_marker_length_m = max(0.0, float(self.get_parameter("pose_axis_marker_length_m").value))
        self.pose_footprint_thickness_m = max(0.001, float(self.get_parameter("pose_footprint_thickness_m").value))
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

            # Debug images are intentionally outside the core publish timing bucket.
            if self.overlay_pub:
                overlay = self._make_overlay(
                    result,
                    bgr_image,
                    poses,
                    cluster_candidates,
                    xyz_img.shape[:2],
                )
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
                    "input": after_input - callback_start,
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
