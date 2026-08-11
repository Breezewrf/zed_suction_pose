"""Depth/normal based suction candidate extraction."""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from .constants import CLUSTER_COLORS_BGR, INSTANCE_COLORS_BGR
from .models import InstanceMask, SuctionPose
from .utils import make_odd_bounded


class SuctionProcessingMixin:
    """Methods that turn masks and organized point clouds into suction candidates."""

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
                if pose is not None:
                    poses.append(pose)

        # This order is the public /items order. Publishing and debug rendering
        # must both preserve it so overlay ID N always identifies /items[N].
        poses.sort(key=lambda pose: pose.suction_score * pose.yolo_score, reverse=True)
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
        valid_mask = mask_ds & finite & (depth > self.min_depth_m) & (depth < self.max_depth_m)

        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < self.min_valid_points:
            return None, None, None

        # Normals are estimated only within the mask, then expanded back into an image map.
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

        return make_odd_bounded(
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

        normal_norm = np.linalg.norm(normal_map, axis=2)
        normal_alignment = np.zeros_like(normal_norm, dtype=np.float32)
        normal_dot = np.abs(normal_map @ self.normal_orientation.astype(np.float32))
        np.divide(
            normal_dot,
            np.maximum(normal_norm, 1e-9),
            out=normal_alignment,
            where=normal_norm > 1e-6,
        )
        surface_valid = object_valid & (
            normal_alignment >= self.surface_normal_min_alignment
        )

        heatmap_binary = (
            (heatmap >= self.heatmap_threshold) & surface_valid
        ).astype(np.uint8)
        if self.cluster_opening_px > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cluster_opening_px, self.cluster_opening_px),
            )
            heatmap_binary = cv2.morphologyEx(
                heatmap_binary,
                cv2.MORPH_OPEN,
                kernel,
            )
            heatmap_binary[~surface_valid] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(heatmap_binary, connectivity=8)
        if num_labels <= 1:
            return []

        candidates: List[Dict] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_cluster_area_px:
                continue

            cluster_valid = (labels == label) & surface_valid
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

            candidates.append(
                {
                    "instance_index": instance.index,
                    "cluster_id": label,
                    "score": score,
                    "center_px": (int(center_x), int(center_y)),
                    "position": point.astype(np.float64),
                    "normal": normal.astype(np.float64),
                    "normal_alignment": float(abs(np.dot(normal, self.normal_orientation))),
                    "cluster_valid": cluster_valid,
                    "patch_size": (int(patch_w), int(patch_h)),
                    "is_fallback": False,
                }
            )

        return candidates

    @staticmethod
    def _select_best_candidate(candidates: List[Dict]) -> Optional[Dict]:
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item["normal_alignment"], item["score"]))

    def _select_pose_candidates(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        if not self.multi_pose_per_mask:
            best = self._select_best_candidate(candidates)
            return [best] if best is not None else []

        ranked = sorted(
            candidates,
            key=lambda item: (item["normal_alignment"], item["score"]),
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
