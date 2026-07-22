"""Geometry utilities that turn suction candidates into 6D poses."""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .models import InstanceMask, SuctionPose


class PoseGeometryMixin:
    """Methods for local surface geometry and quaternion construction."""

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
        # Prefer the candidate plane/cluster over the full YOLO object mask.
        # This makes Detection3D.bbox.size describe the suction plane footprint,
        # not the entire segmented object.
        candidate_valid = candidate.get("cluster_valid", instance.mask & valid_idx)
        points = xyz_img[candidate_valid]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] < self.min_valid_points:
            return None

        pose_frame = self._estimate_candidate_pose_frame(xyz_img, candidate_valid, candidate)
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

        # The OBB is computed in the local tangent plane, then lifted back to 3D.
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
        # Rotation columns are the suction local X/Y/Z axes in the cloud frame.
        # Dotting points with these axes gives the extents in suction-frame
        # coordinates. Translation is irrelevant for min/max range.
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
