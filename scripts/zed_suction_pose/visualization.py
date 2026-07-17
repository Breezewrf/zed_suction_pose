"""Debug image and overlay rendering."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .constants import CLUSTER_COLORS_BGR, INSTANCE_COLORS_BGR
from .models import InstanceMask, SuctionPose


class VisualizationMixin:
    """Methods that render RViz image overlays for debugging."""

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
        for instance in instances:
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

