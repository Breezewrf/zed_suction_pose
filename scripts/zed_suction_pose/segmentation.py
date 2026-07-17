"""YOLO result parsing into binary instance masks."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .models import InstanceMask


class SegmentationMixin:
    """Methods that convert model outputs into masks in point-cloud resolution."""

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
            mask_img = (raw_mask > 0.5).astype(np.uint8)
            if mask_img.shape != image_shape:
                mask_img = cv2.resize(
                    mask_img,
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = cv2.resize(
                mask_img,
                (cloud_w, cloud_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            if self.cloud_mask_dilate_px > 0:
                kernel_size = 2 * self.cloud_mask_dilate_px + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            if int(np.count_nonzero(mask)) < self.min_mask_area_px:
                continue

            class_id = 0
            yolo_score = 1.0
            if boxes is not None and idx < len(boxes):
                class_id = int(boxes.cls[idx].item()) if boxes.cls is not None else 0
                yolo_score = float(boxes.conf[idx].item()) if boxes.conf is not None else 1.0

            class_name = str(names.get(class_id, class_id))
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

