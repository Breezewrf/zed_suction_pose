"""ROS Image and PointCloud2 conversion helpers."""

import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2


class ImageCloudMixin:
    """Methods that convert ROS transport messages into numpy arrays."""

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

    @staticmethod
    def _pointcloud_msg_to_xyz_image(msg: PointCloud2) -> np.ndarray:
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

