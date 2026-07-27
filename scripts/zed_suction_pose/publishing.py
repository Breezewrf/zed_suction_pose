"""ROS message publishing and message builders."""

import time
from typing import List, Tuple

import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import Point, Pose, TransformStamped
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .constants import INSTANCE_COLORS_BGR
from .models import SuctionPose


class PublishingMixin:
    """Methods that build and publish ROS messages."""

    def _publish_results(
        self,
        header: Header,
        poses: List[SuctionPose],
        debug_points: List[Tuple[np.ndarray, Tuple[int, int, int]]],
        cluster_points: List[Tuple[np.ndarray, Tuple[int, int, int]]],
    ) -> None:
        if self.pose_pub:
            pose_array = self._make_pose_array(header, poses)
            if pose_array is not None:
                self.pose_pub.publish(pose_array)
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

    def _make_pose_array(self, header: Header, poses: List[SuctionPose]):
        from geometry_msgs.msg import PoseArray

        output_header = Header()
        output_header.stamp = header.stamp
        output_header.frame_id = self.pose_array_frame

        msg = PoseArray()
        msg.header = output_header

        if header.frame_id == self.pose_array_frame:
            msg.poses = [self._pose_msg(pose) for pose in poses]
            return msg

        try:
            transform = self.tf_buffer.lookup_transform(
                self.pose_array_frame,
                header.frame_id,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.1),
            )
        except tf2_ros.TransformException as exc:
            now = time.monotonic()
            if now - self.last_pose_tf_warn_time > 2.0:
                self.get_logger().warn(
                    f"Cannot transform /suction_poses from '{header.frame_id}' "
                    f"to '{self.pose_array_frame}': {exc}"
                )
                self.last_pose_tf_warn_time = now
            return None

        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        transform_rotation = Rotation.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        )

        msg.poses = []
        for pose in poses:
            transformed = self._pose_msg(pose)
            position = transform_rotation.apply(pose.position) + translation
            orientation = (
                transform_rotation * Rotation.from_quat(pose.orientation_xyzw)
            ).as_quat()
            transformed.position.x = float(position[0])
            transformed.position.y = float(position[1])
            transformed.position.z = float(position[2])
            transformed.orientation.x = float(orientation[0])
            transformed.orientation.y = float(orientation[1])
            transformed.orientation.z = float(orientation[2])
            transformed.orientation.w = float(orientation[3])
            msg.poses.append(transformed)
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

            # The footprint cube is oriented by the full 6D pose, so its X/Y axes
            # show the in-plane suction frame in RViz.
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
