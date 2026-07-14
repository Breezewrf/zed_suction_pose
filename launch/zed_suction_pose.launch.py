import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_model_path():
    try:
        return os.path.join(
            get_package_share_directory("zed_suction_pose"),
            "checkpoints",
            "best.pt",
        )
    except Exception:
        return ""


def generate_launch_description():
    pkg_share = get_package_share_directory("zed_suction_pose")
    default_config = os.path.join(pkg_share, "config", "zed_suction_pose.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("model_path", default_value=_default_model_path()),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/zed/zed_node/rgb/color/rect/image",
            ),
            DeclareLaunchArgument(
                "pointcloud_topic",
                default_value="/zed/zed_node/point_cloud/cloud_registered",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/zed/zed_node/rgb/color/rect/camera_info",
            ),
            Node(
                package="zed_suction_pose",
                executable="zed_suction_pose_node.py",
                name="zed_suction_pose",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                        "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                    },
                ],
            ),
        ]
    )
