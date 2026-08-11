import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    zed_wrapper_share = get_package_share_directory("zed_wrapper")
    default_config = os.path.join(pkg_share, "config", "zed_suction_pose.yaml")
    default_rviz = os.path.join(pkg_share, "rviz2", "zed_suction_pose.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_name", default_value="zed"),
            DeclareLaunchArgument("camera_model"),
            DeclareLaunchArgument("start_zed_node", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("http_api", default_value="true"),
            DeclareLaunchArgument("http_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("http_port", default_value="4444"),
            DeclareLaunchArgument("http_overlay_topic", default_value="/suction_debug/overlay"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("model_path", default_value=_default_model_path()),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(zed_wrapper_share, "launch", "zed_camera.launch.py")
                ),
                launch_arguments={
                    "camera_name": LaunchConfiguration("camera_name"),
                    "camera_model": LaunchConfiguration("camera_model"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("start_zed_node")),
            ),
            Node(
                package="zed_suction_pose",
                executable="zed_suction_pose_node.py",
                name="zed_suction_pose",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {"model_path": LaunchConfiguration("model_path")},
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="zed_suction_pose_rviz2",
                output="screen",
                arguments=["-d", default_rviz],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            Node(
                package="zed_suction_pose",
                executable="zed_suction_http_api.py",
                name="zed_suction_http_api",
                output="screen",
                parameters=[
                    {
                        "http_host": LaunchConfiguration("http_host"),
                        "http_port": LaunchConfiguration("http_port"),
                        "overlay_topic": LaunchConfiguration("http_overlay_topic"),
                    }
                ],
                condition=IfCondition(LaunchConfiguration("http_api")),
            ),
        ]
    )
