cmake_minimum_required(VERSION 3.8)
project(zed_suction_pose)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/zed_suction_pose_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  checkpoints
  launch
  rviz2
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
