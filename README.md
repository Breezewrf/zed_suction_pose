# zed_suction_pose

Runs YOLO segmentation on ZED RGB images, filters the organized ZED point cloud
by each instance mask, evaluates planar suction regions with Open3D normal
consistency, and publishes all valid 6D suction poses.

Default inputs:

- `/zed/zed_node/rgb/color/rect/image`
- `/zed/zed_node/point_cloud/cloud_registered`
- `/zed/zed_node/rgb/color/rect/camera_info`

Default outputs:

- `/suction_poses` (`geometry_msgs/PoseArray`)
- `/suction_detections` (`vision_msgs/Detection3DArray`)
- `/suction_markers` (`visualization_msgs/MarkerArray`)
- `/suction_debug/overlay` (`sensor_msgs/Image`)
- `/suction_debug/masked_cloud` (`sensor_msgs/PointCloud2`)

Run with an existing ZED node:

```bash
source /home/breeze/Desktop/workplace/ultralytics/.venv/bin/activate
source install/local_setup.bash
ros2 launch zed_suction_pose zed_suction_pose.launch.py
```

Run ZED, suction pose detection, and RViz together:

```bash
source /home/breeze/Desktop/workplace/ultralytics/.venv/bin/activate
source install/local_setup.bash
ros2 launch zed_suction_pose zed_suction_with_camera.launch.py camera_model:=zed2i
```

The combined launch also starts a FastAPI server on port `4444`:

```bash
curl http://aisys-jetson-4.local:4444/cv/ecommerce/items
```

Open the live detection overlay in a browser at:

```text
http://aisys-jetson-4.local:4444/cv/ecommerce/items-vis
```

This endpoint streams `/suction_debug/overlay` as MJPEG.
Only clusters selected as final suction poses are overlaid. Their `ID N` labels
correspond directly to element `N` in the `/cv/ecommerce/items` response.

The response is an array ordered by confidence, so index `0` is the preferred
pick. An empty detection result is returned as `[]`. Positions and extents are
in meters; `rx`, `ry`, and `rz` are XYZ Euler angles in degrees. The surface
length is always `extent_x`, the width is `extent_y`, and the pose local Z axis
is the surface approach direction.
