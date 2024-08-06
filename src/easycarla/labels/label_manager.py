import carla
import numpy as np
from dataclasses import dataclass
import cv2 
from pathlib import Path

from easycarla.labels.label_data import LabelData
from easycarla.labels.label_types import map_carla_to_kitti, map_kitti_to_carla
from easycarla.labels.kitti import KITTIDatasetGenerator
from easycarla.sensors import CameraSensor, DepthCameraSensor, LidarSensor, WorldSensor
from easycarla.tf import Transformation

class LabelManager:
    """
    Class to manage labels for objects in the CARLA simulation.

    Attributes:
        carla_types (set[carla.CityObjectLabel]): Set of CARLA object types to label.
        world_sensor (WorldSensor): World sensor object.
        camera_sensor (CameraSensor): Camera sensor object.
        depth_sensor (DepthCameraSensor): Depth camera sensor object.
        lidar_sensor (LidarSensor): LiDAR sensor object.
        distance (float): Maximum distance for labels.
        show_points (bool): Whether to show points in visualization.
        output_dir (Path|str): Directory to save output data.
        frame_interval (int): Interval of frames to process.
        frame_count (int): Number of frames to process.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
    """
    def __init__(self, 
                 carla_types: set[carla.CityObjectLabel],
                 world_sensor: WorldSensor,
                 camera_sensor: CameraSensor,
                 depth_sensor: DepthCameraSensor,
                 lidar_sensor: LidarSensor,
                 distance: float = 50,
                 show_points: bool = False, 
                 output_dir: Path|str = None,
                 frame_interval: int = 0,
                 frame_count = 1000, 
                 train_ratio = 0.7, 
                 val_ratio = 0.15, 
                 test_ratio = 0.15) -> None:
        self.carla_types = carla_types
        self.world_sensor = world_sensor
        self.world = world_sensor.world
        self.camera_sensor = camera_sensor
        self.depth_sensor = depth_sensor
        self.lidar_sensor = lidar_sensor
        self.distance = distance
        self.show_points = show_points
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self.env_objects: list[carla.EnvironmentObject] = []
        self.actors: list[carla.Actor] = []
        self.ego_vehicle: carla.Actor = camera_sensor.actor.parent
        
        self.env_labels: LabelData = None
        self.actor_labels: LabelData = None
        self.labels: LabelData = None
        
        world_to_camera = self.camera_sensor.get_world_to_actor()
        lidar_to_world = self.lidar_sensor.get_actor_to_world()
        self.lidar_to_camera = world_to_camera @ lidar_to_world

        if self.output_dir is not None:
            self.kitti = KITTIDatasetGenerator(self.output_dir, frame_interval, frame_count, train_ratio, val_ratio, test_ratio)
            self.kitti.set_calibration(P2=self.camera_sensor.calibration, Tr_velo_to_cam=self.lidar_to_camera)

        self.bbse2d: np.ndarray = None
        self.pc2d: np.ndarray = None

        self.init_objects()

    def init_objects(self):
        """Initialize environment objects and actors."""
        # Retrieve and store bounding boxes for environmental objects
        for carla_type in self.carla_types:
            self.env_objects.extend(self.world.get_environment_objects(carla_type))

        # Retrieve all relevant actors
        actor_list = self.world.get_actors()
        self.actors = ([a for a in actor_list.filter("walker.*")] + 
                       [a for a in actor_list.filter("vehicle.*")]) 
        
        # Remove ego vehicle from actor list
        if self.ego_vehicle is not None:
            self.actors = [a for a in self.actors if a.id != self.ego_vehicle.id]

    def create_env_labels(self) -> LabelData:
        """
        Create labels for environment objects.

        Returns:
            LabelData: Label data for environment objects.
        """
        env_objects = self.env_objects
        # Extract transformations and types
        id_list = np.array([int(obj.id) for obj in env_objects])

        # Extract transformations and types
        transform_list = np.array([obj.transform.get_matrix() for obj in env_objects])
        
        # Extract dimensions
        dimension_list = np.array([
            (obj.bounding_box.extent.x * 2, obj.bounding_box.extent.y * 2, obj.bounding_box.extent.z * 2)
            for obj in env_objects
        ])
        
        # Extract types
        types_list = [{map_carla_to_kitti(obj.type)} for obj in env_objects]
        
        # Adjust the z translation of the transform matrix
        loc_z_list = [obj.bounding_box.location.z for obj in env_objects]
        transform_list[:, 2, 3] += loc_z_list
        
        return LabelData(
            id=id_list,
            transform=transform_list,
            dimension=dimension_list,
            types=types_list
        )

    def create_actor_labels(self) -> LabelData:
        """
        Create labels for actors.

        Returns:
            LabelData: Label data for actors.
        """
        id = []
        transform = [] 
        dimension = []
        types = []
        location = []
        for actor in self.actors:
            if not actor.is_alive or not actor.is_active:
                continue
            snapshot = self.world_sensor.sensor_data.find(actor.id)
            if snapshot is None:
                continue
            id.append(snapshot.id)
            transform.append(snapshot.get_transform().get_matrix())
            dimension.append((actor.bounding_box.extent.x * 2, actor.bounding_box.extent.y * 2, actor.bounding_box.extent.z * 2))
            types.append({map_carla_to_kitti(tag) for tag in actor.semantic_tags})
            location.append((actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z))
        id = np.array(id)
        transform = np.array(transform) 
        dimension = np.array(dimension)
        types = np.array(types)
        location = np.array(location)

        # Adjust the z translation of the transform matrix
        transform[:, 2, 3] += location[:, 2]
        
        return LabelData(
            id=id,
            transform=transform,
            dimension=dimension,
            types=types
        )
    
    def update(self):
        """Update labels and generate dataset if needed."""
        # First retrieve labels in world space
        if self.env_labels is None:
            self.env_labels = self.create_env_labels()
        self.actor_labels = self.create_actor_labels()
        
        # Now, combine them together as we don't differentiate in moving state
        labels = self.actor_labels + self.env_labels
        if len(labels) == 0:
            return
        
        # Filter initially in world space by distance to reduce bounding boxes
        lidar_pos, lidar_forward = self.lidar_sensor.get_transform()
        labels = labels.filter_by_distance(distance=self.distance, target=lidar_pos)
        if len(labels) == 0:
            return

        # Transform to sensor coordinate system
        world_to_camera = self.camera_sensor.get_world_to_actor()
        labels = labels.apply_transform(world_to_camera)

        # Project vertices onto sensor
        bbs3d = labels.vertices
        num_of_vertices = bbs3d.shape[1]
        bbsv2d, bbsv2d_depth = self.camera_sensor.project(bbs3d)

        # Calculate truncation with projection
        mask_within_image = (bbsv2d[:, :, 0] >= 0) & (bbsv2d[:, :, 0] < self.camera_sensor.image_size[0]) & \
                            (bbsv2d[:, :, 1] >= 0) & (bbsv2d[:, :, 1] < self.camera_sensor.image_size[1]) & \
                            (bbsv2d_depth > 0)
        num_of_verts_outside_image = np.count_nonzero(np.invert(mask_within_image), axis=1)
        labels.truncation = num_of_verts_outside_image / num_of_vertices

        # Remove fully truncated bounding boxes
        mask_not_fully_truncated = labels.truncation < 1.0
        labels = labels.filter(mask_not_fully_truncated)
        bbsv2d = bbsv2d[mask_not_fully_truncated]
        bbsv2d_depth = bbsv2d_depth[mask_not_fully_truncated]
        mask_within_image = mask_within_image[mask_not_fully_truncated]

        # Calculate occlusion with depth data
        depth_values = self.depth_sensor.depth_values
        bbs_occluded_depth = np.ones_like(bbsv2d_depth)
        bbs_pos_within_image = bbsv2d[mask_within_image]
        depth_image_coords = bbs_pos_within_image.T[::-1]
        bbs_pos_within_image_depth = depth_values[depth_image_coords[0], depth_image_coords[1]]
        bbs_occluded_depth[mask_within_image] = bbs_pos_within_image_depth
        bbs_verts_occluded = bbsv2d_depth > bbs_occluded_depth + 0.1
        num_of_verts_occluded = np.count_nonzero(bbs_verts_occluded, axis=1)
        labels.occlusion = num_of_verts_occluded / num_of_vertices

        # Remove fully occluded bounding boxes
        mask_not_fully_occluded = labels.occlusion < 1.0
        labels = labels.filter(mask_not_fully_occluded)
        bbsv2d = bbsv2d[mask_not_fully_occluded]
        bbsv2d_depth = bbsv2d_depth[mask_not_fully_occluded]

        # Now, we have our final label data
        self.labels = labels

        # Generate dataset
        if self.output_dir is not None:
            timestamp = self.world_sensor.sensor_data.timestamp.elapsed_seconds 
            frame = self.world_sensor.sensor_data.frame
            self.kitti.process_frame(pointcloud=self.lidar_sensor.pointcloud, 
                                    image=self.camera_sensor.rgb_image, 
                                    depth_image=self.depth_sensor.rgb_image, 
                                    labels=self.labels,
                                    timestamp=timestamp,
                                    world_frame_id=frame)

        # Retrieve pointcloud & draw onto image
        if self.show_points:
            pointcloud = self.lidar_sensor.pointcloud
            pc3d = pointcloud[:, :3]
            pc3d = Transformation.transform_with_matrix(pc3d, self.lidar_to_camera)
            pc2d, pc2d_depth = self.camera_sensor.project(pc3d)
            mask_within_image = (pc2d[:, 0] >= 0) & (pc2d[:, 0] < self.camera_sensor.image_size[0]) & \
                                (pc2d[:, 1] >= 0) & (pc2d[:, 1] < self.camera_sensor.image_size[1]) & \
                                (pc2d_depth > 0)
            pc2d = pc2d[mask_within_image]
            self.pc2d = pc2d
            for point in pc2d:
                cv2.circle(self.camera_sensor.preview_image, point, 1, (200, 200, 200), 1)

        # Filter labels for visualization
        mask_truncation_threshold = labels.truncation < 0.95
        mask_occlusion_threshold = labels.occlusion < 0.95
        mask = mask_truncation_threshold & mask_occlusion_threshold
        bbsv2d_depth = bbsv2d_depth[mask]
        bbsv2d = bbsv2d[mask]

        # Retrieve edges from the vertices of the bounding boxes
        bbse2d = bbsv2d[:, labels.EDGE_INDICES]

        # Draw edges within image boundaries
        image_width, image_height = self.camera_sensor.image_size
        if bbse2d is None or bbse2d.shape[0] == 0:
            return
        for bb in bbse2d:
            for edge in bb:
                p1, p2 = edge
                ret, p1, p2 = cv2.clipLine((0, 0, image_width, image_height), p1.astype(int), p2.astype(int))
                if ret:
                    cv2.line(self.camera_sensor.preview_image, p1, p2, (0, 0, 255), 1)

        self.bbse2d = bbse2d

    def export(self):
        pass
    