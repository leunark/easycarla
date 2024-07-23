import carla
import numpy as np
import logging
from dataclasses import dataclass
import cv2 

from easycarla.sensors import Sensor, CameraSensor, DepthCameraSensor, LidarSensor, InsegCameraSensor
from easycarla.labels.label_data import LabelData, ObjectType
from easycarla.labels.label_types import map_carla_to_kitti, map_kitti_to_carla
from easycarla.labels.calib_data import CalibrationData
from easycarla.labels.kitti import KITTIDatasetGenerator
from easycarla.tf import Transformation

class LabelManager:

    def __init__(self, 
                 world: carla.World,
                 carla_types: set[carla.CityObjectLabel],
                 camera_sensor: CameraSensor,
                 depth_sensor: DepthCameraSensor,
                 lidar_sensor: LidarSensor) -> None:
        self.world = world
        self.carla_types = carla_types
        self.camera_sensor = camera_sensor
        self.depth_sensor = depth_sensor
        self.lidar_sensor = lidar_sensor

        self.env_objects: list[carla.EnvironmentObject] = []
        self.actors: list[carla.Actor] = []
        self.ego_vehicle: carla.Actor = camera_sensor.actor.parent
        
        self.env_labels: LabelData = None
        self.actor_labels: LabelData = None
        self.labels: LabelData = None
        
        world_to_camera = self.camera_sensor.get_world_to_actor()
        lidar_to_world = self.lidar_sensor.get_actor_to_world()
        self.lidar_to_camera = world_to_camera @ lidar_to_world

        self.kitti = KITTIDatasetGenerator("data/kitti")
        self.kitti.set_calibration(P2=self.camera_sensor.calibration, Tr_velo_to_cam=self.lidar_to_camera)

        self.init_bbs()

    def init_bbs(self):
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
        
    def create_labels(self, label_objects: list[carla.EnvironmentObject | carla.Actor]) -> LabelData:
        # Extract transformations and types
        id_list = np.array([int(obj.id) for obj in label_objects])

        # Extract transformations and types
        transform_list = np.array([
            (obj.get_transform().get_matrix() if isinstance(obj, carla.Actor) else obj.transform.get_matrix())
            for obj in label_objects
        ])
        
        # Extract dimensions
        dimension_list = np.array([
            (obj.bounding_box.extent.x * 2, obj.bounding_box.extent.y * 2, obj.bounding_box.extent.z * 2)
            for obj in label_objects
        ])
        
        # Extract types
        types_list = [
            ({map_carla_to_kitti(tag) for tag in obj.semantic_tags} if isinstance(obj, carla.Actor) else {map_carla_to_kitti(obj.type)})
            for obj in label_objects
        ]
        
        # Adjust the z translation of the transform matrix
        loc_z_list = [obj.bounding_box.location.z for obj in label_objects]
        transform_list[:, 2, 3] += loc_z_list
        
        return LabelData(
            id=id_list,
            transform=transform_list,
            dimension=dimension_list,
            types=types_list
        )
    
    def update(self):
        # First retrieve labels in world space
        if self.env_labels is None:
            self.env_labels = self.create_labels(self.env_objects)
        self.actor_labels = self.create_labels(self.actors)
        
        # Now, combine them together as we don't differentiate in moving state
        labels = self.actor_labels + self.env_labels
        if len(labels) == 0:
            return
        
        # Filter initially in world space by distance to reduce bounding boxes
        lidar_pos, lidar_forward = self.lidar_sensor.get_transform()
        labels.filter_by_distance(distance=self.lidar_sensor.range, target=lidar_pos)
        if len(labels) == 0:
            return None

        # Transform to sensor coordinate system
        world_to_camera = self.camera_sensor.get_world_to_actor()
        labels.apply_transform(world_to_camera)

        # Project edges onto sensor
        bbs3d = labels.vertices
        num_of_vertices = bbs3d.shape[1]
        bbs2d, bbs2d_depth = self.camera_sensor.project(bbs3d)

        # Calculate truncation
        mask_within_image = (bbs2d[:, :, 0] >= 0) & (bbs2d[:, :, 0] < self.camera_sensor.image_size[0]) & \
                            (bbs2d[:, :, 1] >= 0) & (bbs2d[:, :, 1] < self.camera_sensor.image_size[1]) & \
                            (bbs2d_depth > 0)
        num_of_verts_outside_image = np.count_nonzero(np.invert(mask_within_image), axis=1)
        labels.truncation = num_of_verts_outside_image / num_of_vertices
        
        # Calculate occlusion with depth data
        depth_values = self.depth_sensor.depth_values
        bbs_occluded_depth = np.ones_like(bbs2d_depth)
        bbs_pos_within_image = bbs2d[mask_within_image]
        depth_image_coords = bbs_pos_within_image.T[::-1]
        bbs_pos_within_image_depth = depth_values[depth_image_coords[0], depth_image_coords[1]]
        bbs_occluded_depth[mask_within_image] = bbs_pos_within_image_depth
        bbs_verts_occluded = bbs2d_depth > bbs_occluded_depth + 0.1
        num_of_verts_occluded = np.count_nonzero(bbs_verts_occluded, axis=1)
        labels.occlusion = num_of_verts_occluded / num_of_vertices
        mask_visible = labels.occlusion < 1.0
        labels.filter(mask_visible)
        bbs2d = bbs2d[mask_visible]
        bbs2d_depth = bbs2d_depth[mask_visible]

        # Calculate alpha in camera space
        labels.alpha = labels.get_alpha()

        # Now, we have our final data and we can generate our dataset
        self.labels = labels
        self.kitti.process_frame(self.lidar_sensor.pointcloud, 
                                 self.camera_sensor.image, 
                                 self.depth_sensor.image, 
                                 self.labels,
                                 self.camera_sensor.sensor_data.frame)

        # Retrieve pointcloud & draw onto image
        pointcloud = self.lidar_sensor.pointcloud
        points = pointcloud[:, :3]
        points = Transformation.transform_with_matrix(points, self.lidar_to_camera)
        points_proj, points_depth = self.camera_sensor.project(points)
        mask_within_image = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < self.camera_sensor.image_size[0]) & \
                            (points_proj[:, 1] >= 0) & (points_proj[:, 1] < self.camera_sensor.image_size[1]) & \
                            (points_depth > 0)
        points_proj = points_proj[mask_within_image]
        for point in points_proj:
            cv2.circle(self.camera_sensor.image_drawable, point, 1, (200, 200, 200), 1)

        # Filter based on truncation
        mask_truncation_threshold = labels.truncation < 0.9
        mask_occlusion_threshold = labels.occlusion < 0.9
        mask = mask_truncation_threshold & mask_occlusion_threshold
        bbs2d_depth = bbs2d_depth[mask]
        bbs2d = bbs2d[mask]

        # Retrieve all edges from bounding boxes
        bbs2d = bbs2d[:, labels.EDGE_INDICES]
        
        return bbs2d

    def export(self):
        pass
    