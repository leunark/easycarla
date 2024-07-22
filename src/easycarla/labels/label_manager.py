import carla
import numpy as np
import logging
from dataclasses import dataclass

from easycarla.sensors import Sensor, CameraSensor, DepthCameraSensor, LidarSensor, InsegCameraSensor
from easycarla.labels.label_data import LabelData
from easycarla.labels.label_types import ObjectType, map_carla_to_kitti, map_kitti_to_carla
from easycarla.labels.calibration_data import CalibrationData


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
        self.hero: carla.Actor = None
        
        self.env_labels: LabelData = None
        self.actor_labels: LabelData = None
        
        self.ego_vehicle = camera_sensor.actor.parent

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
        
    def update(self):
        # First retrieve labels in world space
        #if self.env_labels is None:
        #    self.env_labels = self.create_labels(self.env_objects)
        self.actor_labels = self.create_labels(self.actors)
        
        # Now, combine them together as we don't differentiate in moving state
        labels = self.actor_labels
        if len(labels) == 0:
            return
        
        # Vectorize hero transform
        sensor_pos, sensor_forward = self.camera_sensor.get_transform()

        # Filter in 3d world space
        labels.filter_by_distance(sensor_pos, 200)
        if len(labels) == 0:
            return None
        
        labels.filter_by_direction(sensor_pos, sensor_forward, 0.2)
        if len(labels) == 0:
            return None
        
        # Transform to sensor coordinate system
        world_to_camera = self.camera_sensor.get_world_to_actor()
        labels.apply_transform(world_to_camera)
        
        # Project edges onto sensor
        bbs = self.camera_sensor.project(labels.vertices)
        
        # Read inseg sensor data
        # Calculate number of vertices infront per bounding box
        # Filter based on a threshold
        depth_data = self.depth_sensor.decoded_data

        # Retrieve all edges from bounding boxes
        bbs = bbs[:, labels.EDGE_INDICES]

        return bbs

    @staticmethod
    def get_depth_at_point(depth_image_array, point):
        """Retrieve depth value at the specified point from the depth image array."""
        x, y = int(point[0, 0]), int(point[0, 1])
        
        if 0 <= x < depth_image_array.shape[1] and 0 <= y < depth_image_array.shape[0]:
            # Convert the depth image value to meters
            depth_value = int(depth_image_array[y, x][0]) + int(depth_image_array[y, x][1]) * 256 + int(depth_image_array[y, x][2]) * 256 * 256
            normalized_depth = depth_value / (256 ** 3 - 1)  # Normalize to range [0, 1]
            depth_in_meters = normalized_depth * 1000.0  # Assuming 0-1 maps to 0-1000 meters
            
            return depth_in_meters
        return None

    @staticmethod
    def is_occluded(bounding_box, depth_image_array):
        """Determine if the bounding box is occluded based on depth image array."""
        num_occluded_points = 0
        for point in bounding_box:
            x, y, depth = int(point[0, 0]), int(point[0, 1]), point[0, 2]
            depth_at_point = depth_image_array[y, x]
            if depth >= depth_at_point + 0.1:
                num_occluded_points += 1
        if num_occluded_points > 6:
            return True
        return False
    
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
