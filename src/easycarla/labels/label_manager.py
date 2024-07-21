import carla
import numpy as np
import logging

from easycarla.sensors import Sensor, CameraSensor, DepthCameraSensor, LidarSensor
from easycarla.labels.label_data import LabelData
from easycarla.labels.label_types import ObjectType, map_carla_to_kitti, map_kitti_to_carla
from easycarla.labels.calibration_data import CalibrationData

class LabelManager:

    def __init__(self, 
                 world: carla.World,
                 carla_types: set[carla.CityObjectLabel]) -> None:
        self.world = world
        self.carla_types = carla_types

        self.env_objects: list[carla.EnvironmentObject] = []
        self.actors: list[carla.Actor] = []
        self.hero: carla.Actor = None
        
        self.camera_sensors: list[tuple[CameraSensor, LabelData, CalibrationData]] = []
        self.lidar_sensors: list[tuple[LidarSensor, LabelData, CalibrationData]] = []
        self.target_sensor: CameraSensor= None

        self.env_labels: LabelData = None
        self.actor_labels: LabelData = None
        
        self.camera_sensors = []
        self.lidar_sensors = []

        self.init_bbs()

    def init_bbs(self):
        # Retrieve and store bounding boxes for environmental objects
        for carla_type in self.carla_types:
            self.env_objects.extend(self.world.get_environment_objects(carla_type))

        # First, retrieve all actors
        actor_list = self.world.get_actors()
        self.actors = ([a for a in actor_list.filter("walker.*")] + 
                       [a for a in actor_list.filter("vehicle.*")])      

    def add_sensor(self, sensor: Sensor, is_target: bool = False):
        if issubclass(type(sensor), CameraSensor):
            self.camera_sensors.append(sensor)
            if is_target:
                self.target_sensor = sensor
                # Remove parent object as ego object is typically not part of the dataset
                parent = sensor.actor.parent
                if parent is not None:
                    self.hero = parent
                    self.actors = [a for a in self.actors if a.id != parent.id]
                    logging.info(f"Remove parent actor with id {parent.id}")
                logging.info(f"Set target actor to {parent.id}")

        elif issubclass(type(sensor), LidarSensor):
            self.lidar_sensors.append(sensor)

        else:
            raise TypeError("Type is not supported in label manager")
        
    def update(self):
        if self.target_sensor is None:
            raise RuntimeError("One sensor must be configured as target")
        
        # First retrieve labels in world space
        if self.env_labels is None:
            self.env_labels = self.get_env_bbs()
        self.actor_labels = self.get_actor_bbs()
        
        # Now, combine them together as we don't differentiate in moving state
        labels = self.env_labels + self.actor_labels
        if len(labels) == 0:
            return
        
        # Vectorize hero transform
        sensor_pos, sensor_forward = self.target_sensor.get_transform()

        # Filter in 3d world space
        labels.filter_by_distance(sensor_pos, 50)
        if len(labels) == 0:
            return None
        
        labels.filter_by_direction(sensor_pos, sensor_forward, 0.1)
        if len(labels) == 0:
            return None
        
        # Transform to sensor coordinate system
        labels.transform = self.target_sensor.world_to_sensor(labels.transform)

        # Project edges onto sensor 
        bbs = self.target_sensor.project(labels.transform)

        # Retrieve all edges from bounding boxes
        bbs = self.get_bbs_edges(bbs)

        return bbs

    def create_labels(self, label_objects: list[carla.EnvironmentObject | carla.Actor]) -> LabelData: 
        for label_object in label_objects:
            location = label_object.transform.location
            location = location.x, location.y, location.z
            rotation = label_object.transform.rotation
            rotation = rotation.x, rotation.y, rotation.z
            extent = label_object.bounding_box.extent
            extent = extent.x, extent.y, extent.z


            
        

    def get_env_bbs(self) -> LabelData:
        bbox_3d = []
        object_types = []
        for env_object in self.env_objects:
            # As env objects are already in world coordinate system,
            # we don't need to provide an actor's transform
            vertices = env_object.bounding_box.get_world_vertices(carla.Transform())
            vertices = [(v.x, v.y, v.z) for v in vertices]
            bbox_3d.append(vertices)

            # Retrieve the type
            object_type = map_carla_to_kitti(env_object.type)
            object_types.append({object_type})
        bbox_3d = np.array(bbox_3d)
        return LabelData(object_types=object_types, transform=bbox_3d)

    def get_actor_bbs(self) -> np.ndarray:
        bbox_3d = []
        object_types = []
        for actor in self.actors:
            # For the actors, we must provide the actors' transform
            vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
            vertices = [(v.x, v.y, v.z) for v in vertices]
            bbox_3d.append(vertices)

            # Retrieve the type
            tags = {map_carla_to_kitti(tag) for tag in actor.semantic_tags}
            object_types.append(tags)
        bbox_3d = np.array(bbox_3d)
        return LabelData(object_types=object_types, transform=bbox_3d)
    
    @staticmethod
    def get_bbs_edges(bbs: np.ndarray):
        # Retrieve all edges from bounding boxes
        edges = [[0,1], [1,3], [3,2], [2,0], 
                 [0,4], [4,5], [5,1], [5,7], 
                 [7,6], [6,4], [6,2], [7,3]]
        return bbs[:, edges]
