from pathlib import Path
import logging
import carla
import numpy as np

from easycarla.sensors.sensor import Sensor, MountingDirection, MountingPosition

class RgbSensor(Sensor):

    def __init__(self, 
                 world: carla.World, 
                 attached_actor: carla.Actor, 
                 mounting_position: MountingPosition, 
                 mounting_direction: MountingDirection, 
                 image_size: tuple[int, int], 
                 sensor_options: dict = {}, 
                 max_queue_size: int = 100):
        super().__init__(world, attached_actor, mounting_position, mounting_direction, image_size, sensor_options, max_queue_size)
        self.calibration = self.get_calibration()
        self.image_data = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array.copy()

    def to_img(self) -> np.ndarray:
        img = self.decoded_data.swapaxes(0, 1)
        return img
    
    def project(self, points: np.ndarray) -> np.ndarray:
        return self.get_image_points(
            points=points,
            K=self.calibration, 
            w2c=self.get_world_to_actor())

    def get_calibration(self) -> np.ndarray:
        fov = self.bp.get_attribute('fov').as_float()
        width = self.bp.get_attribute('image_size_x').as_int()
        height = self.bp.get_attribute('image_size_y').as_int()
        return self.build_projection_matrix(width, height, fov=fov)

    @staticmethod
    def get_image_points(points: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
        """Calculate 2D projections of multiple 3D coordinates"""
        # Ensure points is a 2D array
        points = np.atleast_2d(points)
        
        # Add homogeneous coordinate
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Transform to camera coordinates
        points_camera = np.dot(w2c, points_homogeneous.T).T
        
        # Change from UE4's coordinate system to "standard"
        # (x, y, z) -> (y, -z, x)
        points_camera = points_camera[:, [1, 2, 0]]
        points_camera[:, 1] *= -1
        
        # Project 3D->2D using the camera matrix
        points_img = np.dot(K, points_camera.T).T
        
        # Normalize
        points_img[:, :2] /= points_img[:, 2:3]
        
        return points_img[:, :2]
    
    @staticmethod
    def build_projection_matrix(w: int, h: int, fov: float):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K