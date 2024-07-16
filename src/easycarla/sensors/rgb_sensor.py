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
        return array

    def to_img(self) -> np.ndarray:
        img = self.decoded_data.swapaxes(0, 1)
        return img
    
    def project(self):
        return self.get_image_point(
            loc=self.sensor.get_transform(), 
            K=self.calibration, 
            w2c=self.get_world_to_actor())

    def get_calibration(self):
        fov = self.bp.get_attribute('fov').as_float()
        width = self.bp.get_attribute('image_size_x').as_int()
        height = self.bp.get_attribute('image_size_y').as_int()
        return self.build_projection_matrix(width, height, fov=fov)

    @staticmethod
    def get_image_point(loc: carla.Location, K: np.ndarray, w2c: np.ndarray):
        """Calculate 2D projection of 3D coordinate"""

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]
    
    @staticmethod
    def build_projection_matrix(w: int, h: int, fov: float):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K