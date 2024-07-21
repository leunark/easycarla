from pathlib import Path
import logging
import carla
import numpy as np

from easycarla.sensors import Sensor, MountingDirection, MountingPosition
from easycarla.tf import Projection, Transformation


class CameraSensor(Sensor):

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

    def decode(self, data: carla.SensorData) -> np.ndarray:
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.decoded_data = array.copy()
        return self.decoded_data

    def to_img(self) -> np.ndarray:
        return self.decoded_data.swapaxes(0, 1)

    def project(self, points: np.ndarray) -> np.ndarray:
        return Projection.project_to_camera(
            points=points,
            K=self.calibration)

    def get_calibration(self) -> np.ndarray:
        fov = self.bp.get_attribute('fov').as_float()
        width = self.bp.get_attribute('image_size_x').as_int()
        height = self.bp.get_attribute('image_size_y').as_int()
        return Projection.build_projection_matrix(width, height, fov=fov)

    