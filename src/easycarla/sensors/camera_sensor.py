from pathlib import Path
import logging
import carla
import numpy as np
import cv2

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
                 max_queue_size: int = 100,
                 mounting_offset: float = 0.5):
        super().__init__(world, attached_actor, mounting_position, mounting_direction, image_size, sensor_options, max_queue_size, mounting_offset)
        self.calibration = self.get_calibration()
        self.rgb_image = None
        self.preview_image = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData) -> None:
        # Convert the image raw data to a numpy array
        image_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        
        # Reshape array into (height, width, 4) where the last dimension is RGBA
        image_data = np.reshape(image_data, (data.height, data.width, 4))
        
        # Extract the R, G, and B channels (ignore A)
        image_data = image_data[:, :, :3]
        image_data = image_data[:, :, ::-1]
        self.rgb_image = image_data

        # Make a copy to allow drawing on image while keeping original
        self.preview_image = image_data.copy()

    def preview(self) -> np.ndarray:
        return self.preview_image.swapaxes(0, 1)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return Projection.project_to_camera(
            points=points,
            K=self.calibration)

    def get_calibration(self) -> np.ndarray:
        fov = self.bp.get_attribute('fov').as_float()
        width = self.bp.get_attribute('image_size_x').as_int()
        height = self.bp.get_attribute('image_size_y').as_int()
        return Projection.build_projection_matrix(width, height, fov=fov)

    def save(self, file_path: Path) -> None:
        cv2.imwrite(str(file_path), self.rgb_image)
    