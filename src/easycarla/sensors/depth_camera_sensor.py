import carla
import numpy as np

from easycarla.sensors.camera_sensor import CameraSensor, MountingDirection, MountingPosition

class DepthCameraSensor(CameraSensor):

    cache_data: np.ndarray = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData):
        data.convert(carla.ColorConverter.Depth)
        decoded_data = super().decode(data)
        self.cache_data = decoded_data
        # Normalize range and scale to max range
        # to acquire depth values in meter
        decoded_data = decoded_data / 255 * 1000
        return decoded_data
    
    def to_img(self) -> np.ndarray:
        return self.cache_data.swapaxes(0, 1)

