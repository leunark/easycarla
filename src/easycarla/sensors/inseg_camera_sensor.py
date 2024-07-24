import carla
import numpy as np

from easycarla.sensors.camera_sensor import CameraSensor, MountingDirection, MountingPosition

class InsegCameraSensor(CameraSensor):

    cache_data: np.ndarray = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData):
        decoded_data = super().decode(data)
        self.cache_data = decoded_data
        # Instance IDs are encoded in the G and B channels of the RGB image file
        # The R channel contains the standard semantic ID
        semantic_ids = decoded_data[:, :, 0]
        instance_ids = decoded_data[:, :, 1].astype(int) * 255 + decoded_data[:, :, 2].astype(int)
        return np.concatenate((semantic_ids[:, :, None], instance_ids[:, :, None]), axis=2)
    
    def preview(self) -> np.ndarray:
        return self.cache_data.swapaxes(0, 1)
