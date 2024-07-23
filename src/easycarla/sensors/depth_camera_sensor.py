import carla
import numpy as np

from easycarla.sensors.camera_sensor import CameraSensor, MountingDirection, MountingPosition

class DepthCameraSensor(CameraSensor):

    depth_values: np.ndarray = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData) -> None:
        data.convert(carla.ColorConverter.Depth)
        super().decode(data)

        # Read the R, G, and B channels
        R = self.image[:, :, 0].astype(np.float32)
        G = self.image[:, :, 1].astype(np.float32)
        B = self.image[:, :, 2].astype(np.float32)
        
        # Calculate the normalized depth
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        
        # Convert normalized depth to meters
        depth_in_meters = 1000.0 * normalized_depth

        self.depth_values = depth_in_meters
    