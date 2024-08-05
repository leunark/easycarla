import carla
import numpy as np

from easycarla.sensors.camera_sensor import CameraSensor

class DepthCameraSensor(CameraSensor):

    depth_values: np.ndarray = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData) -> None:
        super().decode(data)

        # Read the R, G, and B channels
        R = self.rgb_image[:, :, 0].astype(np.float32)
        G = self.rgb_image[:, :, 1].astype(np.float32)
        B = self.rgb_image[:, :, 2].astype(np.float32)
        
        # Calculate the normalized depth
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        
        # Convert normalized depth to meters
        depth_in_meters = 1000.0 * normalized_depth

        self.depth_values = depth_in_meters
    
    def preview(self) -> np.ndarray:
        # Define the custom colormap from value range [0,1]
        # a < 0
        def custom_colormap(value: np.ndarray, skew: float = 50.0):
            # Apply 1/x transformation
            a = -1/skew
            b = -0.5 + (0.25 - a)**0.5
            c = -a/b
            y = a/(value+b)+c

            # Vectorized approach to create the colormap
            r = y
            g = np.zeros_like(y)
            b = 1 - y
            rgb = (np.stack((r, g, b), axis=-1) * 255).astype(np.uint8)
            return rgb
        
        # Apply the custom colormap to the normalized depth values
        img = custom_colormap(self.depth_values / 1000)
        return img