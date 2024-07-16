import carla
import numpy as np

from easycarla.sensors.rgb_sensor import RgbSensor, MountingDirection, MountingPosition

class DepthSensor(RgbSensor):

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.image_size[0]))
        bp.set_attribute('image_size_y', str(self.image_size[1]))
        return bp

    def decode(self, data: carla.SensorData):
        data.convert(carla.ColorConverter.Depth)
        return super().decode(data)

