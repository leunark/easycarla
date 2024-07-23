from pathlib import Path
import carla
import numpy as np

from easycarla.sensors.sensor import Sensor, MountingDirection, MountingPosition

class LidarSensor(Sensor):

    def __init__(self, 
                 world: carla.World, 
                 attached_actor: carla.Actor, 
                 mounting_position: MountingPosition, 
                 mounting_direction: MountingDirection, 
                 image_size: tuple[int, int], 
                 sensor_options: dict = {}, 
                 max_queue_size: int = 100):
        super().__init__(world, attached_actor, mounting_position, mounting_direction, image_size, sensor_options, max_queue_size)
        self.range = float(self.bp.get_attribute('range'))
        self.pointcloud = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '100')
        bp.set_attribute('dropoff_general_rate', bp.get_attribute('dropoff_general_rate').recommended_values[0])
        bp.set_attribute('dropoff_intensity_limit', bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        bp.set_attribute('dropoff_zero_intensity', bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        return bp

    def decode(self, data: carla.LidarMeasurement) -> None:
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.pointcloud = points

    def to_img(self) -> np.ndarray:
        # Take xy plane and scale to display size
        disp_size = self.image_size
        points = self.pointcloud
        lidar_range = 2.0*float(self.sensor_options['range'])
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        # Move points from origin to first quadrant (+x, +y)
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        return lidar_img
    
    def save(self, file_path: Path) -> None:
        # Ensure the point cloud is a numpy array of type float32
        point_cloud = self.pointcloud.astype(np.float32)
        
        # Save to binary file
        point_cloud.tofile(file_path)
