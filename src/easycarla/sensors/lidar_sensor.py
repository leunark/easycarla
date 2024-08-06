from pathlib import Path
import carla
import numpy as np

from easycarla.sensors.sensor import Sensor, MountingDirection, MountingPosition

class LidarSensor(Sensor):
    """
    Class for LiDAR sensor.

    Attributes:
        range (float): Range of the LiDAR sensor.
        pointcloud (np.ndarray): Point cloud array.
    """

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
        self.range = float(self.bp.get_attribute('range'))
        self.pointcloud = None

    def create_blueprint(self) -> carla.ActorBlueprint:
        """
        Create the blueprint for the LiDAR sensor.

        Returns:
            carla.ActorBlueprint: LiDAR sensor blueprint.
        """
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '100')
        bp.set_attribute('dropoff_general_rate', bp.get_attribute('dropoff_general_rate').recommended_values[0])
        bp.set_attribute('dropoff_intensity_limit', bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        bp.set_attribute('dropoff_zero_intensity', bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        return bp

    def decode(self, data: carla.LidarMeasurement) -> None:
        """
        Decode the sensor data into a point cloud.

        Args:
            data (carla.LidarMeasurement): Sensor data.
        """
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.pointcloud = points

    def preview(self) -> np.ndarray:
        """
        Get the preview image of the point cloud.

        Returns:
            np.ndarray: Preview image array.
        """
        # Take xy plane and scale to display size
        width, height = self.image_size
        points = self.pointcloud
        lidar_range = 2.0*float(self.sensor_options['range'])
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(width, height) / lidar_range
        # Move points from origin to first quadrant (+x, +y)
        lidar_data += (0.5 * width, 0.5 * height)
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (height, width, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        # Transform data to image coordinate system x facing up
        x, y = lidar_data.T
        x = -x + min(width, height) - 1
        lidar_img[(x, y)] = (255, 255, 255)
        return lidar_img
    
    def save(self, file_path: Path) -> None:
        """
        Save the point cloud to file.

        Args:
            file_path (Path): File path to save the point cloud.
        """
        # Ensure the point cloud is a numpy array of type float32
        point_cloud = self.pointcloud.astype(np.float32)
        
        # Save to binary file
        point_cloud.tofile(file_path)
