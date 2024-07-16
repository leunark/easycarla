from pathlib import Path
from enum import Enum
import queue
import time
import numpy as np
import carla
import pygame
import cv2
import logging
import threading

from easycarla.sim.display_manager import DisplayManager, ScaleMode


class SensorType(Enum):
    LIDAR = 1
    LIDAR_SEMANTIC_SEGMENTATION = 2
    CAMERA_RGB = 3
    CAMERA_DEPTH_CAMERA = 4
    CAMERA_INSTANCE_SEGMENTATION = 5

class MountingPosition(Enum):
    TOP = 1
    FRONT = 2
    LEFT = 3
    REAR = 4
    RIGHT = 5

class Sensor:
    def __init__(self, 
                 world: carla.World, 
                 display_man: DisplayManager,
                 display_pos: tuple[int, int],
                 sensor_type: SensorType, 
                 transform: carla.Transform, 
                 attached_actor: carla.Actor, 
                 image_size: tuple[int, int] = [800, 600],
                 scale_mode: ScaleMode = ScaleMode.NORMAL,
                 sensor_options: dict = {}):
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor_type = sensor_type
        self.transform = transform
        self.attached_actor = attached_actor
        self.scale_mode = scale_mode
        self.sensor_options = sensor_options
        self.image_size = image_size

        self.sensor_data = None
        self.decoded_data = None
        self.surface = None
        self.max_queue_size = 100
        self.queue = queue.Queue(self.max_queue_size)
        self.sensor = self.init_sensor()
        self.timer = time.perf_counter

        self.time_processing = 0.0
        self.tics_processing = 0

    def init_sensor(self):
        if self.sensor_type == SensorType.CAMERA_RGB:
            bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(self.image_size[0]))
            bp.set_attribute('image_size_y', str(self.image_size[1]))

        elif self.sensor_type == SensorType.CAMERA_DEPTH_CAMERA:
            bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            bp.set_attribute('image_size_x', str(self.image_size[0]))
            bp.set_attribute('image_size_y', str(self.image_size[1]))
        
        elif self.sensor_type == SensorType.CAMERA_INSTANCE_SEGMENTATION:
            bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            bp.set_attribute('image_size_x', str(self.image_size[0]))
            bp.set_attribute('image_size_y', str(self.image_size[1]))

        elif self.sensor_type == SensorType.LIDAR:
            bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            bp.set_attribute('range', '100')
            bp.set_attribute('dropoff_general_rate', bp.get_attribute('dropoff_general_rate').recommended_values[0])
            bp.set_attribute('dropoff_intensity_limit', bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            bp.set_attribute('dropoff_zero_intensity', bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        
        elif self.sensor_type == SensorType.LIDAR_SEMANTIC_SEGMENTATION:
            bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            bp.set_attribute('range', '100')
        
        else:
            raise Exception(f'Invalid Sensor Type {self.sensor_type}')

        for key in self.sensor_options:
            bp.set_attribute(key, self.sensor_options[key])

        # Calculate once calibration matrix
        if self.sensor_type == SensorType.CAMERA_RGB or self.sensor_type == SensorType.CAMERA_INSTANCE_SEGMENTATION or self.sensor_type == SensorType.CAMERA_DEPTH_CAMERA:
            fov = bp.get_attribute('fov').as_float()
            width = bp.get_attribute('image_size_x').as_int()
            height = bp.get_attribute('image_size_y').as_int()
            self.calibration = self.build_projection_matrix(width, height, fov=fov)
        else:
            self.calibration = None

        sensor = self.world.spawn_actor(bp, self.transform, attach_to=self.attached_actor)
        sensor.listen(self.produce)

        return sensor

    def produce(self, data: carla.SensorData):
        # Add the image to the queue for later synchronized processing
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            logging.warning(f"Queue overflow on frame {data.frame}.")
    
    def consume(self, frame: int, timeout: float = 1.0):
        # Consume items from the queue
        while True:
            data = self.queue.get(block=True, timeout=timeout)
            if data.frame >= frame:
                self.decoded_data = self.decode_data(data)
                self.create_surface(self.decoded_data)
                return data

    def render(self):
        if self.surface is not None:
            self.display_man.draw_surface(self.surface, self.display_pos, self.scale_mode)

    def peek(self):
        if self.queue.empty():
            raise queue.Empty
        return self.queue.queue[0]

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
    
    def decode_data(self, data: carla.SensorData):
        if self.sensor_type == SensorType.CAMERA_RGB:
            decoded_data = self.decode_image_rgb(data)
        elif self.sensor_type == SensorType.CAMERA_DEPTH_CAMERA:
            data.convert(carla.ColorConverter.Depth)
            decoded_data = self.decode_image_rgb(data)
            #image_depths = image[:, :, 0].astype(np.float32) / 255.0 * 1000.0
        elif self.sensor_type == SensorType.CAMERA_INSTANCE_SEGMENTATION:
            decoded_data = self.decode_image_rgb(data)
        elif self.sensor_type == SensorType.LIDAR:
            decoded_data = self.decode_lidar(data)
        else:
            raise Exception(f'Invalid Sensor Type {self.sensor_type}')
        return decoded_data

    def create_surface(self, data: np.ndarray):
        if self.sensor_type == SensorType.LIDAR:
            image = self.pc_to_img(data)
        else:
            image = data.swapaxes(0, 1)
        self.surface = pygame.surfarray.make_surface(image)

    @staticmethod
    def decode_image_rgb(image: carla.Image, color_converter: carla.ColorConverter = carla.ColorConverter.Raw) -> np.ndarray:
        image.convert(color_converter)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    @staticmethod
    def decode_lidar(lidar: carla.LidarMeasurement | carla.SemanticLidarMeasurement) -> np.ndarray:
        points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points

    def pc_to_img(self, points: np.ndarray) -> np.ndarray:
        # Take xy plane and scale to display size
        disp_size = self.display_man.get_display_size()
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

    @staticmethod
    def get_mounting_position(actor: carla.Actor, position: MountingPosition, offset=0.5):
        # Get the actor's bounding box
        bounding_box = actor.bounding_box
        extent = bounding_box.extent
        mount_location = bounding_box.location

        if position == MountingPosition.TOP:
            # Mount on top (increase z-axis by half the height + offset)
            mount_location += carla.Location(0, 0, extent.z + offset)
        elif position == MountingPosition.FRONT:
            # Mount in front (increase x-axis by half the length + offset)
            mount_location += carla.Location(extent.x + offset, 0, 0)
        elif position == MountingPosition.LEFT:
            # Mount on the left side (decrease y-axis by half the width + offset)
            mount_location += carla.Location(0, -extent.y - offset, 0)
        elif position == MountingPosition.REAR:
            # Mount at the rear (decrease x-axis by half the length + offset)
            mount_location += carla.Location(-extent.x - offset, 0, 0)
        elif position == MountingPosition.RIGHT:
            # Mount on the right side (increase y-axis by half the width + offset)
            mount_location += carla.Location(0, extent.y + offset, 0)
        else:
            raise ValueError("Invalid mounting position")

        return mount_location
    
    @staticmethod
    def build_projection_matrix(w: int, h: int, fov: float):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K