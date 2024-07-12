import time
import carla
from pathlib import Path
import numpy as np
from enum import Enum
import pygame

from easycarla.sim.display_manager import DisplayManager


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

class ScaleMode(Enum):
    NORMAL = 1
    ZOOM_CENTER = 2
    STRETCH_FIT = 3

class SensorManager:
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

        self.surface = None
        self.sensor = self.init_sensor()
        self.timer = time.perf_counter

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self):
        if self.sensor_type == SensorType.CAMERA_RGB:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.image_size[0]))
            camera_bp.set_attribute('image_size_y', str(self.image_size[1]))

            for key in self.sensor_options:
                camera_bp.set_attribute(key, self.sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, self.transform, attach_to=self.attached_actor)
            camera.listen(lambda image: self.save_rgb_image(image, carla.ColorConverter.Raw))

            return camera

        elif self.sensor_type == SensorType.CAMERA_DEPTH_CAMERA:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            camera_bp.set_attribute('image_size_x', str(self.image_size[0]))
            camera_bp.set_attribute('image_size_y', str(self.image_size[1]))

            for key in self.sensor_options:
                camera_bp.set_attribute(key, self.sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, self.transform, attach_to=self.attached_actor)
            camera.listen(lambda image: self.save_rgb_image(image, carla.ColorConverter.LogarithmicDepth))

            return camera
        
        elif self.sensor_type == SensorType.CAMERA_INSTANCE_SEGMENTATION:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.image_size[0]))
            camera_bp.set_attribute('image_size_y', str(self.image_size[1]))

            for key in self.sensor_options:
                camera_bp.set_attribute(key, self.sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, self.transform, attach_to=self.attached_actor)
            camera.listen(self.save_rgb_image)

            return camera

        elif self.sensor_type == SensorType.LIDAR:
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in self.sensor_options:
                lidar_bp.set_attribute(key, self.sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, self.transform, attach_to=self.attached_actor)
            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif self.sensor_type == SensorType.LIDAR_SEMANTIC_SEGMENTATION:
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in self.sensor_options:
                lidar_bp.set_attribute(key, self.sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, self.transform, attach_to=self.attached_actor)
            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def read_rgb_image(self, image: carla.Image, color_converter: carla.ColorConverter = carla.ColorConverter.Raw):
        image.convert(color_converter)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array
    
    @staticmethod
    def decode_depth_image(image: carla.Image):
        """
        Decodes a depth image from Carla to create a depth matrix.
        
        Args:
            image (carla.Image): The depth image obtained from Carla.
            
        Returns:
            np.ndarray: The decoded depth matrix.
        """
        # Convert the image raw data to a numpy array
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        
        # Reshape array into (height, width, 4) where the last dimension is RGBA
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        
        # Extract the R, G, and B channels (ignore A)
        R = image_data[:, :, 2].astype(np.float32)
        G = image_data[:, :, 1].astype(np.float32)
        B = image_data[:, :, 0].astype(np.float32)
        
        # Calculate the normalized depth
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        
        # Convert normalized depth to meters
        depth_in_meters = 1000.0 * normalized_depth
        
        return depth_in_meters
    
    def save_rgb_image(self, image, color_converter: carla.ColorConverter = carla.ColorConverter.Raw):
        t_start = self.timer()

        image.convert(color_converter)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        t_start = self.timer()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            display_offset = self.display_man.get_display_offset(self.display_pos)
            display_size = self.display_man.get_display_size()
            dest_rect = pygame.Rect(*display_offset, *display_size)
            self.draw_surface(self.surface, self.display_man.display, dest_rect, self.scale_mode)

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

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
    def draw_surface(src_surface: pygame.Surface, dest_surface: pygame.Surface, dest_rect: pygame.Rect, scale_mode: ScaleMode):
        if scale_mode == ScaleMode.NORMAL:
            # Normal scaling, blit the source surface onto the destination surface
            src_rect = src_surface.get_rect()
            crop_rect = pygame.Rect(
                (src_rect.width - dest_rect.width) // 2, 
                (src_rect.height - dest_rect.height) // 2,
                dest_rect.width,
                dest_rect.height
            )
            src_surface = src_surface.subsurface(crop_rect)
            dest_surface.blit(src_surface, dest_rect.topleft)

        elif scale_mode == ScaleMode.ZOOM_CENTER:
            # Zoom and center the source surface to fit inside the destination rectangle
            src_rect = src_surface.get_rect()
            scale = max(dest_rect.width / src_rect.width, dest_rect.height / src_rect.height)
            src_surface = pygame.transform.scale(src_surface, (src_rect.width * scale, src_rect.height * scale))
            src_rect = src_surface.get_rect()
            crop_rect = pygame.Rect(
                (src_rect.width - dest_rect.width) // 2, 
                (src_rect.height - dest_rect.height) // 2,
                dest_rect.width,
                dest_rect.height
            )
            src_surface = src_surface.subsurface(crop_rect)
            dest_surface.blit(src_surface, dest_rect.topleft)

        elif scale_mode == ScaleMode.STRETCH_FIT:
            # Stretch the source surface to fit the destination rectangle
            src_surface = pygame.transform.scale(src_surface, (dest_rect.width, dest_rect.height))
            dest_surface.blit(src_surface, dest_rect.topleft)