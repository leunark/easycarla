from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
import queue
import time
import numpy as np
import carla
import pygame
import cv2
import logging


class MountingPosition(Enum):
    TOP = 1
    FRONT = 2
    LEFT = 3
    REAR = 4
    RIGHT = 5

class MountingDirection(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4

class Sensor(ABC):

    world: carla.World
    transform: carla.Transform
    attached_actor: carla.Actor
    image_size: tuple[int, int]
    sensor_options: dict

    q: queue.Queue
    bp: carla.ActorBlueprint
    sensor: carla.Actor

    sensor_data: carla.SensorData = None
    decoded_data: np.ndarray = None
    image: np.ndarray = None


    def __init__(self, 
                 world: carla.World, 
                 attached_actor: carla.Actor, 
                 mounting_position: MountingPosition,
                 mounting_direction: MountingDirection,
                 image_size: tuple[int, int],
                 sensor_options: dict = {},
                 max_queue_size: int = 100):
        self.world = world
        self.attached_actor = attached_actor
        self.transform = self.get_mounting_transform(mounting_position, mounting_direction)
        self.sensor_options = sensor_options
        self.image_size = image_size

        self.q = queue.Queue(max_queue_size)
        self.bp = self.create_blueprint()
        for key in self.sensor_options:
            self.bp.set_attribute(key, self.sensor_options[key])

        self.sensor = self.world.spawn_actor(self.bp, self.transform, attach_to=self.attached_actor)
        self.sensor.listen(self.produce)

    @abstractmethod
    def create_blueprint(self) -> carla.ActorBlueprint:
        ...

    @abstractmethod
    def to_img(self) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, data: carla.SensorData):
        ...
    
    def save(self, output_folder: Path):
        path = output_folder / f"{self.sensor_data.frame}.png"
        logging.info(f"Not saved {path}.")

    def produce(self, data: carla.SensorData):
        # Add the image to the queue for later synchronized processing
        try:
            self.q.put_nowait(data)
        except queue.Full:
            logging.warning(f"Queue overflow on frame {data.frame}.")
    
    def consume(self, frame: int, timeout: float = None):
        # Consume items from the queue
        while True:
            data = self.q.get(block=True, timeout=timeout)
            if data.frame >= frame:
                self.sensor_data = data
                self.decoded_data = self.decode(data)
                break

    def render(self):
        if self.surface is not None:
            self.display_man.draw_surface(self.surface, self.display_pos, self.scale_mode)

    def peek(self):
        if self.q.empty():
            raise queue.Empty
        return self.queue.queue[0]

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()

    def get_world_to_actor(self):
        return np.array(self.sensor.get_transform().get_inverse_matrix())

    def get_mounting_transform(self, 
                              mounting_position: MountingPosition, 
                              mounting_direction: MountingDirection, 
                              offset: float = 0.5) -> carla.Transform:
        # Mounting position based on bounding box
        bounding_box = self.attached_actor.bounding_box
        extent = bounding_box.extent
        mount_location = bounding_box.location

        if mounting_position == MountingPosition.TOP:
            # Mount on top (increase z-axis by half the height + offset)
            mount_location += carla.Location(0, 0, extent.z + offset)
        elif mounting_position == MountingPosition.FRONT:
            # Mount in front (increase x-axis by half the length + offset)
            mount_location += carla.Location(extent.x + offset, 0, 0)
        elif mounting_position == MountingPosition.LEFT:
            # Mount on the left side (decrease y-axis by half the width + offset)
            mount_location += carla.Location(0, -extent.y - offset, 0)
        elif mounting_position == MountingPosition.REAR:
            # Mount at the rear (decrease x-axis by half the length + offset)
            mount_location += carla.Location(-extent.x - offset, 0, 0)
        elif mounting_position == MountingPosition.RIGHT:
            # Mount on the right side (increase y-axis by half the width + offset)
            mount_location += carla.Location(0, extent.y + offset, 0)
        else:
            raise ValueError("Invalid mounting position")
        
        # Get mounting rotation
        if mounting_direction == MountingDirection.FORWARD:
            mount_rotation = carla.Rotation(0, 0, 0)
        elif mounting_direction == MountingDirection.LEFT:
            mount_rotation = carla.Rotation(0, 0, 90)
        elif mounting_direction == MountingDirection.BACKWARD:
            mount_rotation = carla.Rotation(0, 0, 180)
        elif mounting_direction == MountingDirection.RIGHT:
            mount_rotation = carla.Rotation(0, 0, -90)
        else:
            raise ValueError("Invalid mounting direction")
        
        mount_transform = carla.Transform(mount_location, mount_rotation)
        return mount_transform