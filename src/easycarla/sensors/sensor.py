from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
import queue
import numpy as np
import carla
import logging

from easycarla.tf import Transformation

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
    """
    Abstract base class for sensors.

    Attributes:
        world (carla.World): CARLA world object.
        transform (carla.Transform): Sensor transform.
        attached_actor (carla.Actor): Actor to which the sensor is attached.
        image_size (tuple[int, int]): Image size.
        sensor_options (dict): Sensor options.
        q (queue.Queue): Queue for sensor data.
        bp (carla.ActorBlueprint): Sensor blueprint.
        actor (carla.Actor): Sensor actor.
        sensor_data (carla.SensorData): Sensor data.
    """
    world: carla.World
    transform: carla.Transform
    attached_actor: carla.Actor
    image_size: tuple[int, int]
    sensor_options: dict

    q: queue.Queue
    bp: carla.ActorBlueprint
    actor: carla.Actor

    sensor_data: carla.SensorData = None

    def __init__(self, 
                 world: carla.World, 
                 attached_actor: carla.Actor, 
                 mounting_position: MountingPosition,
                 mounting_direction: MountingDirection,
                 image_size: tuple[int, int],
                 sensor_options: dict = {},
                 max_queue_size: int = 100,
                 mounting_offset: float = 0.0):
        self.world = world
        self.attached_actor = attached_actor
        self.transform = self.get_mounting_transform(mounting_position, mounting_direction, mounting_offset)
        self.sensor_options = sensor_options
        self.image_size = image_size

        self.q = queue.Queue(max_queue_size)
        self.bp = self.create_blueprint()
        for key in self.sensor_options:
            self.bp.set_attribute(key, self.sensor_options[key])

        self.actor = self.world.spawn_actor(self.bp, self.transform, attach_to=self.attached_actor)
        self.actor.listen(self.produce)
    
    @abstractmethod
    def create_blueprint(self) -> carla.ActorBlueprint:
        """
        Create the blueprint for the sensor.

        Returns:
            carla.ActorBlueprint: Sensor blueprint.
        """
        ...

    @abstractmethod
    def preview(self) -> np.ndarray:
        """
        Get the preview image.

        Returns:
            np.ndarray: Preview image array.
        """
        ...

    @abstractmethod
    def decode(self, data: carla.SensorData) -> None:
        """
        Decode the sensor data.

        Args:
            data (carla.SensorData): Sensor data.
        """
        ...
    
    @abstractmethod
    def save(self, file_path: Path) -> None:
        """
        Save the sensor data to file.

        Args:
            file_path (Path): File path to save the data.
        """
        ...

    def produce(self, data: carla.SensorData):
        """
        Produce sensor data and add to the queue.

        Args:
            data (carla.SensorData): Sensor data.
        """
        # Add the image to the queue for later synchronized processing
        try:
            self.q.put_nowait(data)
        except queue.Full:
            logging.warning(f"Queue overflow on frame {data.frame}.")
    
    def consume(self, frame: int, timeout: float = None):
        """
        Consume sensor data from the queue.

        Args:
            frame (int): Frame number.
            timeout (float, optional): Timeout value. Defaults to None.
        """
        # Consume items from the queue
        while True:
            data = self.q.get(block=True, timeout=timeout)
            if data.frame >= frame:
                self.sensor_data = data
                self.decode(data)
                break

    def render(self):
        """Render the sensor data"""

        if self.surface is not None:
            self.display_man.draw_surface(self.surface, self.display_pos, self.scale_mode)

    def peek(self):
        """
        Peek at the next item in the queue.

        Returns:
            carla.SensorData: Next sensor data.
        """
        if self.q.empty():
            raise queue.Empty
        return self.queue.queue[0]

    def destroy(self):
        """Destroy the sensor actor."""
        if self.actor:
            self.actor.stop()
            self.actor.destroy()

    def world_to_sensor(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from world coordinates to sensor coordinates.

        Args:
            points (np.ndarray): Points in world coordinates.

        Returns:
            np.ndarray: Points in sensor coordinates.
        """
        return Transformation.transform_with_matrix(
            points=points,
            matrix=self.get_world_to_actor())
    
    def get_world_to_actor(self) -> np.ndarray:
        """
        Get the transformation matrix from world to actor coordinates.

        Returns:
            np.ndarray: Transformation matrix.
        """
        return np.array(self.actor.get_transform().get_inverse_matrix())

    def get_actor_to_world(self) -> np.ndarray:
        """
        Get the transformation matrix from actor to world coordinates.

        Returns:
            np.ndarray: Transformation matrix.
        """
        return np.array(self.actor.get_transform().get_matrix())

    def get_mounting_transform(self, 
                              mounting_position: MountingPosition, 
                              mounting_direction: MountingDirection, 
                              offset: float = 0.0) -> carla.Transform:
        """
        Get the mounting transform for the sensor.

        Args:
            mounting_position (MountingPosition): Mounting position.
            mounting_direction (MountingDirection): Mounting direction.
            offset (float, optional): Offset value. Defaults to 0.0.

        Returns:
            carla.Transform: Mounting transform.
        """
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

    def get_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the position and direction vector of the sensor in the world.

        Returns:
            tuple[np.ndarray, np.ndarray]: Position and direction vector.
        """
        "Returns postion and direction vector as numpy arrays in world"
        pos = self.actor.get_transform().location
        pos = np.array([pos.x, pos.y, pos.z])
        forward = self.actor.get_transform().get_forward_vector()
        forward = np.array([forward.x, forward.y, forward.z])
        return pos, forward
    