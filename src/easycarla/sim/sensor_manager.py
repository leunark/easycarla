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
from easycarla.sim.sensor import Sensor, SensorType
from easycarla.sim.bbs import BBS

class SensorManager:

    def __init__(self, 
                 world: carla.World, 
                 display_manager: DisplayManager, 
                 sensors: list[Sensor], 
                 timeout=1.0) -> None:
        self.sensors = sensors
        self.world = world
        self.display_manager = display_manager
        self.timeout = timeout
        self.queue = queue.Queue()
        self.world.on_tick(self.queue.put)
        self._stop_event = threading.Event()
        self._thread = None

    def draw_sensors(self):
        for sensor in self.sensors:
            sensor.render()

    def consume(self):
        try:
            # Consume world snapshot
            world_snapshot = self.queue.get(block=True, timeout=self.timeout)
            
            # Consume sensor data
            sensor_data_list: list[Sensor] = []
            decoded_data_list = []
            for sensor in self.sensors:
                data = sensor.consume(world_snapshot.frame, timeout=1.0)
                sensor_data_list.append(data)
                #decoded_data = sensor.decode_data(data)
                #decoded_data_list.append(decoded_data)

            # Get bounding boxes in world coordinate
            bbs = BBS.get_vehicles_and_pedestrians_bbs(self.world)

            

            # Create surface for visualization
            #for sensor, decoded_data in zip(self.sensors, decoded_data_list):
            #    sensor.create_surface(decoded_data)

            # Project bounding boxes onto image
            # https://carla.readthedocs.io/en/0.9.14/tuto_G_bounding_boxes/
            
            # Render data
            self.draw_sensors()
            self.display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            self.display_manager.tick()
        except queue.Empty:
            pass

    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()
