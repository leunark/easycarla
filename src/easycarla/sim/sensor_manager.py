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


class SensorManager:

    def __init__(self, world: carla.World, display_manager: DisplayManager, sensors: list[Sensor], timeout=1.0) -> None:
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
            synced_data = []
            for index, sensor in enumerate(self.sensors):
                data = sensor.consume(world_snapshot.frame, timeout=1.0)
                if data is not None:
                    sensor.create_surface(data)
                synced_data.append(data)
            
            # Get Bounding Boxes and filter them
            # Decode sensors
            # Project bounding boxes onto image
            # Filter occluded objects with instance segmentation
            # Export data 

            # Render data
            self.draw_sensors()
            self.display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            self.display_manager.tick()
        except queue.Empty:
            pass

    def run(self):
        while not self._stop_event.is_set():
            self.consume()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def destroy(self):
        self.stop()
        for sensor in self.sensors:
            sensor.destroy()
