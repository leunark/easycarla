from pathlib import Path
from queue import Queue
import carla
import numpy as np


def get_vehicle_bounding_boxes(world: carla.World, world_snapshot: carla.WorldSnapshot):
    # Get all actors from the world
    actors = world.get_actors()

    # Filter out the vehicles from all actors
    vehicles = actors.filter('vehicle.*')

    # Retrieve bounding boxes for all vehicles
    vehicle_bounding_boxes = {}
    for vehicle in vehicles:
        vehicle_snapshot = world_snapshot.find(vehicle.id)
        if vehicle_snapshot:
            # Get the bounding box of the vehicle
            bounding_box = vehicle.bounding_box
            # Transform the bounding box to the vehicle's current transform
            bounding_box.location = vehicle_snapshot.get_transform().transform(bounding_box.location)
            vehicle_bounding_boxes[vehicle.id] = bounding_box

    return vehicle_bounding_boxes

class CarlaSyncMode:
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context.

    Example usage:
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world: carla.World, *sensors: carla.Sensor, fps: int = 20):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues: list[Queue] = []
        self._settings = None

    def __enter__(self) -> 'CarlaSyncMode':
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        # listen event will pass sensor data to q.put which adds it to its queue
        def make_queue(register_event):
            q = Queue()
            register_event(q.put)
            self._queues.append(q)

        # Let on tick add carla.WorldSnapshot to world queue
        make_queue(self.world.on_tick)
        # Let each sensor add carla.SensorDate to its own sensor queue
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout: float) -> list[carla.WorldSnapshot|carla.SensorData]:
        """Advance the simulation and wait for the data from all sensors.
        Returns a list of carla.WorldSnapshot and the sensor data in the same order 
        they were registered all at the same frame.
        """
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args: any) -> None:
        self.world.apply_settings(self._settings)
    
    def _retrieve_data(self, sensor_queue: Queue, timeout: float) -> any:
        """Retrieve data from the sensor queue for the current frame."""
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data