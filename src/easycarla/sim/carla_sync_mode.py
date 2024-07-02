from pathlib import Path
from queue import Queue
import carla
import numpy as np


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