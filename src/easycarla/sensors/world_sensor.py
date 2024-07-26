import carla
import queue
import logging
import numpy as np

class WorldSensor:

    def __init__(self, world: carla.World, max_queue_size: int = 100) -> None:
        self.world = world
        self.q = queue.Queue(max_queue_size)

        self.sensor_data = None
        self.world.on_tick(self.produce)

    def produce(self, data: carla.WorldSnapshot):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            logging.warning(f"Queue overflow on frame {data.frame}.")

    def consume(self, frame: int = None, timeout: float = 1.0):
        while True:
            data = self.q.get(block=True, timeout=timeout)
            if frame is not None and data.frame < frame:
                continue
            self.sensor_data = data
            self.decoded_data = self.decode(data)
            break

    def decode(self, data: carla.WorldSnapshot):
        return None
