import carla
import queue
import logging

class WorldSensor:
    """
    Class for world sensor.

    Attributes:
        world (carla.World): CARLA world object.
        q (queue.Queue): Queue for world snapshots.
        sensor_data (carla.WorldSnapshot): World snapshot data.
    """
    def __init__(self, world: carla.World, max_queue_size: int = 100) -> None:
        self.world = world
        self.q = queue.Queue(max_queue_size)

        self.sensor_data = None
        self.world.on_tick(self.produce)

    def produce(self, data: carla.WorldSnapshot):
        """
        Produce world snapshot data and add to the queue.

        Args:
            data (carla.WorldSnapshot): World snapshot data.
        """
        try:
            self.q.put_nowait(data)
        except queue.Full:
            logging.warning(f"Queue overflow on frame {data.frame}.")

    def consume(self, frame: int = None, timeout: float = 1.0):
        """
        Consume world snapshot data from the queue.

        Args:
            frame (int, optional): Frame number. Defaults to None.
            timeout (float, optional): Timeout value. Defaults to 1.0.
        """
        while True:
            data = self.q.get(block=True, timeout=timeout)
            if frame is not None and data.frame < frame:
                continue
            self.sensor_data = data
            self.decoded_data = self.decode(data)
            break

    def decode(self, data: carla.WorldSnapshot):
        """
        Decode the world snapshot data.

        Args:
            data (carla.WorldSnapshot): World snapshot data.

        Returns:
            None
        """
        return None
