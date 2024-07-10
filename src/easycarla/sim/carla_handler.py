import carla
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from easycarla.sim.carla_sync_mode import CarlaSyncMode
from easycarla.sim.traffic_simulation import TrafficSimulation
from easycarla.sim.process_data import process_data

class CarlaHandler:
    def __init__(self, host='127.0.0.1', port=2000, timeout=2.0, num_vehicles=30, num_walkers=40, fps=30):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.sync_mode = None
        self.sensors = None
        self.num_vehicles = num_vehicles
        self.num_walkers = num_walkers
        self.fps = fps

        self.traffic_sim = TrafficSimulation(self.client, fps=self.fps)
        self.executor = ThreadPoolExecutor(max_workers=4)  # Number of threads

    def __enter__(self):
        # Spawn actors
        self.traffic_sim.spawn_vehicles(self.num_vehicles)
        self.traffic_sim.spawn_walkers(self.num_walkers)

        # Make one vehicle hero and retrieve sensors
        self.sensors = self.traffic_sim.make_hero()

        # Start carla sim with synced sensors
        self.sync_mode = CarlaSyncMode(self.world, *self.sensors, fps=self.fps)
        self.sync_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sync_mode.__exit__(exc_type, exc_val, exc_tb)
        self.traffic_sim.destroy()
        self.executor.shutdown(wait=True)

    def tick(self, timeout=2.0):
        # snapshots contains carla.WorldSnapShot and the registered sensors data carla.SensorData
        snapshots = self.sync_mode.tick(timeout=timeout)
        # Additionally, we add the world to be able to retrieve e.g. actor ids and run it parallel
        self.executor.submit(process_data, self.world, *snapshots)
        return self.world, *snapshots
