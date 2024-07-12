import carla
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from easycarla.sim.carla_sync_mode import CarlaSyncMode
from easycarla.sim.process_data import process_data

class SimulationManager:
    def __init__(self, host='127.0.0.1', port=2000, timeout=40, fixed_delta_seconds=0.05, map_name: str = "Town10HD_Opt", sync: bool = False, reset=False, no_rendering=False):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.map_name = map_name    
        self.client = None
        self.world = None
        self.sync_mode = None
        self.sensors = None
        self.fixed_delta_seconds = fixed_delta_seconds
        self.sync = sync
        self.reset = reset
        self.no_rendering = no_rendering

        self.init()

    def init(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        if self.reset:
            self.reset_world()
        self.world = self.client.get_world()

        self.executor = ThreadPoolExecutor(max_workers=4)  # Number of threads

        # Change settings but backup original first
        self._settings = self.world.get_settings()

        # Set sync settings
        settings = self.world.get_settings()
        self.synchronous_master = False
        if self.sync:
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if self.no_rendering:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        return self

    def destroy(self):
        #self.sync_mode.__exit__(exc_type, exc_val, exc_tb)
        self.world.apply_settings(self._settings)
        self.executor.shutdown(wait=True)

    def tick(self) -> carla.WorldSnapshot:
        if self.sync:
            self.world.tick()
            world_snapshot = self.world.get_snapshot()
        else:
            world_snapshot = self.world.wait_for_tick()
        return world_snapshot
        
    def load_world(self, map_name: str):
        world = self.client.get_world()
        for available_map in self.client.get_available_maps():
            if map_name == available_map.split('/')[-1]:
                world = self.client.load_world_if_different(available_map)
        return world

    def reset_world(self):
        self.client.reload_world()
    