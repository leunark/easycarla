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

        self.actors = None
        self.vehicle_bbs = []
        self.pedestrian_bbs = []
        self.bicycle_bbs = []
        self.environment_bbs = []

        self.initialize_bbs()

    def initialize_bbs(self):
        # Define the types you want to retrieve
        bb_types = [
            carla.CityObjectLabel.Pedestrians,
            carla.CityObjectLabel.Car,
            carla.CityObjectLabel.Bus,
            carla.CityObjectLabel.Truck,
            carla.CityObjectLabel.Motorcycle,
            carla.CityObjectLabel.Bicycle,
            carla.CityObjectLabel.Rider
        ]

        # Retrieve and store bounding boxes for environmental objects
        for bb_type in bb_types:
            env_objects = self.world.get_environment_objects(bb_type)
            self.environment_bbs.extend(env_obj.bounding_box for env_obj in env_objects)

        # First, retrieve all actors
        actor_list = self.world.get_actors()
        self.actors = ([a for a in actor_list.filter("walker.*")] + 
                       [a for a in actor_list.filter("vehicle.*")])

    def update_dynamic_bbs(self):
        # Clear current bounding boxes
        self.vehicle_bbs.clear()
        self.pedestrian_bbs.clear()
        self.bicycle_bbs.clear()

        # Update bounding boxes for dynamic objects
        actors = self.world.get_actors()
        for actor in actors:
            if 'vehicle' in actor.type_id:
                self.vehicle_bbs.append(actor.bounding_box)
            elif 'walker' in actor.type_id:
                self.pedestrian_bbs.append(actor.bounding_box)
            elif 'bicycle' in actor.type_id:
                self.bicycle_bbs.append(actor.bounding_box)

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

    def get_bounding_boxes(self) -> np.ndarray:
        # Update dynamic bounding boxes
        actor_bbs = []
        for actor in self.actors:
            vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
            vertices = [(v.x, v.y, v.z) for v in vertices]
            actor_bbs.append(vertices)

        env_bbs = []
        for bb in self.environment_bbs:
            vertices = bb.get_world_vertices(carla.Transform())
            vertices = [(v.x, v.y, v.z) for v in vertices]
            env_bbs.append(vertices)

        # Combine all bounding boxes into one list
        all_bbs = env_bbs + actor_bbs

        return np.array(all_bbs)

        # Convert bounding boxes to numpy array of vertices
        vertices_list = []
        for bb in all_bbs:
            vertices = bb.get_world_vertices(carla.Transform())
            vertices_list.append([[v.x, v.y, v.z] for v in vertices])

        return np.array(vertices_list)

