import carla
import numpy as np
import logging

class LabelManager:

    def __init__(self, world: carla.World, hero: carla.Actor = None) -> None:
        self.world = world

        self.hero = hero
        self.hero_pos = None
        self.hero_forward = None

        self.actors = None
        self.vehicle_bbs = []
        self.pedestrian_bbs = []
        self.bicycle_bbs = []
        self.environment_bbs = []

        self.init_bbs()

    def init_bbs(self):
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

        # Remove hero vehicle from actors
        if self.hero is not None:
            self.actors = [a for a in self.actors if a.id != self.hero.id]

    def update(self):
        pass

    def get_bbs(self) -> np.ndarray:
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
    
    @staticmethod
    def filter_bbs_distance(bbs: np.ndarray, target: np.ndarray, distance: float) -> np.ndarray:
        # Filter bbs within squared distance
        delta_pos = (bbs[:, 0] - target)
        sq_dist = (delta_pos**2).sum(axis=1)
        bbs = bbs[sq_dist <= distance**2]
        return bbs
    
    @staticmethod
    def filter_bbs_direction(bbs: np.ndarray, target: np.ndarray, forward: float, dot_threshold = 0.1) -> np.ndarray:
        # Filter bbs that are in front
        delta_pos = (bbs[:, 0] - target)
        delta_pos_unit: np.ndarray = delta_pos / np.linalg.norm(delta_pos, axis=1)[:, None]
        bbs = bbs[delta_pos_unit.dot(forward) > dot_threshold]
        return bbs
    
    @staticmethod
    def get_bbs_edges(bbs: np.ndarray):
        # Retrieve all edges from bounding boxes
        edges = [[0,1], [1,3], [3,2], [2,0], 
                 [0,4], [4,5], [5,1], [5,7], 
                 [7,6], [6,4], [6,2], [7,3]]
        return bbs[:, edges]

    @staticmethod
    def get_transform(actor: carla.Actor) -> tuple[np.ndarray, np.ndarray]:
        "Returns postion and direction vector as numpy arrays in world"
        pos = actor.get_transform().location
        pos = np.array([pos.x, pos.y, pos.z])
        forward = actor.get_transform().get_forward_vector()
        forward = np.array([forward.x, forward.y, forward.z])
        return pos, forward