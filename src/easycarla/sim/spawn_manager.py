import logging
from numpy import random
import carla
from dataclasses import dataclass, field


@dataclass
class SpawnManagerConfig:
    """
    Configuration class for the spawn manager.

    Attributes:
        seed (int): Random seed.
        tm_port (int): Traffic manager port.
        respawn_dormant_vehicles (bool): Whether to respawn dormant vehicles.
        hybrid_physics_mode (bool): Whether to enable hybrid physics mode.
        hybrid_physics_mode_radius (float): Radius for hybrid physics mode.
        percentage_pedestrians_crossing (float): Percentage of pedestrians crossing.
        percentage_pedestrians_running (float): Percentage of pedestrians running.
        percentage_speed_difference (float): Percentage speed difference.
        distance_to_leading_vehicle (float): Distance to leading vehicle.
        percentage_vehicles_lights_on (float): Percentage of vehicles with lights on.
    """
    seed: int = None
    tm_port: int = 8000
    respawn_dormant_vehicles: bool = False
    hybrid_physics_mode: bool = True
    hybrid_physics_mode_radius: float = 50.0
    percentage_pedestrians_crossing: float = 0.2
    percentage_pedestrians_running: float = 0.05
    percentage_speed_difference: float = 30.0
    distance_to_leading_vehicle: float = 3.0
    percentage_vehicles_lights_on: float = 0.5


class SpawnManager:
    """
    Class to manage the spawning of vehicles and pedestrians in the CARLA simulation.

    Attributes:
        client (carla.Client): CARLA client object.
        config (SpawnManagerConfig): Configuration for the spawn manager.
        hero (carla.Actor): Hero vehicle actor.
        vehicles (list[carla.Actor]): List of spawned vehicle actors.
        pedestrians (list[carla.Actor]): List of spawned pedestrian actors.
        pedestrian_controllers (list[carla.Actor]): List of pedestrian controllers.
        world (carla.World): CARLA world object.
        sync (bool): Whether to enable synchronous mode.
        traffic_manager (carla.TrafficManager): Traffic manager object.
        vehicle_blueprint_library (carla.BlueprintLibrary): Vehicle blueprint library.
        pedestrian_blueprint_library (carla.BlueprintLibrary): Pedestrian blueprint library.
    """
    def __init__(self, client: carla.Client, config: SpawnManagerConfig) -> None:
        self.client = client
        self.config = config

        # Lists to track spawned vehicles and walkers
        self.hero = None
        self.vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []

        # Get the world from the client
        self.world = self.client.get_world()
        self.world.set_pedestrians_cross_factor(self.config.percentage_pedestrians_crossing)
        self.sync = self.world.get_settings().synchronous_mode

        # Set up the traffic manager
        self.client.get_trafficmanager(self.config.tm_port).shut_down()
        self.traffic_manager = self.client.get_trafficmanager(self.config.tm_port)
        self.traffic_manager.global_percentage_speed_difference(self.config.percentage_speed_difference)
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.config.distance_to_leading_vehicle)
        self.traffic_manager.set_respawn_dormant_vehicles(self.config.respawn_dormant_vehicles)
        self.traffic_manager.set_hybrid_physics_mode(self.config.hybrid_physics_mode)
        self.traffic_manager.set_hybrid_physics_radius(self.config.hybrid_physics_mode_radius)
        if self.config.seed is not None:
            self.traffic_manager.set_random_device_seed(self.config.seed)

        # Run traffic manager in sync mode with the server
        self.traffic_manager.set_synchronous_mode(self.sync)

        # Preload blueprint libraries
        self.vehicle_blueprint_library = self.world.get_blueprint_library().filter('vehicle.*')
        self.pedestrian_blueprint_library = self.world.get_blueprint_library().filter('walker.pedestrian.*')
 
    def spawn_vehicle(self, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform, is_hero: bool = False):
        """
        Spawn a vehicle in the simulation.

        Args:
            blueprint (carla.ActorBlueprint): Vehicle blueprint.
            spawn_point (carla.Transform): Spawn point for the vehicle.
            is_hero (bool, optional): Whether the vehicle is a hero vehicle. Defaults to False.

        Returns:
            carla.Actor: Spawned vehicle actor.
        """
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false' if not is_hero else 'true')
        blueprint.set_attribute('role_name', 'autopilot' if not is_hero else 'hero')
        vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        if not vehicle:
            return None
        
        vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.random_left_lanechange_percentage(vehicle, random.uniform(0, 5))
        self.traffic_manager.random_right_lanechange_percentage(vehicle, random.uniform(0, 5))
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 20))
        self.traffic_manager.update_vehicle_lights(vehicle, random.uniform() < self.config.percentage_vehicles_lights_on)
        return vehicle

    def spawn_hero(self, filter: str = "vehicle.tesla.model3"):
        """
        Spawn the hero vehicle.

        Args:
            filter (str, optional): Filter for the hero vehicle blueprint. Defaults to "vehicle.tesla.model3".

        Returns:
            carla.Actor: Spawned hero vehicle actor.
        """
        if self.hero is not None:
            raise RuntimeError("Hero has been spawned already")
        vehicles = self.spawn_vehicles(1, filter, is_hero=True)
        if len(vehicles) == 0:
            raise RuntimeError("Failed to spawn hero")
        self.hero = vehicles[0]
        return self.hero

    def spawn_vehicles(self, number_of_vehicles: int, filter: str = "vehicle.*", is_hero: bool = False):
        """
        Spawn multiple vehicles in the simulation.

        Args:
            number_of_vehicles (int): Number of vehicles to spawn.
            filter (str, optional): Filter for the vehicle blueprints. Defaults to "vehicle.*".
            is_hero (bool, optional): Whether the vehicles are hero vehicles. Defaults to False.

        Returns:
            list[carla.Actor]: List of spawned vehicle actors.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        blueprint_library = self.vehicle_blueprint_library.filter(filter)
        blueprints = [bp for bp in blueprint_library if bp.get_attribute("base_type") not in []]

        vehicles = []
        for spawn_point in spawn_points:
            if len(vehicles) >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            vehicle = self.spawn_vehicle(blueprint, spawn_point, is_hero)
            if vehicle:
                vehicles.append(vehicle)
        if len(vehicles) < number_of_vehicles:
            logging.warn(f"Failed to spawn {number_of_vehicles - len(vehicles)} vehicles.")
        logging.info(f"Successfully spawned {len(vehicles)} vehicles.")
        self.vehicles.extend(vehicles)
        return vehicles

    def spawn_pedestrian(self, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform):
        """
        Spawn a pedestrian in the simulation.

        Args:
            blueprint (carla.ActorBlueprint): Pedestrian blueprint.
            spawn_point (carla.Transform): Spawn point for the pedestrian.

        Returns:
            tuple[carla.Actor, carla.Actor]: Spawned pedestrian actor and controller.
        """
        pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)
        if not pedestrian:
            return None, None
        
        # Spawn the walker controller
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
        if not controller:
            pedestrian.destroy()
            return None, None
        return pedestrian, controller

    def spawn_pedestrians(self, num_pedestrians: int):
        """
        Spawn multiple pedestrians in the simulation.

        Args:
            num_pedestrians (int): Number of pedestrians to spawn.

        Returns:
            tuple[list[carla.Actor], list[carla.Actor]]: List of spawned pedestrians and controllers.
        """
        pedestrians = []
        pedestrian_controllers = []

        while len(pedestrians) < num_pedestrians:
            blueprint = random.choice(self.pedestrian_blueprint_library)
            spawn_point = carla.Transform(self.world.get_random_location_from_navigation())
            pedestrian, controller = self.spawn_pedestrian(blueprint, spawn_point)
            if not pedestrian or not controller:
                continue
            pedestrians.append(pedestrian)
            pedestrian_controllers.append(controller)

        # IMPORTANT NOTE: We need to tick the world once to make the controllers available in sync mode
        if self.sync:
            self.world.tick()

        # Then, activate the controllers
        for controller in pedestrian_controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

            # Get the pedestrian blueprint to access recommended speed values
            pedestrian = controller.parent
            pedestrian_bp = self.world.get_blueprint_library().find(pedestrian.type_id)
            if pedestrian_bp.has_attribute('speed'):
                recommended_speeds = pedestrian_bp.get_attribute('speed').recommended_values
                speed = recommended_speeds[1] if random.random() > self.config.percentage_pedestrians_running else recommended_speeds[0]
                controller.set_max_speed(float(speed))
            else:
                controller.set_max_speed(0.0)  # Fallback to previous method

        self.pedestrians.extend(pedestrians)
        self.pedestrian_controllers.extend(pedestrian_controllers)
        logging.info(f"Successfully spawned {len(self.pedestrians)} pedestrians.")

        return pedestrians, pedestrian_controllers

    def destroy(self):
        """Destroy all spawned actors and shut down the traffic manager."""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        for  controller in self.pedestrian_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
        for pedestrian in self.pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()
        self.traffic_manager.shut_down()