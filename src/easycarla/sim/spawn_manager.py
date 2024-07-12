import os
import logging
from numpy import random
import carla
from pathlib import Path
from dataclasses import dataclass, field

from easycarla.utils.carla_helper import get_actor_blueprints
from easycarla.utils.algorithm import select_percentage_threshold


@dataclass
class SpawnManagerConfig:
    seed: int = None
    sync: bool = False
    image_width: int = 800
    image_height: int = 600
    tm_port: int = 8000
    respawn_dormant_vehicles: bool = True
    hybrid_physics_mode: bool = False
    hybrid_physics_mode_radius: float = 70.0
    percentage_pedestrians_crossing: float = 0.3
    percentage_pedestrians_running: float = 0.05
    percentage_speed_difference: float = 30.0
    distance_to_leading_vehicle: float = 2.5
    percentage_vehicles_lights_on: float = 0.5
    output_folder: Path = field(default_factory=lambda: Path("data"))


class SpawnManager:

    def __init__(self, client: carla.Client, config: SpawnManagerConfig) -> None:
        self.client = client
        self.config = config

        # Lists to track spawned vehicles and walkers
        self.vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        self.sensors = []

        # Get the world from the client
        self.world = self.client.get_world()
        self.world.set_pedestrians_cross_factor(self.config.percentage_pedestrians_crossing)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(self.config.tm_port)
        self.traffic_manager.global_percentage_speed_difference(self.config.percentage_speed_difference)
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.config.distance_to_leading_vehicle)
        self.traffic_manager.set_respawn_dormant_vehicles(self.config.respawn_dormant_vehicles)
        self.traffic_manager.set_hybrid_physics_mode(self.config.hybrid_physics_mode)
        self.traffic_manager.set_hybrid_physics_radius(self.config.hybrid_physics_mode_radius)
        if self.config.seed is not None:
            self.traffic_manager.set_random_device_seed(self.config.seed)

        # Run traffic manager in sync mode with the server
        self.traffic_manager.set_synchronous_mode(self.config.sync)

        # Preload blueprint libraries
        self.vehicle_blueprint_library = self.world.get_blueprint_library().filter('vehicle.*')
        self.pedestrian_blueprint_library = self.world.get_blueprint_library().filter('walker.pedestrian.*')
 
    def spawn_vehicle(self, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform, is_hero: bool = False):
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
        
    def spawn_vehicles(self, number_of_vehicles: int):
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            if len(self.vehicles) >= number_of_vehicles:
                break
            blueprint = random.choice(self.vehicle_blueprint_library)
            vehicle = self.spawn_vehicle(blueprint, spawn_point)
            if vehicle:
                self.vehicles.append(vehicle)
        if len(self.vehicles) < number_of_vehicles:
            logging.warn(f"Failed to spawn {number_of_vehicles - len(self.vehicles)} vehicles.")

        logging.info(f"Successfully spawned {len(self.vehicles)} vehicles.")

    def spawn_pedestrian(self, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform):
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
        return pedestrian, controller

    def spawn_pedestrians(self, num_pedestrians: int):
        while len(self.pedestrians) < num_pedestrians:
            blueprint = random.choice(self.pedestrian_blueprint_library)
            spawn_point = carla.Transform(self.world.get_random_location_from_navigation())
            pedestrian, controller = self.spawn_pedestrian(blueprint, spawn_point)
            if not pedestrian or not controller:
                continue
            self.pedestrians.append(pedestrian)
            self.pedestrian_controllers.append(controller)

        logging.info(f"Successfully spawned {len(self.pedestrians)} pedestrians.")

    def spawn_hero(self):
        # Spawn new vehicle as hero
        self.hero = self.spawn_vehicle()

        hero_id = hero_id if hero_id is not None else self.vehicles[0].id
        hero_vehicle = self.vehicles[0]
        hero_vehicle.set_simulate_physics(False)

        # Get the bounding box of the hero vehicle
        bounding_box = hero_vehicle.bounding_box
        extent = bounding_box.extent  # Get the dimensions of the bounding box

        # Calculate the positions for sensors based on the bounding box
        lidar_height = extent.z + 1.0  # Place the LiDAR 1.0 meters above the top of the vehicle
        camera_height = extent.z * 0.5  # Place the camera 1.0 meters above the top of the vehicle
        camera_forward = extent.x + 0.5  # Place the camera 0.5 meters forward of the vehicle's front

        blueprint_library = self.world.get_blueprint_library()

        # Set up the LiDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('role_name', 'lidar')
        lidar_bp.set_attribute('range', '200')
        lidar_bp.set_attribute('rotation_frequency', 10)
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1200000')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-30')

        lidar = self.world.spawn_actor(
            lidar_bp, 
            carla.Transform(carla.Location(x=0, y=0, z=lidar_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors.append(lidar)

        # Set up the camera sensor
        camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        camera_rgb_bp.set_attribute('role_name', 'camera')
        camera_rgb_bp.set_attribute('image_size_x', str(self.config.image_width))
        camera_rgb_bp.set_attribute('image_size_y', str(self.config.image_height))
        camera_rgb_bp.set_attribute('fov', '120')  # Wide field of view
        camera_rgb_bp.set_attribute('enable_postprocess_effects', 'true')  # Enable HDR

        camera_rgb = self.world.spawn_actor(
            camera_rgb_bp, 
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors.append(camera_rgb)

        # Set up depth camera
        camera_depth_bp = blueprint_library.find('sensor.camera.depth')
        camera_depth_bp.set_attribute('role_name', 'camera_depth')
        camera_depth_bp.set_attribute('image_size_x', str(self.config.image_width))
        camera_depth_bp.set_attribute('image_size_y', str(self.config.image_height))
        camera_depth_bp.set_attribute('fov', '120')  # Wide field of view

        camera_depth = self.world.spawn_actor(
            camera_depth_bp,
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors.append(camera_depth)

        # Set up instance semantic segmentation
        camera_insemseg_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        camera_insemseg_bp.set_attribute('role_name', 'camera_insemseg')
        camera_insemseg_bp.set_attribute('image_size_x', str(self.config.image_width))
        camera_insemseg_bp.set_attribute('image_size_y', str(self.config.image_height))
        camera_insemseg_bp.set_attribute('fov', '120')  # Wide field of view

        camera_insemseg = self.world.spawn_actor(
            camera_insemseg_bp,
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors.append(camera_insemseg)

        logging.info(f"Hero vehicle {hero_vehicle.id} with LiDAR and Camera sensors set up")

        return lidar, camera_rgb, camera_depth, camera_insemseg

    def destroy(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        for controller in self.pedestrian_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()
        for pedestrian in self.pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()
        self.traffic_manager.shut_down()