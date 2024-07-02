import os
import logging
from numpy import random
import carla
from pathlib import Path

from easycarla.utils.carla_helper import get_actor_blueprints
from easycarla.utils.algorithm import select_percentage_threshold

# Direct import from carla doesn't work 
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

class TrafficSimulation:

    def __init__(self, 
            client: carla.Client,
            seed: int = None,
            fps: int = 30,
            image_width: int = 800,
            image_height: int = 600) -> None:
        
        # Init parameters 
        self.client = client
        self.seed = seed
        self.fps = fps
        self.image_width = image_width
        self.image_height = image_height

        self.tm_port = 8000
        self.respawn_dormant_vehicles = True
        self.hybrid_physics_mode = False
        self.hybrid_physics_mode_radius = 70.0
        self.percentage_pedestrians_crossing = 0.3
        self.percentage_pedestrians_running = 0.05
        self.percentage_speed_difference = 30.0
        self.distance_to_leading_vehicle = 2.5
        self.percentage_vehicles_lights_on = 0.5
        self.hero = True

        # Lists to track spawned vehicles and walkers
        self.sensors_list = []
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        # Get the world from the client
        self.world = self.client.get_world()
        self.world.set_pedestrians_cross_factor(self.percentage_pedestrians_crossing)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.global_percentage_speed_difference(self.percentage_speed_difference)
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.distance_to_leading_vehicle)
        if self.respawn_dormant_vehicles:
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        if self.hybrid_physics_mode:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(self.hybrid_physics_mode_radius)
        if self.seed is not None:
            self.traffic_manager.set_random_device_seed(self.seed)

        # Run traffic manager in sync mode with the server!!!
        self.traffic_manager.set_synchronous_mode(True)
    
    def spawn_vehicles(self, number_of_vehicles: int):
        # Get blueprints for vehicles
        blueprints = get_actor_blueprints(self.world, "vehicle.*", "All")
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")

        # Sort blueprints by ID
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # Get available spawn points and shuffle them if needed
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print(f"Available spawn points: {number_of_spawn_points}")

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # Prepare the batch for spawning vehicles
        batch = []
        for index, transform in enumerate(spawn_points):
            if index == number_of_vehicles:
                break

            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # Add the spawn command to the batch
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        # Apply the batch and handle responses
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Turn on car lights
        vehicles_list_lights_on = select_percentage_threshold(self.vehicles_list, self.percentage_vehicles_lights_on)
        vehicle_actors_lights_on = self.world.get_actors(vehicles_list_lights_on)
        for actor in vehicle_actors_lights_on:
            self.traffic_manager.update_vehicle_lights(actor, True)

        print(f'Spawned {len(self.vehicles_list)} vehicles')
    
    def spawn_walkers(self, number_of_walkers: int):
        # Get blueprints for walkers
        blueprintsWalkers = get_actor_blueprints(self.world, "walker.pedestrian.*", "2")
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        # Set seed for randomness if provided
        if self.seed:
            self.world.set_pedestrians_seed(self.seed)
            random.seed(self.seed)
        
        # Generate spawn points for walkers
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # Prepare the batch for spawning walkers
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                if random.random() > self.percentage_pedestrians_running:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        
        # Apply the batch and handle responses
        results = self.client.apply_batch_sync(batch, False)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        
        # Spawn AI controllers for walkers
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, False)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        
        # Populate the all_id list with both walker and controller IDs
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # Start the walker controllers and set their destinations and speeds
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].start()
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print(f'Spawned {len(self.walkers_list)} walkers')

    def make_hero(self, hero_id: int = None, output_folder: Path|str = "data") -> tuple[carla.Actor, carla.Actor, carla.Actor, carla.Actor]:
        # Select the frist vehicle as hero
        hero_id = hero_id if hero_id is not None else self.vehicles_list[0]
        hero_vehicle = self.world.get_actor(hero_id)
        if hero_vehicle is None:
            raise ValueError(f"Hero vehicle with ID {hero_id} not found")
        hero_vehicle.set_simulate_physics(False)

        # Get the bounding box of the hero vehicle
        bounding_box = hero_vehicle.bounding_box
        extent = bounding_box.extent  # Get the dimensions of the bounding box

        # Calculate the positions for sensors based on the bounding box
        lidar_height = extent.z + 1.0  # Place the LiDAR 1.0 meters above the top of the vehicle
        camera_height = extent.z * 0.5  # Place the camera 1.0 meters above the top of the vehicle
        camera_forward = extent.x + 0.5  # Place the camera 0.5 meters forward of the vehicle's front

        blueprint_library = self.world.get_blueprint_library()

        # IMPORTAN:
        # - rotation_frequency must be at least fps to get full point clouds
        # - sensor_tick should be unset, so data is triggered at every frame 

        # Set up the LiDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('role_name', 'lidar')
        lidar_bp.set_attribute('range', '200')
        lidar_bp.set_attribute('rotation_frequency', str(self.fps))
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1200000')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-30')

        lidar = self.world.spawn_actor(
            lidar_bp, 
            carla.Transform(carla.Location(x=0, y=0, z=lidar_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors_list.append(lidar)

        # Set up the camera sensor
        camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        camera_rgb_bp.set_attribute('role_name', 'camera')
        camera_rgb_bp.set_attribute('image_size_x', '800')
        camera_rgb_bp.set_attribute('image_size_y', '600')
        camera_rgb_bp.set_attribute('fov', '120')  # Wide field of view
        camera_rgb_bp.set_attribute('enable_postprocess_effects', 'true')  # Enable HDR

        camera_rgb = self.world.spawn_actor(
            camera_rgb_bp, 
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors_list.append(camera_rgb)

        # Set up depth camera
        camera_depth_bp = blueprint_library.find('sensor.camera.depth')
        camera_depth_bp.set_attribute('role_name', 'camera_depth')
        camera_depth_bp.set_attribute('image_size_x', '800')
        camera_depth_bp.set_attribute('image_size_y', '600')
        camera_depth_bp.set_attribute('fov', '120')  # Wide field of view

        camera_depth = self.world.spawn_actor(
            camera_depth_bp,
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors_list.append(camera_depth)

        # Set up instance semantic segmentation
        camera_insemseg_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        camera_insemseg_bp.set_attribute('role_name', 'camera_insemseg')
        camera_insemseg_bp.set_attribute('image_size_x', '800')
        camera_insemseg_bp.set_attribute('image_size_y', '600')
        camera_insemseg_bp.set_attribute('fov', '120')  # Wide field of view

        camera_insemseg = self.world.spawn_actor(
            camera_insemseg_bp,
            carla.Transform(carla.Location(x=camera_forward, y=0, z=camera_height), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=hero_vehicle)
        self.sensors_list.append(camera_insemseg)

        print(f"Hero vehicle {hero_id} with LiDAR and Camera sensors set up")

        return lidar, camera_rgb, camera_depth, camera_insemseg

    def destroy(self):
        # Stop and destroy all sensors
        for sensor in self.sensors_list:
            sensor.stop()
            sensor.destroy()
        
        # Destroy all vehicles
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # Stop all walker controllers
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        # Destroy all walkers and their controllers
        print('\nDestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        
        # Clear all lists
        self.sensors_list.clear()
        self.vehicles_list.clear()
        self.walkers_list.clear()
