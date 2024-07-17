import carla
import pygame
import logging
import random
import traceback
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from easycarla.visu.pygame_handler import PygameHandler
from easycarla.sim.simulation_manager import SimulationManager
from easycarla.sim.bounding_boxes import BoundingBoxes
from easycarla.utils.carla_helper import extract_image_rgb
from easycarla.sim.display_manager import DisplayManager, ScaleMode
from easycarla.sensors import Sensor, RgbSensor, LidarSensor, DepthSensor, MountingDirection, MountingPosition
from easycarla.sensors.world_sensor import WorldSensor
from easycarla.sim.spawn_manager import SpawnManager, SpawnManagerConfig

logging.basicConfig(level=logging.INFO)

host = '127.0.0.1'
port = 2000
client_timeout = 60
sync = True
fixed_delta_seconds = 0.05
fps = 30
timeout=0.01
num_vehicles = 30
num_pedestrians = 40
seed = 123
reset = False

def main():
    simulation_manager = None
    spawn_manager = None
    display_manager = None

    try:
        # Set seed for deterministic random
        random.seed(seed)

        # Setup simulation
        simulation_manager = SimulationManager(
            host=host,
            port=port,
            timeout=client_timeout,
            sync=sync, 
            fixed_delta_seconds=fixed_delta_seconds,
            reset=reset)
        
        # The, we can start our spawn manager to spawn vehicles before the simulation loop
        spawn_manager = SpawnManager(simulation_manager.client, SpawnManagerConfig(seed=seed))
        spawn_manager.spawn_vehicles(num_vehicles)
        spawn_manager.spawn_pedestrians(num_pedestrians)

        # Display Manager organizes all the sensors an its display in a window
        # It is easy to configure the grid and total window size
        # If fps is set here, the framerate will be max locked to it
        display_manager = DisplayManager(grid_size=[1, 1], fps=fps)

        # Choose hero vehicle
        hero = random.choice(spawn_manager.vehicles)
        
        # Spawn sensors
        world_sensor = WorldSensor(world=simulation_manager.world)
        rgb_sensor = RgbSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.FRONT, 
            mounting_direction=MountingDirection.FORWARD,
            image_size=[800,600])
        depth_sensor = DepthSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.FRONT, 
            mounting_direction=MountingDirection.FORWARD,
            image_size=[800,600])
        lidar_sensor = LidarSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.TOP, 
            mounting_direction=MountingDirection.FORWARD,
            image_size=[400,400],
            sensor_options={'channels' : '64', 'range' : '200',  'points_per_second': '250000', 'rotation_frequency': '30'})

        # Register sensors to be rendered
        display_manager.add_sensor(rgb_sensor, (0, 0), ScaleMode.ZOOM_CENTER)
        #display_manager.add_sensor(depth_sensor, (0, 2), ScaleMode.ZOOM_CENTER)
        #display_manager.add_sensor(lidar_sensor, (0, 0), ScaleMode.SCALE_FIT)

        sensors: list[Sensor] = [
            rgb_sensor, 
            depth_sensor,
            lidar_sensor, 
        ]

        # Remember the edge pairs
        edges = [[0,1], [1,3], [3,2], [2,0], 
                 [0,4], [4,5], [5,1], [5,7], 
                 [7,6], [6,4], [6,2], [7,3]]

        def process():
            # Consume sensor data
            try:
                world_sensor.consume()
                world_snapshot = world_sensor.sensor_data
                for sensor in sensors:
                    sensor.consume(world_snapshot.frame, timeout=1.0)
            except Exception as ex:
                logging.warning(f"Failed to consume on frame {world_snapshot.frame}")

            # Gizmo on hero vehicle
            world_sensor.world.debug.draw_arrow(
                            hero.get_transform().location, 
                            hero.get_transform().location + hero.get_transform().get_forward_vector(),
                            thickness=0.1, arrow_size=0.1, color=carla.Color(255,0,0), life_time=10)
            
            # Draw bounding boxes
            try:
                # Vectorize hero transform
                hero_pos = hero.get_transform().location
                hero_pos = np.array([hero_pos.x, hero_pos.y, hero_pos.z])
                hero_forward = hero.get_transform().get_forward_vector()
                hero_forward = np.array([hero_forward.x, hero_forward.y, hero_forward.z])

                # Retrieve bounding boxes
                bbs = world_sensor.get_bounding_boxes()
                if len(bbs) > 0:
                    # Filter bbs within squared distance
                    delta_pos = (bbs[:, 0] - hero_pos)
                    sq_dist = (delta_pos**2).sum(axis=1)
                    bbs = bbs[sq_dist < 50**2]
                    
                    # Filter bbs that are in front
                    delta_pos = (bbs[:, 0] - hero_pos)
                    bbs = bbs[delta_pos.dot(hero_forward) > 1]

                    # Project edges onto sensor
                    bbs_proj = rgb_sensor.project(bbs.reshape((-1, 3))).astype(int)
                    bbs_proj = bbs_proj.reshape((*bbs.shape[:-1], -1))

                    # Retrieve all edges from bounding boxes
                    bbs_proj_edges = bbs_proj[:, edges]

                    # Draw edges within image boundaries
                    image_width, image_height = rgb_sensor.image_size
                    for bb in bbs_proj_edges:
                        for edge in bb:
                            p1, p2 = edge
                            if 0 <= p1[0] < image_width and 0 <= p1[1] < image_height and \
                                0 <= p2[0] < image_width and 0 <= p2[1] < image_height:
                                cv2.line(rgb_sensor.decoded_data, (p1[0],p1[1]), (p2[0],p2[1]), (0,0,255), 1)
                
            except Exception as ex:
                traceback.print_exc()

            # Render data
            display_manager.draw_sensors()
            display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            display_manager.tick()

        while True:
            # Carla Tick
            simulation_manager.tick()
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(process)
            #process(world_sensor, sensors, display_manager)

            # Must listen to events to prevent unresponsive window
            if display_manager.should_quit():
                break

    except KeyboardInterrupt as ex:
        logging.warning('\nCancelled by user. Bye!')

    except Exception as ex:
        traceback.print_exc()
        raise ex
    
    finally:
        if spawn_manager:
            spawn_manager.destroy()
        if simulation_manager:
            simulation_manager.destroy()

if __name__ == '__main__':
    main()

