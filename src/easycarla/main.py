import carla
import pygame
import logging
import random
import traceback
import time
import cv2
import numpy as np
from easycarla.sim.simulation_manager import SimulationManager
from easycarla.visu.display_manager import DisplayManager, ScaleMode
from easycarla.sensors import Sensor, CameraSensor, LidarSensor, DepthCameraSensor, MountingDirection, MountingPosition
from easycarla.sensors.world_sensor import WorldSensor
from easycarla.sim.spawn_manager import SpawnManager, SpawnManagerConfig
from easycarla.labels import LabelManager, ObjectType

logging.basicConfig(level=logging.INFO)

# Enable human readable numpy output for debugging
np.set_printoptions(suppress=True)

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
distance = 50
show_points = False
output_dir = None #"data/kitti"

def main():
    simulation_manager = None
    spawn_manager = None
    display_manager = None
    label_manager = None

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

        # Choose hero vehicle
        hero = spawn_manager.spawn_hero()
        
        # Spawn sensors
        world_sensor = WorldSensor(world=simulation_manager.world)
        rgb_sensor = CameraSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.FRONT, 
            mounting_direction=MountingDirection.FORWARD,
            mounting_offset=0.0,
            image_size=[800,600])
        depth_sensor = DepthCameraSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.FRONT, 
            mounting_direction=MountingDirection.FORWARD,
            mounting_offset=0.0,
            image_size=[800,600])
        lidar_sensor = LidarSensor(world=simulation_manager.world, 
            attached_actor=hero,
            mounting_position=MountingPosition.TOP, 
            mounting_direction=MountingDirection.FORWARD,
            mounting_offset=0.5,
            image_size=[400,400],
            # Sensor tick of 0 will let the sensor create a scan for every world tick.
            # Combined with rotation frequency be the same as the simulation tick frequency
            # gives us one full rotation of the point cloud for every tick.
            # Other settings are similar to kitti velodyne sensors.
            sensor_options={
                'channels' : '64', 
                'range' : '120',  
                'points_per_second': '1300000', 
                'rotation_frequency': str(1/fixed_delta_seconds), 
                'sensor_tick': '0',
            })
        
        # Make sure to tick after sensors are created, so they are immediately available
        simulation_manager.tick()
    
        # Display Manager organizes all the sensors an its display in a window
        # It is easy to configure the grid and total window size
        # If fps is set here, the framerate will be max locked to it
        display_manager = DisplayManager(grid_size=[3, 1], fps=fps)
        display_manager.add_sensor(lidar_sensor, (0, 0), ScaleMode.SCALE_FIT)
        display_manager.add_sensor(rgb_sensor, (1, 0), ScaleMode.ZOOM_CENTER)
        display_manager.add_sensor(depth_sensor, (2, 0), ScaleMode.ZOOM_CENTER)

        sensors: list[Sensor] = [
            rgb_sensor, 
            depth_sensor,
            lidar_sensor,
        ]

        # Create label manager for 2d and 3d bounding boxes
        label_manager = LabelManager(
            world=simulation_manager.world, 
            carla_types={
                carla.CityObjectLabel.Pedestrians,
                carla.CityObjectLabel.Car,
                carla.CityObjectLabel.Bus,
                carla.CityObjectLabel.Truck,
                carla.CityObjectLabel.Motorcycle,
                carla.CityObjectLabel.Bicycle,
                carla.CityObjectLabel.Rider}, 
            camera_sensor=rgb_sensor, 
            depth_sensor=depth_sensor, 
            lidar_sensor=lidar_sensor,
            distance=distance,
            show_points=show_points,
            output_dir=output_dir)

        def process():
            # Consume first sensors, so all data is available after for the same frame
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
            
            # Retrieve bounding boxes
            label_manager.update()

            # Draw edges within image boundaries
            image_width, image_height = rgb_sensor.image_size
            if label_manager.bbse2d is None or label_manager.bbse2d.shape[0] == 0:
                return
            for bb in label_manager.bbse2d:
                for edge in bb:
                    p1, p2 = edge
                    ret, p1, p2 = cv2.clipLine((0, 0, image_width, image_height), p1.astype(int), p2.astype(int))
                    if ret:
                        cv2.line(rgb_sensor.preview_image, p1, p2, (0, 0, 255), 1)
    
            # Render data
            display_manager.draw_sensors()
            display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            display_manager.tick()

        while True:
            # Carla Tick
            simulation_manager.tick()
            
            process()

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

