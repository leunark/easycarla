import carla
import pygame
import logging
import random
from easycarla.visu.pygame_handler import PygameHandler
from easycarla.sim.simulation_manager import SimulationManager
from easycarla.sim.bounding_boxes import BoundingBoxes
from easycarla.utils.carla_helper import extract_image_rgb
from easycarla.sim.display_manager import DisplayManager
from easycarla.sim.sensor_manager import Sensor, SensorType, MountingPosition, ScaleMode
from easycarla.sim.spawn_manager import SpawnManager, SpawnManagerConfig

logging.basicConfig(level=logging.INFO)

host = '127.0.0.1'
port = 2000
timeout = 40
sync = True
fixed_delta_seconds = 0.05
num_vehicles = 30
num_pedestrians = 40
seed = 123
reset = False

def main():

    simulation_manager = None
    spawn_manager = None
    display_manager = None

    try:
        # Setup simulation 
        simulation_manager = SimulationManager(
            host=host,
            port=port,
            timeout=timeout,
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
        display_manager = DisplayManager(grid_size=[1, 4], fps=30)

        # SensorManager spawns RGBCamera, DepthCamera, LiDARs, ... as needed
        # and assign each of them to a grid position,
        hero = random.choice(spawn_manager.vehicles)
        Sensor(world=simulation_manager.world, 
            display_man=display_manager,
            display_pos=[0, 0],
            sensor_type=SensorType.LIDAR,
            transform=carla.Transform(Sensor.get_mounting_position(hero, MountingPosition.TOP), carla.Rotation(yaw=+00)), 
            attached_actor=hero,
            image_size=[800,600],
            scale_mode=ScaleMode.NORMAL,
            sensor_options={'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '50'})
        Sensor(world=simulation_manager.world, 
            display_man=display_manager,
            display_pos=[0, 1],
            sensor_type=SensorType.CAMERA_RGB,
            transform=carla.Transform(Sensor.get_mounting_position(hero, MountingPosition.FRONT), carla.Rotation(yaw=+00)), 
            attached_actor=hero,
            image_size=[800,600],
            scale_mode=ScaleMode.ZOOM_CENTER,
            sensor_options={})
        Sensor(world=simulation_manager.world, 
            display_man=display_manager,
            display_pos=[0, 2],
            sensor_type=SensorType.CAMERA_DEPTH_CAMERA,
            transform=carla.Transform(Sensor.get_mounting_position(hero, MountingPosition.FRONT), carla.Rotation(yaw=+00)), 
            attached_actor=hero,
            image_size=[800,600],
            scale_mode=ScaleMode.ZOOM_CENTER,
            sensor_options={})
        Sensor(world=simulation_manager.world, 
            display_man=display_manager,
            display_pos=[0, 3],
            sensor_type=SensorType.CAMERA_INSTANCE_SEGMENTATION,
            transform=carla.Transform(Sensor.get_mounting_position(hero, MountingPosition.FRONT), carla.Rotation(yaw=+00)), 
            attached_actor=hero,
            image_size=[800,600],
            scale_mode=ScaleMode.ZOOM_CENTER,
            sensor_options={})

        while True:
            # Carla Tick
            world_snapshot = simulation_manager.tick()
            # Render received data
            display_manager.draw_sensors()
            display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            display_manager.tick()

    except KeyboardInterrupt as ex:
        logging.warn('\nCancelled by user. Bye!')

    except Exception as ex:
        logging.error(ex)
        raise ex
    
    finally:
        if display_manager:
            display_manager.destroy()
        if spawn_manager:
            spawn_manager.destroy()
        if simulation_manager:
            simulation_manager.destroy()

if __name__ == '__main__':
    main()



#            # Visualize display stats
#            pygame_handler.draw_fps(world_snapshot.timestamp.delta_seconds)
#
#            # Draw images
#            pygame_handler.draw_image(extract_image_rgb(image_rgb))
#            pygame_handler.draw_image(extract_image_rgb(image_insemseg), blend=True)
#
#            # Test
#            bbs = world.get_level_bbs(carla.CityObjectLabel.Car)
#            
#            # Visualize filtered bounding boxes
#            vehicles = [actor for actor in world.get_actors().filter('vehicle.*')]
#            bb_boxes = BoundingBoxes.get_camera_bounding_boxes(vehicles, image_rgb)
#            bb_boxes = BoundingBoxes.filter_occluded(bb_boxes, image_depth)
#            pygame_handler.draw_bounding_boxes(bb_boxes)
#            print(f"Detected {len(bb_boxes)} bounding boxes of {len(vehicles)}")