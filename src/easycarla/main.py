import carla
import logging
import random
import traceback
import numpy as np
from easycarla.sim import SimulationManager
from easycarla.visu import DisplayManager, ScaleMode
from easycarla.sensors import Sensor, CameraSensor, LidarSensor, DepthCameraSensor, MountingDirection, MountingPosition, WorldSensor
from easycarla.sim import SpawnManager, SpawnManagerConfig
from easycarla.labels import LabelManager
import argparse

logging.basicConfig(level=logging.INFO)

# Enable human readable numpy output for debugging
np.set_printoptions(suppress=True)

def parse_args():
    """
    Parse command line arguments for the CARLA simulation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="CARLA simulation script with adjustable parameters")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=2000, help='Port number')
    parser.add_argument('--client_timeout', type=int, default=60, help='Client timeout in seconds')
    parser.add_argument('--sync', type=bool, default=True, help='Enable synchronous mode')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.05, help='Fixed delta seconds')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    parser.add_argument('--timeout', type=float, default=0.01, help='Timeout')
    parser.add_argument('--num_vehicles', type=int, default=30, help='Number of vehicles')
    parser.add_argument('--num_pedestrians', type=int, default=40, help='Number of pedestrians')
    parser.add_argument('--seed', type=int, default=999, help='Random seed')
    parser.add_argument('--reset', type=bool, default=False, help='Reset simulation')
    parser.add_argument('--distance', type=int, default=50, help='Distance for labels')
    parser.add_argument('--show_points', type=bool, default=True, help='Show points in visualization')
    parser.add_argument('--show_gizmo', type=bool, default=False, help='Show gizmo')
    parser.add_argument('--output_dir', type=str, default="data/kitti", help='Output directory')
    parser.add_argument('--frame_interval', type=int, default=20, help='Frame interval for export')
    parser.add_argument('--frame_count', type=int, default=100, help='Number of frames to export')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio')
    return parser.parse_args()

def main():
    """
    Main function to set up and run the CARLA simulation with adjustable parameters.
    """
    args = parse_args()

    simulation_manager = None
    spawn_manager = None
    display_manager = None
    label_manager = None

    try:
        # Set seed for deterministic random
        random.seed(args.seed)

        # Setup simulation
        simulation_manager = SimulationManager(
            host=args.host,
            port=args.port,
            timeout=args.client_timeout,
            sync=args.sync, 
            fixed_delta_seconds=args.fixed_delta_seconds,
            reset=args.reset)
        
        # The, we can start our spawn manager to spawn vehicles before the simulation loop
        spawn_manager = SpawnManager(simulation_manager.client, SpawnManagerConfig(seed=args.seed))
        spawn_manager.spawn_vehicles(args.num_vehicles)
        spawn_manager.spawn_pedestrians(args.num_pedestrians)

        # Spawn hero vehicle
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
                'rotation_frequency': str(1/args.fixed_delta_seconds), 
                'sensor_tick': '0',
            })
        
        # Make sure to tick after sensors are created, so they are immediately available
        simulation_manager.tick()
    
        # Display Manager organizes all the sensors an its display in a window
        # It is easy to configure the grid and total window size
        # If fps is set here, the framerate will be max locked to it
        display_manager = DisplayManager(grid_size=[2, 4], fps=args.fps)
        # Grid rect is (row, column, row_offset, column_offset)
        display_manager.add_sensor(lidar_sensor, grid_rect=(0, 0, 2, 1), scale_mode=ScaleMode.SCALE_FIT)
        display_manager.add_sensor(rgb_sensor, grid_rect=(1, 1, 1, 3), scale_mode=ScaleMode.ZOOM_CENTER)
        display_manager.add_sensor(depth_sensor, grid_rect=(0, 1, 1, 3), scale_mode=ScaleMode.ZOOM_CENTER)

        sensors: list[Sensor] = [
            rgb_sensor, 
            depth_sensor,
            lidar_sensor,
        ]

        # Create label manager for 2d and 3d bounding boxes
        label_manager = LabelManager(
            carla_types={
                carla.CityObjectLabel.Pedestrians,
                carla.CityObjectLabel.Car,
                carla.CityObjectLabel.Bus,
                carla.CityObjectLabel.Truck,
                carla.CityObjectLabel.Motorcycle,
                #carla.CityObjectLabel.Bicycle, # Bicycle bounding boxes are buggy
                carla.CityObjectLabel.Rider}, 
            world_sensor=world_sensor,
            camera_sensor=rgb_sensor, 
            depth_sensor=depth_sensor, 
            lidar_sensor=lidar_sensor,
            distance=args.distance,
            show_points=args.show_points,
            output_dir=args.output_dir,
            frame_interval=args.frame_interval,
            frame_count=args.frame_count,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio)

        while True:
            # Carla Tick
            simulation_manager.tick()

            # Consume first sensors, so all data is available after for the same frame
            try:
                world_sensor.consume()
                world_snapshot = world_sensor.sensor_data
                for sensor in sensors:
                    sensor.consume(world_snapshot.frame, timeout=1.0)
            except Exception as ex:
                logging.warning(f"Failed to consume on frame {world_snapshot.frame}")
            
            # Gizmo on hero vehicle
            if args.show_gizmo:
                world_sensor.world.debug.draw_arrow(
                                hero.get_transform().location, 
                                hero.get_transform().location + hero.get_transform().get_forward_vector(),
                                thickness=0.1, arrow_size=0.1, color=carla.Color(255,0,0), life_time=10)
        
            # Retrieve bounding boxes
            label_manager.update()
    
            # Render data
            display_manager.draw_sensors()
            display_manager.draw_fps(world_snapshot.timestamp.delta_seconds)
            display_manager.tick()

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

