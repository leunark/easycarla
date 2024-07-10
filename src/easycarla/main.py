from easycarla.visu.pygame_handler import PygameHandler
from easycarla.sim.carla_handler import CarlaHandler

def main():
    with CarlaHandler() as carla_handler, PygameHandler() as pygame_handler:
        while True:
            world, world_snapshot, *sensors_data = carla_handler.tick()
            if not pygame_handler.tick():
                break
            pygame_handler.update_display(world, world_snapshot, *sensors_data)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
