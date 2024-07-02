from easycarla.visu.pygame_handler import PygameHandler
from easycarla.sim.carla_handler import CarlaHandler

def main():
    with CarlaHandler() as carla_handler, PygameHandler() as pygame_handler:
        while True:
            if not pygame_handler.tick():
                break

            snapshot, *sensors_data = carla_handler.tick()
            
            pygame_handler.update_display(snapshot, *sensors_data)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
