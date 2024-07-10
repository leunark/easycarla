import pygame
import carla
import numpy as np
from easycarla.sim.bounding_boxes import BoundingBoxes


class PygameHandler:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

    def __enter__(self):
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = self.get_font()
        self.clock = pygame.time.Clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pygame.quit()

    def tick(self) -> bool:
        if self.should_quit():
            return False
        self.clock.tick()
        return True

    def update_display(self, world, snapshot, lidar, image_rgb, image_depth, image_insemseg):

        # Visualize display stats
        fps_simulated = round(1.0 / snapshot.timestamp.delta_seconds)
        fps_real = self.clock.get_fps()

        self.draw_image(self.display, image_rgb)
        self.draw_image(self.display, image_insemseg, blend=True)
        self.display.blit(
            self.font.render(f'{fps_real} FPS (real)', True, (255, 255, 255)),
            (8, 10))
        self.display.blit(
            self.font.render(f'{fps_simulated} FPS (simulated)', True, (255, 255, 255)),
            (8, 28))

        # Visualize filtered bounding boxes
        vehicles = [actor for actor in world.get_actors().filter('vehicle.*')]
        bb_boxes = BoundingBoxes.get_camera_bounding_boxes(vehicles, image_rgb)
        bb_boxes = BoundingBoxes.filter_occluded(bb_boxes, image_depth)
        BoundingBoxes.draw_bounding_boxes(self.display, bb_boxes, image_depth)
        print(f"Detected {len(bb_boxes)} bounding boxes of {len(vehicles)}")
        
        pygame.display.flip()

    @staticmethod
    def draw_image(surface: pygame.Surface, image: carla.Image, blend: bool = False) -> None:
        """Draw the CARLA sensor image on a Pygame surface."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Keep only RGB channels
        array = array[:, :, ::-1]  # Convert from BGR to RGB
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))

    @staticmethod
    def get_font() -> pygame.font.Font:
        """Get the default font for Pygame."""
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)

    @staticmethod
    def should_quit() -> bool:
        """Check if the Pygame window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

