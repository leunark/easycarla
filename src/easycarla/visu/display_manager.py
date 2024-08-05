import pygame
import numpy as np
from enum import Enum
from easycarla.sensors.sensor import Sensor


class ScaleMode(Enum):
    NORMAL = 1
    SCALE_FIT = 2
    ZOOM_CENTER = 3
    STRETCH_FIT = 4

class DisplayManager:

    DEFAULT_DISPLAY_RESOLUTION_FACTOR = 0.75

    def __init__(self, grid_size: tuple[int, int], window_size: tuple[int, int] = None, fps: float = 0.0):
        """ If fps is 0, the framerate will be uncapped and the simulation runs as fast as it can. 
        Setting fps will lock the framerate by forcing a delay of 1/fps sec till the next tick.
        """
        self.grid_size = grid_size
        self.fps = fps

        # Create the window
        pygame.init()
        pygame.font.init()
        info = pygame.display.Info()
        self.window_size = (int(info.current_w * self.DEFAULT_DISPLAY_RESOLUTION_FACTOR), 
                            int(info.current_h * self.DEFAULT_DISPLAY_RESOLUTION_FACTOR)) if window_size is None else window_size
        self.display = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.font = self.get_font()

        self.sensors: list[tuple[Sensor, tuple[int, int, int, int], ScaleMode]] = []

    def get_window_size(self):
        return (int(self.window_size[0]), int(self.window_size[1]))

    def get_display_size(self):
        return (int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0]))

    def get_display_rect(self, grid_rect: tuple[int, int, int, int]):
        if grid_rect[0] < 0 or grid_rect[0] + grid_rect[2] > self.grid_size[0] or \
            grid_rect[1] < 0 or grid_rect[1] + grid_rect[3] > self.grid_size[1]:
            raise ValueError("Specified grid position lies outside the configured grid size")
        cell_width, cell_height = (
            int(self.window_size[0]/self.grid_size[1]), 
            int(self.window_size[1]/self.grid_size[0]))
        rect = (
            int(grid_rect[1] * cell_width),
            int(grid_rect[0] * cell_height),
            int(grid_rect[3] * cell_width), 
            int(grid_rect[2] * cell_height),
        )
        return rect

    def add_sensor(self, sensor: Sensor, grid_rect: tuple[int, int, int, int], scale_mode: ScaleMode = ScaleMode.NORMAL):
        self.sensors.append((sensor, grid_rect, scale_mode))

    def render_enabled(self):
        return self.display != None
    
    def tick(self) -> bool:
        # Must listen to events to prevent unresponsive window
        if self.should_quit():
            return False

        # Update display
        pygame.display.flip()
        
        # Clean the surface by filling it with black color
        self.display.fill((0, 0, 0))

        # Tick the clock to measure frame times
        self.clock.tick(self.fps)

        return True        

    def draw_image(self, image: np.ndarray, blend: bool = False) -> None:
        """Draw an rgb image on the Pygame surface. Input is of dimension h x w x rgb"""
        image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        self.display.blit(image_surface, (0, 0))

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
    
    def draw_points(self, points: np.ndarray):
        """
        Draw points onto the display.

        :param points: A numpy array of shape (n, 2), where n is the number of points.
                    Each point is represented as (x, y).
        """
        # Ensure the points array is two-dimensional and each point has two coordinates
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (n, 2)")

        for point in points:
            x, y = point
            pygame.draw.circle(self.display, (255, 255, 255), (int(x), int(y)), 2)

    def draw_fps(self, delta_seconds: int):
        fps_simulated = round(1.0 / delta_seconds)
        fps_real = self.clock.get_fps()
        self.display.blit(
            self.font.render(f'{fps_real:.1f} FPS (real)', True, (255, 255, 255)),
            (8, 10))
        self.display.blit(
            self.font.render(f'{fps_simulated:.1f} FPS (simulated)', True, (255, 255, 255)),
            (8, 28))

    def draw_sensors(self):
        for sensor, grid_rect, scale_mode in self.sensors:
            img = sensor.preview()
            if img is not None:
                # Pygame expects the image data to be in the shape (width, height, 3), not (height, width, 3)
                img = img.swapaxes(0, 1)
                surface = pygame.surfarray.make_surface(img)
                self.draw_surface(surface, grid_rect, scale_mode)

    def draw_surface(self, surface: pygame.Surface, grid_rect: tuple[int, int, int, int], scale_mode: ScaleMode):
        display_rect = self.get_display_rect(grid_rect)
        display_rect = pygame.Rect(*display_rect)
        self.draw_surface_on_surface(surface, self.display, display_rect, scale_mode)

    @staticmethod
    def draw_surface_on_surface(src_surface: pygame.Surface, dest_surface: pygame.Surface, dest_rect: pygame.Rect, scale_mode: ScaleMode):
        if scale_mode == ScaleMode.NORMAL:
            # Normal scaling, blit the source surface onto the destination surface
            dest_surface.blit(src_surface, dest_rect.topleft)

        elif scale_mode == ScaleMode.SCALE_FIT:
            # Scale the source surface to fit inside the destination rectangle
            src_rect = src_surface.get_rect()
            scale = min(dest_rect.width / src_rect.width, dest_rect.height / src_rect.height)
            src_surface = pygame.transform.scale(src_surface, (src_rect.width * scale, src_rect.height * scale))
            src_rect = src_surface.get_rect()
            dest = (dest_rect.topleft[0] + (dest_rect.width - src_rect.width) // 2, dest_rect.topleft[1] + (dest_rect.height - src_rect.height) // 2)
            dest_surface.blit(src_surface, dest)

        elif scale_mode == ScaleMode.ZOOM_CENTER:
            # Zoom and center the source surface to fit inside the destination rectangle
            src_rect = src_surface.get_rect()
            scale = max(dest_rect.width / src_rect.width, dest_rect.height / src_rect.height)
            src_surface = pygame.transform.scale(src_surface, (src_rect.width * scale, src_rect.height * scale))
            src_rect = src_surface.get_rect()
            crop_rect = pygame.Rect(
                (src_rect.width - dest_rect.width) // 2, 
                (src_rect.height - dest_rect.height) // 2,
                dest_rect.width,
                dest_rect.height
            )
            src_surface = src_surface.subsurface(crop_rect)
            dest_surface.blit(src_surface, dest_rect.topleft)

        elif scale_mode == ScaleMode.STRETCH_FIT:
            # Stretch the source surface to fit the destination rectangle
            src_surface = pygame.transform.scale(src_surface, (dest_rect.width, dest_rect.height))
            dest_surface.blit(src_surface, dest_rect.topleft)
