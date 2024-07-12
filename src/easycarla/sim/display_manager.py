import pygame
import numpy as np
from enum import Enum

class ScaleMode(Enum):
    NORMAL = 1
    ZOOM_CENTER = 2
    STRETCH_FIT = 3

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
        
        self.sensor_list = []

    def get_window_size(self):
        return (int(self.window_size[0]), int(self.window_size[1]))

    def get_display_size(self):
        return (int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0]))

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return (int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1]))

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def draw_sensors(self):
        if self.render_enabled():
            for s in self.sensor_list:
                s.render()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None
    
    def tick(self) -> bool:
        # Must listen to events to prevent unresponsive window
        if self.should_quit():
            return False

        # Update display
        pygame.display.flip()

        # Tick the clock to measure frame times
        self.clock.tick(self.fps)

        return True        

    def draw_fps(self, delta_seconds: int):
        fps_simulated = round(1.0 / delta_seconds)
        fps_real = self.clock.get_fps()
        self.display.blit(
            self.font.render(f'{fps_real} FPS (real)', True, (255, 255, 255)),
            (8, 10))
        self.display.blit(
            self.font.render(f'{fps_simulated} FPS (simulated)', True, (255, 255, 255)),
            (8, 28))

    def draw_bounding_boxes(self, bounding_boxes: np.ndarray):
        """
        Draws bounding boxes on pygame display.
        """
        width, height = self.display.get_size()
        bb_surface = pygame.Surface((width, height))
        bb_surface.set_colorkey((0, 0, 0))
        bb_color = (248, 64, 24)
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, bb_color, points[0], points[1])
            pygame.draw.line(bb_surface, bb_color, points[1], points[2])
            pygame.draw.line(bb_surface, bb_color, points[2], points[3])
            pygame.draw.line(bb_surface, bb_color, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, bb_color, points[4], points[5])
            pygame.draw.line(bb_surface, bb_color, points[5], points[6])
            pygame.draw.line(bb_surface, bb_color, points[6], points[7])
            pygame.draw.line(bb_surface, bb_color, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, bb_color, points[0], points[4])
            pygame.draw.line(bb_surface, bb_color, points[1], points[5])
            pygame.draw.line(bb_surface, bb_color, points[2], points[6])
            pygame.draw.line(bb_surface, bb_color, points[3], points[7])
        self.display.blit(bb_surface, (0, 0))

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
    
    def draw_surface(self, surface: pygame.Surface, display_pos: tuple[int, int], scale_mode: ScaleMode):
        display_offset = self.get_display_offset(display_pos)
        display_size = self.get_display_size()
        dest_rect = pygame.Rect(*display_offset, *display_size)
        self.draw_surface_on_surface(surface, self.display, dest_rect, scale_mode)

    @staticmethod
    def draw_surface_on_surface(src_surface: pygame.Surface, dest_surface: pygame.Surface, dest_rect: pygame.Rect, scale_mode: ScaleMode):
        if scale_mode == ScaleMode.NORMAL:
            # Normal scaling, blit the source surface onto the destination surface
            src_rect = src_surface.get_rect()
            crop_rect = pygame.Rect(
                (src_rect.width - dest_rect.width) // 2, 
                (src_rect.height - dest_rect.height) // 2,
                dest_rect.width,
                dest_rect.height
            )
            src_surface = src_surface.subsurface(crop_rect)
            dest_surface.blit(src_surface, dest_rect.topleft)

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
