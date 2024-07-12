import pygame
import numpy as np

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
        # Update display
        pygame.display.flip()
        # Tick the clock to calculate fps
        self.clock.tick()
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

