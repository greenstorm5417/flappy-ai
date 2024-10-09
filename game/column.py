# column.py
import random
import pygame.sprite
import game.assets
import game.configs
from game.layer import Layer

class Column(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.OBSTACLE
        self.gap = 150  # Adjusted gap for larger screen

        self.sprite = game.assets.get_sprite("pipe-green")
        if self.sprite is None:
            raise ValueError("Sprite 'pipe-green' not found in assets.")
        self.sprite_rect = self.sprite.get_rect()

        self.pipe_bottom = self.sprite
        self.pipe_bottom_rect = self.pipe_bottom.get_rect(topleft=(0, self.sprite_rect.height + self.gap))

        self.pipe_top = pygame.transform.flip(self.sprite, False, True)
        self.pipe_top_rect = self.pipe_top.get_rect(topleft=(0, 0))

        self.image = pygame.Surface((self.sprite_rect.width, self.sprite_rect.height * 2 + self.gap),
                                    pygame.SRCALPHA)
        self.image.blit(self.pipe_bottom, self.pipe_bottom_rect)
        self.image.blit(self.pipe_top, self.pipe_top_rect)

        sprite_floor_height = game.assets.get_sprite("floor").get_rect().height
        min_y = 100
        max_y = game.configs.SCREEN_HEIGHT - sprite_floor_height - 100

        self.rect = self.image.get_rect(midleft=(game.configs.SCREEN_WIDTH, random.uniform(min_y, max_y)))
        self.mask = pygame.mask.from_surface(self.image)

        self.passed = False

        super().__init__(*groups)

    def update(self, *args, **kwargs):
        self.rect.x -= 4  # Increased speed for larger screen

        if self.rect.right <= 0:
            self.kill()

    def is_passed(self):
        if self.rect.x < 50 and not self.passed:
            self.passed = True
            return True
        return False

    def get_gap_center_y(self):
        """
        Returns the y-coordinate of the center of the gap between the pipes.
        """
        return self.rect.y + self.pipe_top_rect.height + self.gap / 2
