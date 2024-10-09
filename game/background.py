# background.py
import pygame.sprite

import game.assets
import game.configs
from game.layer import Layer

class Background(pygame.sprite.Sprite):
    def __init__(self, index, *groups):
        self._layer = Layer.BACKGROUND
        original_image = game.assets.get_sprite("background")
        if original_image is None:
            raise ValueError("Sprite 'background' not found in assets.")

        # Scale the background image to match the screen size
        self.image = pygame.transform.scale(original_image, (game.configs.SCREEN_WIDTH, game.configs.SCREEN_HEIGHT))
        self.rect = self.image.get_rect(topleft=(game.configs.SCREEN_WIDTH * index, 0))

        super().__init__(*groups)

    def update(self, *args, **kwargs):
        self.rect.x -= 1  # Adjust speed if needed

        if self.rect.right <= 0:
            self.rect.x = game.configs.SCREEN_WIDTH
