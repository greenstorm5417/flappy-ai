# floor.py
import pygame.sprite

import game.assets
import game.configs
from game.layer import Layer

class Floor(pygame.sprite.Sprite):
    def __init__(self, index, *groups):
        self._layer = Layer.FLOOR
        self.image = game.assets.get_sprite("floor")
        if self.image is None:
            raise ValueError("Sprite 'floor' not found in assets.")
        
        # Get the width of the floor image
        floor_image_width = self.image.get_width()
        
        # Position each floor sprite based on its width
        self.rect = self.image.get_rect(bottomleft=(floor_image_width * index, game.configs.SCREEN_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        super().__init__(*groups)

    def update(self, *args, **kwargs):
        self.rect.x -= 4  # Increased speed to match columns

        # If the floor sprite has moved off the screen, reposition it to the right
        if self.rect.right <= 0:
            # Get the width of the floor image
            floor_image_width = self.image.get_width()
            
            # Reposition to the right of the last floor sprite
            self.rect.x += floor_image_width * len([s for s in self.groups()[0] if isinstance(s, Floor)])
