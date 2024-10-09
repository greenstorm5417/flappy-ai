# score.py
import pygame.sprite
import game.assets
import game.configs
from game.layer import Layer
from game.column import Column  # Import Column to check instance

class Score(pygame.sprite.Sprite):
    def __init__(self, *groups):
        self._layer = Layer.UI
        self.value = 0
        self.font = pygame.font.SysFont(None, 48)  # Use a default system font
        self.image = self.font.render(str(self.value), True, (255, 255, 255))
        self.rect = self.image.get_rect(center=(game.configs.SCREEN_WIDTH / 2, 50))
        super().__init__(*groups)

    def update(self, *args, **kwargs):
        sprites_group = None
        if args:
            potential_group = args[0]
            if isinstance(potential_group, pygame.sprite.Group):
                sprites_group = potential_group
        elif 'sprites_group' in kwargs:
            potential_group = kwargs['sprites_group']
            if isinstance(potential_group, pygame.sprite.Group):
                sprites_group = potential_group

        if sprites_group is not None:
            # Increment score when passing a column
            for sprite in sprites_group.get_sprites_from_layer(Layer.OBSTACLE):
                if isinstance(sprite, Column) and sprite.is_passed():
                    self.value += 1
                    break  # Avoid multiple increments in one frame
            self.image = self.font.render(str(self.value), True, (255, 255, 255))
            self.rect = self.image.get_rect(center=(game.configs.SCREEN_WIDTH / 2, 50))
        else:
            # Optionally, handle cases where sprites_group is not provided
            pass

