# bird.py
import pygame.sprite

import game.assets
import game.configs
from game.layer import Layer
from game.column import Column
from game.floor import Floor

class Bird(pygame.sprite.Sprite):
    def __init__(self, *groups, controller=None, color=(255, 255, 255)):
        self._layer = Layer.PLAYER

        # Load the bird sprite
        original_image = game.assets.get_sprite("redbird-midflap")
        if original_image is None:
            raise ValueError("Sprite 'redbird-midflap' not found in assets.")
        
        # Tint the bird based on controller for differentiation
        self.image = original_image.copy()
        tint_surface = pygame.Surface(self.image.get_size(), pygame.SRCALPHA)
        tint_surface.fill(color)
        self.image.blit(tint_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        self.rect = self.image.get_rect(topleft=(-50, 200))
        self.mask = pygame.mask.from_surface(self.image)
        self.flap = 0
        self.controller = controller  # Assign the AI controller

        super().__init__(*groups)

    def update(self, game_state=None):
        # Apply gravity and update position
        self.flap += game.configs.GRAVITY
        self.rect.y += self.flap

        # Horizontal movement (optional, based on your game design)
        if self.rect.x < 50:
            self.rect.x += 3

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.flap = -6  # Manual flap for demonstration

    def check_collision(self, sprites):
        for sprite in sprites:
            if isinstance(sprite, (Column, Floor)) and sprite.mask.overlap(self.mask, (self.rect.x - sprite.rect.x, self.rect.y - sprite.rect.y)):
                return True
        if self.rect.bottom >= game.configs.SCREEN_HEIGHT or self.rect.top <= 0:
            return True
        return False

    def reset(self):
        self.rect.topleft = (-50, 50)
        self.flap = 0
        if self.controller:
            self.controller.reset()
