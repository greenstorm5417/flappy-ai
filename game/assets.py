import os
import pygame

sprites = {}

def load_sprites():
    path = os.path.join("assets")
    for file in os.listdir(path):
        if file.endswith(('.png')):
            sprite_name = file.split('.')[0]
            sprites[sprite_name] = pygame.image.load(os.path.join(path, file)).convert_alpha()

def get_sprite(name):
    sprite = sprites.get(name)
    if sprite is None:
        print(f"Sprite '{name}' not found! Please ensure it exists in 'assets/' directory.")
    return sprite

