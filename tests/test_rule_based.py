import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import sys
import numpy as np
import os

from ai.rule_based_controller import RuleBasedController

from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.state import get_game_state
from game.layer import Layer

pygame.init()

# Get the current display resolution
display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h

# Initialize a borderless window
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird Rule-Based Training")
clock = pygame.time.Clock()

# Create a separate surface for the game
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

# Calculate top-left position to align game_surface 50 pixels from the left and centered vertically
x_pos = 50  # 50 pixels from the left edge
y_pos = (window_height - SCREEN_HEIGHT) // 2  # Centered vertically

INFO_AREA_HEIGHT = 200  # Adjust as needed
info_surface = pygame.Surface((SCREEN_WIDTH, INFO_AREA_HEIGHT))

load_sprites()

def draw_bird_inputs(screen, bird, game_state):
    """
    Draws visual representations of the bird's inputs with added text 'goal' next to the green dot.
    """
    if not game_state:
        return

    # Extract input parameters
    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']
    dist_to_top_pipe = game_state['dist_to_top_pipe']
    dist_to_bottom_pipe = game_state['dist_to_bottom_pipe']

    # Define colors
    red = (255, 0, 0)
    yellow = (255, 255, 0)
    green = (0, 255, 0)

    # Define sizes
    dot_radius = 5

    # 1. Bird's Y Position
    pygame.draw.circle(screen, red, (int(bird.rect.x), int(bird_y)), dot_radius)

    # 2. Bird's Velocity
    vel_display_y = int(bird_y + bird_vel * 10)
    vel_display_y = max(0, min(vel_display_y, SCREEN_HEIGHT))
    pygame.draw.circle(screen, red, (int(bird.rect.x), vel_display_y), dot_radius)

    # 3. Pipe Distance and 'goal' text
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.circle(screen, green, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius)
        pygame.draw.circle(screen, yellow, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius, 1)  # Outline

        # Draw a line between the bird and the next pipe gap
        pygame.draw.line(screen, green, (int(bird.rect.x), int(bird.rect.y)), (int(next_pipe_x), int(next_pipe_gap_y)), 1)

        # Add 'goal' text next to the green dot
        font = pygame.font.SysFont(None, 24)  # Define the font (you can adjust the size as needed)
        goal_text = font.render('goal', True, (255, 255, 255))  # White text
        screen.blit(goal_text, (int(next_pipe_x) + 10, int(next_pipe_gap_y) - 10))  # Adjust position slightly to the right of the dot

    # Bird to Velocity Line
    pygame.draw.line(screen, red, (int(bird.rect.x), int(bird_y)), (int(bird.rect.x), vel_display_y), 1)

def draw_bird_info_text(info_surface, bird, game_state):
    if not game_state:
        return

    # Extract input parameters
    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']

    # Define font
    font = pygame.font.SysFont(None, 24)

    # Prepare text surfaces
    bird_y_text = font.render(f"Bird Y Position: {bird_y}", True, (255, 255, 255))
    bird_vel_text = font.render(f"Bird Velocity: {bird_vel:.2f}", True, (255, 255, 255))
    pipe_dist_text = font.render(f"Next Pipe Distance X: {next_pipe_x}", True, (255, 255, 255))
    pipe_gap_text = font.render(f"Next Pipe Gap Y: {next_pipe_gap_y}", True, (255, 255, 255))

    # Clear the info_surface
    info_surface.fill((0, 0, 0))

    # Blit text surfaces onto the info_surface
    info_surface.blit(bird_y_text, (10, 10))
    info_surface.blit(bird_vel_text, (10, 40))
    info_surface.blit(pipe_dist_text, (10, 70))
    info_surface.blit(pipe_gap_text, (10, 100))

def reset_game(sprites, controller):
    # Clear all sprites
    sprites.empty()
    
    # Recreate background and floor
    floor_image_width = get_sprite("floor").get_width()
    num_floors = (SCREEN_WIDTH // floor_image_width) + 2
    for i in range(num_floors):
        Background(i, sprites)
        Floor(i, sprites)
    
    # Reset columns by adding the first Column
    sprites.add(Column(sprites))
    last_column_time = pygame.time.get_ticks()
    
    # Reset bird
    bird = Bird(sprites, controller=controller, color=(255, 255, 0))  # Yellow bird for Rule-Based

    return bird, last_column_time

def train_rule_based():
    # Define rules for the controller (Customizable)
    rules = {
        'flap_if_below_gap': True,
        'flap_if_descending': False,
        'flap_if_close_to_pipe': False,
        'distance_threshold': 50,  # Pixels
    }

    # Initialize Rule-Based Controller with custom rules
    rule_based_controller = RuleBasedController(rules=rules)

    sprites = pygame.sprite.LayeredUpdates()
    bird, last_column_time = reset_game(sprites, rule_based_controller)

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Implement column spawning based on elapsed time
        current_time = pygame.time.get_ticks()
        if current_time - last_column_time > 1500:
            sprites.add(Column(sprites))
            last_column_time = current_time

        # Get current game state
        game_state = get_game_state(sprites, bird)

        # Decide action based on rules
        action = rule_based_controller.decide_action(bird, game_state)
        if action == 'flap':
            bird.flap = -6  # Bird jumps

        # Update all sprites
        sprites.update(game_state)

        # Check collisions and handle game over
        if bird.check_collision(sprites):
            # Reset game
            bird, last_column_time = reset_game(sprites, rule_based_controller)

        # Draw everything on the game_surface
        game_surface.fill((0, 0, 0))
        sprites.draw(game_surface)

        # Draw bird inputs on the game_surface
        draw_bird_inputs(game_surface, bird, game_state)

        # Draw text information on the info_surface
        draw_bird_info_text(info_surface, bird, game_state)

        # Fill the main screen with black
        screen.fill((0, 0, 0))

        # Blit the info_surface onto the main screen at the top
        screen.blit(info_surface, (x_pos, 0))

        # Blit the game_surface onto the main screen below the info area
        screen.blit(game_surface, (x_pos, y_pos))

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    train_rule_based()
