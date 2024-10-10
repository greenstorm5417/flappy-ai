import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import torch
import torch.nn as nn  # For accessing the DQN model's layers

from ai.dql_controller import DQNController
from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.score import Score
from game.state import get_game_state
from game.layer import Layer

pygame.init()

# Screen setup
display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird DQL Test")
clock = pygame.time.Clock()

# Surfaces
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
info_surface = pygame.Surface((SCREEN_WIDTH, 200))
x_pos = 50
y_pos = (window_height - SCREEN_HEIGHT) // 2

# Neural network visualization surface
nn_surface = pygame.Surface((600, SCREEN_HEIGHT))
nn_x_pos = x_pos + SCREEN_WIDTH
nn_y_pos = y_pos

# Ensure neural network visualization fits within the screen
NN_WIDTH = 600
if nn_x_pos + NN_WIDTH > window_width:
    raise ValueError("Neural Network visualization window exceeds screen width. Adjust NN_WIDTH or window size.")

# Load sprites
load_sprites()

def clamp(value, min_value=0, max_value=255):
    return max(min_value, min(int(value), max_value))

def draw_dqn_neural_network(screen, model, input_labels, output_labels, position=(50, 50), scale=600, weight_threshold=0.5):
    """
    Draws the neural network architecture and weights from a PyTorch model.

    Args:
        screen (pygame.Surface): The surface to draw on.
        model (torch.nn.Module): The neural network model.
        input_labels (list of str): Labels for the input nodes.
        output_labels (list of str): Labels for the output nodes.
        position (tuple): Top-left position to start drawing.
        scale (int): Scaling factor for the visualization.
        weight_threshold (float): Minimum absolute weight value to display a connection.
    """
    # Extract layers and weights from the model
    layers = []
    layer_sizes = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            layers.append(module)
            layer_sizes.append(module.in_features)
    # Add output layer size
    layer_sizes.append(layers[-1].out_features)

    total_layers = len(layer_sizes) - 1  # Number of connections

    # Calculate spacing
    horizontal_spacing = scale // (total_layers + 1)
    vertical_spacing = scale // (max(layer_sizes) + 1)

    # Determine node positions
    node_positions = {}
    for layer_idx, size in enumerate(layer_sizes):
        x = position[0] + horizontal_spacing * (layer_idx)
        for node_idx in range(size):
            y = position[1] + vertical_spacing * (node_idx + 1)
            node_positions[(layer_idx, node_idx)] = (x, y)

    # Draw connections
    for layer_idx, layer in enumerate(layers):
        weights = layer.weight.detach().cpu().numpy()
        for from_idx in range(weights.shape[1]):
            for to_idx in range(weights.shape[0]):
                weight = weights[to_idx, from_idx]
                if abs(weight) < weight_threshold:
                    continue  # Skip drawing connections with small weights
                from_pos = node_positions[(layer_idx, from_idx)]
                to_pos = node_positions[(layer_idx + 1, to_idx)]
                # Determine color based on weight sign and magnitude
                if weight > 0:
                    color = (0, clamp(int(abs(weight) * 255), max_value=255), 0)  # Greenish for positive weights
                else:
                    color = (clamp(int(abs(weight) * 255), max_value=255), 0, 0)  # Reddish for negative weights
                # Line thickness based on weight magnitude
                thickness = max(1, int(abs(weight) * 3))
                pygame.draw.line(screen, color, from_pos, to_pos, thickness)

    # Draw nodes with labels
    node_radius = 10
    font = pygame.font.SysFont(None, 18)

    for (layer_idx, node_idx), pos in node_positions.items():
        if layer_idx == 0:
            color = (255, 165, 0)  # Orange for Input
            label = input_labels[node_idx] if node_idx < len(input_labels) else f"I{node_idx}"
        elif layer_idx == len(layer_sizes) - 1:
            color = (0, 0, 255)    # Blue for Output
            label = output_labels[node_idx] if node_idx < len(output_labels) else f"O{node_idx}"
        else:
            color = (255, 255, 255)  # White for Hidden
            label = f"H{layer_idx}_{node_idx}"
        pygame.draw.circle(screen, color, pos, node_radius)
        # Optional: Comment out labels to reduce drawing overhead
        label_surface = font.render(label, True, (255, 255, 255))
        label_rect = label_surface.get_rect(center=(pos[0], pos[1] - node_radius - 10))
        screen.blit(label_surface, label_rect)

def draw_bird_inputs(screen, bird, game_state):
    if not game_state:
        return

    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']
    pipe_gap_size = game_state['pipe_gap_size']

    # Colors and sizes
    red = (255, 0, 0)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    dot_radius = 5
    reward_zone_margin = 30

    # Reward zone
    if next_pipe_x < SCREEN_WIDTH:
        top = max(0, next_pipe_gap_y - reward_zone_margin)
        bottom = min(SCREEN_HEIGHT, next_pipe_gap_y + reward_zone_margin)
        reward_zone = pygame.Surface((SCREEN_WIDTH, bottom - top), pygame.SRCALPHA)
        reward_zone.fill((0, 255, 0, 50))
        screen.blit(reward_zone, (0, top))

    # Bird's Y Position
    pygame.draw.circle(screen, red, (int(bird.rect.x), int(bird_y)), dot_radius, 1)

    # Bird's Velocity
    vel_y = max(0, min(int(bird_y + bird_vel * 10), SCREEN_HEIGHT))
    pygame.draw.circle(screen, red, (int(bird.rect.x), vel_y), dot_radius)

    # Pipe Gap
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.circle(screen, yellow, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius, 1)
        pipe_gap_half = pipe_gap_size / 2
        top_pipe_y = next_pipe_gap_y - pipe_gap_half
        bottom_pipe_y = next_pipe_gap_y + pipe_gap_half
        pygame.draw.line(screen, red, (int(next_pipe_x), 0), (int(next_pipe_x), int(top_pipe_y)), 5)
        pygame.draw.line(screen, red, (int(next_pipe_x), SCREEN_HEIGHT), (int(next_pipe_x), int(bottom_pipe_y)), 5)

    # Connections
    pygame.draw.line(screen, red, (int(bird.rect.x), int(bird_y)), (int(bird.rect.x), vel_y), 1)
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.line(screen, red, (int(bird.rect.x), int(bird.rect.y)), (int(next_pipe_x), int(next_pipe_gap_y)), 1)

def draw_bird_info_text(info_surface, bird, game_state):
    if not game_state:
        return

    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']
    next_pipe_end_x = game_state.get('next_pipe_end_x', SCREEN_WIDTH + get_sprite("pipe-green").get_width())

    font = pygame.font.SysFont(None, 24)
    texts = [
        f"Bird Y Position: {bird_y}",
        f"Bird Velocity: {bird_vel:.2f}",
        f"Next Pipe Distance X: {next_pipe_x}",
        f"Next Pipe Gap Y: {next_pipe_gap_y}",
        f"Next Pipe End X: {next_pipe_end_x}"
    ]

    info_surface.fill((0, 0, 0))
    for idx, text in enumerate(texts):
        text_surf = font.render(text, True, (255, 255, 255))
        info_surface.blit(text_surf, (10, 10 + idx * 30))

def reset_game(sprites, controller):
    sprites.empty()
    for i in range((SCREEN_WIDTH // get_sprite("floor").get_width()) + 2):
        Background(i, sprites)
        Floor(i, sprites)
    sprites.add(Column(sprites))
    last_column_time = pygame.time.get_ticks()
    bird = Bird(sprites, controller=controller, color=(0, 0, 255))
    score = Score(sprites)
    return bird, last_column_time, score

def main():
    dqn_controller = DQNController(is_training=False)
    dqn_models_dir = 'dqn_models'
    os.makedirs(dqn_models_dir, exist_ok=True)
    model_path = os.path.join(dqn_models_dir, 'current_dql_model.pth')
    if not os.path.exists(model_path):
        print(f"Trained DQL model not found at '{model_path}'. Please train the model first.")
        sys.exit(1)
    dqn_controller.load_model(model_path)

    # Load sprites
    load_sprites()
    sprites = pygame.sprite.LayeredUpdates()
    bird, last_column_time, score = reset_game(sprites, dqn_controller)

    # Prepare input and output labels
    input_labels = [
        "Bird Y Position",
        "Bird Velocity",
        "Pipe Distance X",
        "Pipe Gap Y",
        "Dist to Top Pipe",
        "Dist to Bottom Pipe"
    ]
    output_labels = [
        "No Flap",
        "Flap"
    ]

    # Draw neural network visualization once
    nn_surface.fill((0, 0, 0))
    draw_dqn_neural_network(
        nn_surface,
        dqn_controller.policy_net,
        input_labels=input_labels,
        output_labels=output_labels,
        position=(50, 50),
        scale=500,
        weight_threshold=1.0  # Adjust this threshold to reduce connections
    )

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        # Game logic and updates
        current_time = pygame.time.get_ticks()
        if current_time - last_column_time > 1500:
            sprites.add(Column(sprites))
            last_column_time = current_time

        game_state = get_game_state(sprites, bird)
        state = np.array([
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ])
        action = dqn_controller.select_action(state)
        bird_action = 'flap' if action == 1 else 'no_flap'
        if bird_action == 'flap':
            bird.flap = -6

        sprites.update()
        if bird.check_collision(sprites):
            bird, last_column_time, score = reset_game(sprites, dqn_controller)

        # Drawing
        game_surface.fill((0, 0, 0))
        sprites.draw(game_surface)
        draw_bird_inputs(game_surface, bird, game_state)
        draw_bird_info_text(info_surface, bird, game_state)

        # Render everything on the main screen
        screen.fill((0, 0, 0))
        screen.blit(info_surface, (x_pos, 0))
        screen.blit(game_surface, (x_pos, y_pos))
        screen.blit(nn_surface, (nn_x_pos, nn_y_pos))
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
