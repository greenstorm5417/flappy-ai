# tests/test_genetic.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import pickle

from ai.genetic_controller import GeneticController
from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, PIPE_GAP_SIZE
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.layer import Layer
from game.state import get_game_state

pygame.init()

# Screen setup
display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird Genetic Algorithm Test")
clock = pygame.time.Clock()

# Surfaces
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
info_surface = pygame.Surface((SCREEN_WIDTH, 200))
nn_surface = pygame.Surface((600, window_height))  # Neural network visualization surface
x_pos = 50
y_pos = (window_height - SCREEN_HEIGHT) // 2
nn_x_pos = x_pos + SCREEN_WIDTH
nn_y_pos = 0

# Ensure neural network visualization fits within the screen
NN_WIDTH, NN_HEIGHT = 600, 600
if nn_x_pos + NN_WIDTH  > window_width:
    raise ValueError("Neural Network visualization window exceeds screen width. Adjust NN_WIDTH or window size.")

# Load game assets
load_sprites()

def clamp(value, min_value=0, max_value=255):
    """Clamp the value between min_value and max_value."""
    return max(min_value, min(int(value), max_value))

def draw_neural_network(screen, genome, input_size, hidden_sizes, output_size, input_labels, output_labels, position=(50, 50), scale=600):
    """
    Draws a neural network based on a GeneticController genome with labeled inputs and outputs.

    Args:
        screen (pygame.Surface): The surface to draw on.
        genome (list of np.ndarray): The genome representing weight matrices for each layer.
        input_size (int): Number of input nodes.
        hidden_sizes (list of int): Number of nodes in each hidden layer.
        output_size (int): Number of output nodes.
        input_labels (list of str): Descriptive labels for input nodes.
        output_labels (list of str): Descriptive labels for output nodes.
        position (tuple): Top-left position to start drawing.
        scale (int): Scaling factor for the visualization.
    """
    # Define layer information
    layers = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_sizes))] + ['Output']
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    total_layers = len(layers)

    # Calculate spacing
    horizontal_spacing = scale // (total_layers + 1)
    vertical_spacing = scale // (max(layer_sizes) + 1)

    # Determine node positions
    node_positions = {}
    for layer_idx, (layer_name, size) in enumerate(zip(layers, layer_sizes)):
        x = position[0] + horizontal_spacing * (layer_idx + 1)
        for node_idx in range(size):
            y = position[1] + vertical_spacing * (node_idx + 1)
            node_positions[(layer_idx, node_idx)] = (x, y)

    # Draw connections
    for layer_idx in range(total_layers - 1):
        for from_node in range(layer_sizes[layer_idx]):
            for to_node in range(layer_sizes[layer_idx + 1]):
                from_pos = node_positions[(layer_idx, from_node)]
                to_pos = node_positions[(layer_idx + 1, to_node)]
                weight = genome[layer_idx][from_node, to_node]
                # Determine color based on weight sign and magnitude
                if weight > 0:
                    color = (0, clamp(int(abs(weight) * 255)), 0)  # Greenish for positive weights
                else:
                    color = (clamp(int(abs(weight) * 255)), 0, 0)  # Reddish for negative weights
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
        elif layer_idx == total_layers -1:
            color = (0, 0, 255)    # Blue for Output
            label = output_labels[node_idx] if node_idx < len(output_labels) else f"O{node_idx}"
        else:
            color = (255, 255, 255)  # White for Hidden
            label = f"H{layer_idx}_{node_idx}"
        pygame.draw.circle(screen, color, pos, node_radius)
        label_surface = font.render(label, True, (255, 255, 255))
        label_rect = label_surface.get_rect(center=(pos[0], pos[1] - node_radius - 10))
        screen.blit(label_surface, label_rect)

    # Add layer titles
    title_font = pygame.font.SysFont(None, 24)
    layer_titles = {
        "input": "Inputs",
        "hidden": "Hidden",
        "output": "Outputs"
    }
    for layer_name, nodes in zip(layers, layer_sizes):
        # Skip if no nodes in this layer
        if nodes == 0:
            continue
        first_node_idx = 0  # Assuming at least one node exists
        first_pos = node_positions.get((layers.index(layer_name), first_node_idx))
        if first_pos:
            title = layer_titles.get(layer_name.lower(), layer_name)
            title_surface = title_font.render(title, True, (255, 255, 255))
            x_pos_title = first_pos[0] - title_surface.get_width() // 2
            screen.blit(title_surface, (x_pos_title, position[1] - 30))

def draw_bird_inputs(screen, bird, game_state):
    """
    Draws visual representations of the bird's inputs, including a green strip indicating the reward zone near the pipe gap.
    """
    if not game_state:
        return

    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']
    pipe_gap_size = game_state['pipe_gap_size']
    red = (255, 0, 0)
    yellow = (255, 255, 0)
    green = (0, 255, 0)
    dot_radius = 5
    reward_zone_margin = 30

    # Draw reward zone
    if next_pipe_x < SCREEN_WIDTH:
        top_of_reward_zone = max(0, next_pipe_gap_y - reward_zone_margin)
        bottom_of_reward_zone = min(SCREEN_HEIGHT, next_pipe_gap_y + reward_zone_margin)
        reward_zone_surface = pygame.Surface((SCREEN_WIDTH, bottom_of_reward_zone - top_of_reward_zone), pygame.SRCALPHA)
        reward_zone_surface.fill((0, 255, 0, 50))  # Green with ~20% opacity
        screen.blit(reward_zone_surface, (0, top_of_reward_zone))

    # Bird's Y Position
    pygame.draw.circle(screen, red, (int(bird.rect.x), int(bird_y)), dot_radius, 1)

    # Bird's Velocity
    vel_display_y = max(0, min(int(bird_y + bird_vel * 10), SCREEN_HEIGHT))
    pygame.draw.circle(screen, red, (int(bird.rect.x), vel_display_y), dot_radius)
    pygame.draw.line(screen, red, (int(bird.rect.x), int(bird_y)), (int(bird.rect.x), vel_display_y), 1)

    # Pipe Distance
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.circle(screen, yellow, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius, 1)
        pipe_gap_half = pipe_gap_size / 2
        top_pipe_y = next_pipe_gap_y - pipe_gap_half
        bottom_pipe_y = next_pipe_gap_y + pipe_gap_half
        pygame.draw.line(screen, red, (int(next_pipe_x), 0), (int(next_pipe_x), int(top_pipe_y)), 5)
        pygame.draw.line(screen, red, (int(next_pipe_x), SCREEN_HEIGHT), (int(next_pipe_x), int(bottom_pipe_y)), 5)
        pygame.draw.line(screen, red, (int(bird.rect.x), int(bird.rect.y)), (int(next_pipe_x), int(next_pipe_gap_y)), 1)

def draw_bird_info_text(info_surface, bird, game_state, score, generation, max_score):
    """
    Draws informational text about the bird and game state, including fitness and score.

    Args:
        info_surface (pygame.Surface): The surface for displaying information.
        bird (Bird): The Bird object.
        game_state (dict): The current game state.
        score (int): Current game score of the bird.
        generation (int): Current generation number.
        max_score (int): Maximum allowed score for the bird.
    """
    if not game_state:
        return
    bird_y = bird.rect.y
    bird_vel = bird.flap
    font = pygame.font.SysFont(None, 24)
    color = (0, 255, 0)  # Green color for genetic algorithm birds

    texts = [
        font.render(f"Generation: {generation}", True, color),
        font.render(f"Max Score: {max_score}", True, color),
        font.render(f"Bird Y Position: {bird_y}", True, color),
        font.render(f"Bird Velocity: {bird_vel:.2f}", True, color),
        font.render(f"Score: {score}", True, color)
    ]

    y_offset = 10
    info_surface.fill((0, 0, 0))  # Clear the surface
    for idx, text in enumerate(texts):
        info_surface.blit(text, (10, y_offset + idx * 30))

def reset_game(sprites, genome, genetic_controller):
    """
    Resets the game by clearing sprites and initializing background, floor, columns, and bird.

    Args:
        sprites (pygame.sprite.LayeredUpdates): The sprite group.
        genome (list of np.ndarray): The genome controlling the bird.
        genetic_controller (GeneticController): The GeneticController instance.

    Returns:
        Bird: The initialized Bird object.
        int: Timestamp of the last column spawn.
    """
    sprites.empty()

    # Initialize background and floor
    floor_image_width = get_sprite("floor").get_width()
    num_floors = (SCREEN_WIDTH // floor_image_width) + 2
    for i in range(num_floors):
        Background(i, sprites)
        Floor(i, sprites)

    # Add the first column
    sprites.add(Column(sprites))
    last_column_time = pygame.time.get_ticks()

    # Create Bird controlled by the genome
    bird = Bird(sprites, color=(0, 255, 0))  # Green color for GA bird
    bird.genome = genome
    bird.score = 0  # Initialize score
    bird.passed_columns = set()  # Initialize passed columns

    return bird, last_column_time

def main():
    # Initialize Genetic Controller
    genetic_controller = GeneticController(
        input_size=6,
        hidden_sizes=[10, 10],
        output_size=2,
        population_size=100,
        mutation_rate=0.1
    )

    # Load the best genome from the trained models
    best_genome_path = os.path.join('genetic_models\\pipegap_150_20241009_222924\\best_genome_gen_50.pkl')  # Update accordingly

    if not os.path.exists(best_genome_path):
        print(f"Best genome not found at '{best_genome_path}'. Please provide the correct path.")
        sys.exit(1)

    # Load the genome
    with open(best_genome_path, 'rb') as f:
        best_genome = pickle.load(f)

    # Initialize generation count (for display purposes)
    generation = 1  # Update if needed

    # Reset the game with the best genome
    sprites = pygame.sprite.LayeredUpdates()
    bird, last_column_time = reset_game(sprites, best_genome, genetic_controller)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Spawn new columns based on time
        current_time = pygame.time.get_ticks()
        if current_time - last_column_time > 1500:
            sprites.add(Column(sprites))
            last_column_time = current_time

        # Get current game state
        game_state = get_game_state(sprites, bird)

        # Prepare inputs for GA decision
        inputs = np.array([
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ])

        # Decide action using GeneticController
        action = genetic_controller.decide_action(best_genome, inputs)
        bird_action = 'flap' if action == 0 else 'no_flap'  # Assuming 0: Flap, 1: No Flap

        # Apply action
        if bird_action == 'flap':
            bird.flap = -6

        # Update all sprites
        sprites.update()

        # Check for passing columns and update score
        for sprite in sprites.get_sprites_from_layer(Layer.OBSTACLE):
            if isinstance(sprite, Column) and sprite.is_passed():
                if sprite not in bird.passed_columns:
                    bird.score += 1
                    bird.passed_columns.add(sprite)
                    # Optionally, you can add a fitness increment here if needed

        # Check for collision and reset if necessary
        if bird.check_collision(sprites):
            print("Collision detected! Resetting the game.")
            bird, last_column_time = reset_game(sprites, best_genome, genetic_controller)
            generation += 1  # Increment generation if applicable
            continue

        # Draw everything on the game_surface
        game_surface.fill((0, 0, 0))
        sprites.draw(game_surface)
        draw_bird_inputs(game_surface, bird, game_state)
        draw_bird_info_text(info_surface, bird, game_state, score=bird.score, generation=generation, max_score=generation * 5)

        # Draw neural network visualization
        nn_surface.fill((0, 0, 0))
        # Define input and output labels
        input_labels = [
            "Bird Y Position",
            "Bird Velocity",
            "Pipe Distance X",
            "Pipe Gap Y",
            "Dist to Top Pipe",
            "Dist to Bottom Pipe"
        ]
        output_labels = [
            "Flap",
            "No Flap"
        ]
        draw_neural_network(
            nn_surface,
            best_genome,
            input_size=genetic_controller.input_size,
            hidden_sizes=genetic_controller.hidden_sizes,
            output_size=genetic_controller.output_size,
            input_labels=input_labels,
            output_labels=output_labels,
            position=(50, 50),
            scale=600
        )

        # Render everything on the main screen
        screen.fill((0, 0, 0))
        screen.blit(info_surface, (x_pos, 0))
        screen.blit(game_surface, (x_pos, y_pos))
        screen.blit(nn_surface, (nn_x_pos, y_pos))
        pygame.display.flip()

        # Control the frame rate
        clock.tick(FPS)

if __name__ == "__main__":
    main()
