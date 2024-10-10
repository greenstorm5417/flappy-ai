# train\genetic_train.py

import sys
import os
import pygame
import numpy as np
import random
import copy
import pickle
import datetime  # Added for timestamp-based unique identifier

# Ensure that the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.genetic_controller import GeneticController
from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, PIPE_GAP_SIZE
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.layer import Layer
from game.state import get_game_state

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

def draw_bird_info_text(info_surface, bird, game_state, fitness, score, generation, max_score):
    """
    Draws informational text about the bird and game state, including fitness and score.

    Args:
        info_surface (pygame.Surface): The surface for displaying information.
        bird (Bird): The Bird object.
        game_state (dict): The current game state.
        fitness (float): Current fitness score of the genome.
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
        font.render(f"Fitness: {fitness:.2f}", True, color),
        font.render(f"Score: {score}", True, color)
    ]

    y_offset = 10
    info_surface.fill((0, 0, 0))  # Clear the surface
    for idx, text in enumerate(texts):
        info_surface.blit(text, (10, y_offset + idx * 30))

def reset_game(sprites):
    """
    Resets the game by clearing sprites and initializing background, floor, and columns.

    Args:
        sprites (pygame.sprite.LayeredUpdates): The sprite group.

    Returns:
        int: Timestamp of the last column spawn.
    """
    sprites.empty()
    floor_image_width = get_sprite("floor").get_width()
    num_floors = (SCREEN_WIDTH // floor_image_width) + 2
    for i in range(num_floors):
        Background(i, sprites)
        Floor(i, sprites)
    sprites.add(Column(sprites))
    last_column_time = pygame.time.get_ticks()
    return last_column_time

def train_genetic():
    pygame.init()
    display_info = pygame.display.Info()
    window_width, window_height = display_info.current_w, display_info.current_h
    screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
    pygame.display.set_caption("Flappy Bird Genetic Algorithm Training")
    clock = pygame.time.Clock()

    # Surfaces
    game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    info_surface_height = 200  # Adjust as needed
    info_surface = pygame.Surface((SCREEN_WIDTH, info_surface_height))
    x_pos = 50
    y_pos = (window_height - SCREEN_HEIGHT) // 2
    nn_surface = pygame.Surface((600, window_height))  # Neural network visualization surface
    nn_x_pos = x_pos + SCREEN_WIDTH
    nn_y_pos = 0

    # Ensure neural network visualization fits within the screen
    NN_WIDTH, NN_HEIGHT = 600, window_height
    if nn_x_pos + NN_WIDTH + 50 > window_width:
        raise ValueError("Neural Network visualization window exceeds screen width. Adjust NN_WIDTH or window size.")

    load_sprites()

    # Initialize Genetic Controller with a population size of 100
    genetic_controller = GeneticController(
        input_size=6,  # Number of inputs to the neural network
        hidden_sizes=[10, 10],  # Hidden layers sizes
        output_size=2,  # Two possible actions: flap or no flap
        population_size=100,  # Increased to 100
        mutation_rate=0.1
    )

    num_generations = 50
    best_genome = None
    best_fitness = float('-inf')

    # Define input and output labels
    input_labels = [
        "Bird Y Position",
        "Bird Velocity",
        "Pipe Distance X",
        "Pipe Gap Y",
        "Distance to Top Pipe",
        "Distance to Bottom Pipe"
    ]

    output_labels = [
        "Flap",
        "No Flap"
    ]

    # Create a unique subfolder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_identifier = f"{timestamp}"
    folder_name = f"pipegap_{PIPE_GAP_SIZE}_{unique_identifier}"
    save_dir = os.path.join('genetic_models', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")

    for generation in range(1, num_generations + 1):
        print(f"Generation {generation}")
        sprites = pygame.sprite.LayeredUpdates()
        last_column_time = reset_game(sprites)

        # Calculate max score for this generation
        max_score = 20 + (2 * generation)

        # Create birds for the entire population
        birds = []
        fitnesses = []
        scores = []
        for genome_idx, genome in enumerate(genetic_controller.population):
            bird = Bird(sprites, color=(0, 255, 0))  # Green color for genetic algorithm birds
            bird.genome = genome
            bird.genome_idx = genome_idx
            bird.score = 0
            bird.passed_columns = set()
            birds.append(bird)
            fitnesses.append(0)
            scores.append(0)

        running = True
        while running and len(birds) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            current_time = pygame.time.get_ticks()
            if current_time - last_column_time > 1500:
                sprites.add(Column(sprites))
                last_column_time = current_time

            # Get the game state
            game_state = get_game_state(sprites, birds[0])  # Assuming all birds have the same game state

            # Update each bird
            for bird in birds[:]:  # Copy the list to avoid modification during iteration
                genome = bird.genome
                genome_idx = bird.genome_idx
                inputs = np.array([
                    bird.rect.y / SCREEN_HEIGHT,
                    bird.flap / 10,
                    game_state['pipe_dist'] / SCREEN_WIDTH,
                    game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
                    game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
                    game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
                ])

                action = genetic_controller.decide_action(genome, inputs)
                bird_action = 'flap' if action == 0 else 'no_flap'
                if bird_action == 'flap':
                    bird.flap = -6

                fitnesses[genome_idx] += 0.1  # Small reward for staying alive

                # Check for passing pipes and update score
                for column in sprites.get_sprites_from_layer(Layer.OBSTACLE):
                    if isinstance(column, Column):
                        if (column.rect.x + column.rect.width < bird.rect.x) and (column not in bird.passed_columns):
                            bird.score += 1
                            scores[genome_idx] = bird.score
                            bird.passed_columns.add(column)
                            fitnesses[genome_idx] += 5  # Reward for passing a pipe
                            # Enforce max score
                            if bird.score >= max_score:
                                fitnesses[genome_idx] += 10  # Additional reward for reaching max score
                                birds.remove(bird)
                                bird.kill()
                            break

                # Update bird and check for collision
                if bird in birds:  # Check if bird hasn't been removed due to max score
                    if bird.check_collision(sprites):
                        fitnesses[genome_idx] -= 50  # Penalty for dying
                        birds.remove(bird)
                        bird.kill()

            # Update other sprites
            sprites.update()

            # Drawing
            game_surface.fill((0, 0, 0))
            sprites.draw(game_surface)

            # Only draw info and inputs for one bird
            if birds:
                bird = birds[0]
                draw_bird_inputs(game_surface, bird, game_state)
                draw_bird_info_text(info_surface, bird, game_state, fitnesses[bird.genome_idx], bird.score, generation, max_score)

                # Optionally, visualize the neural network of the best genome in the current generation
                current_best_fitness = max(fitnesses)
                current_best_idx = fitnesses.index(current_best_fitness)
                current_best_genome = genetic_controller.population[current_best_idx]
                nn_surface.fill((0, 0, 0))  # Clear previous NN visualization
                draw_neural_network(
                    nn_surface,
                    current_best_genome,
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
            if birds:
                screen.blit(nn_surface, (nn_x_pos, nn_y_pos))
            pygame.display.flip()
            clock.tick(FPS)

        # After all birds are done for this generation
        # Store the fitness scores
        genetic_controller.fitness_scores = fitnesses

        # Find the best genome in this generation
        max_fitness_in_gen = max(fitnesses)
        if max_fitness_in_gen > best_fitness:
            best_fitness = max_fitness_in_gen
            best_genome_idx = fitnesses.index(max_fitness_in_gen)
            best_genome = copy.deepcopy(genetic_controller.population[best_genome_idx])

        # Evolve the population
        genetic_controller.evolve()

        print(f"Best fitness in generation {generation}: {best_fitness}")

        # Save the best genome periodically in the unique subfolder
        if generation % 10 == 0:
            save_path = os.path.join(save_dir, f'best_genome_gen_{generation}.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"Best genome saved to '{save_path}'")

    # After all generations, save the final best genome in the unique subfolder
    final_save_path = os.path.join(save_dir, 'best_genetic_genome.pkl')
    with open(final_save_path, 'wb') as f:
        pickle.dump(best_genome, f)
    print("Training completed. Best genome saved.")

def main():
    train_genetic()

if __name__ == "__main__":
    main()
