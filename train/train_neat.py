import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
import pygame
import numpy as np
import neat
import pickle
import datetime  # For timestamp-based unique identifiers
import copy

from ai.neat_controller import NEATController
from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, PIPE_GAP_SIZE
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.layer import Layer
from game.state import get_game_state

# Initialize Pygame
pygame.init()

# Display and window setup
display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird NEAT Training")
clock = pygame.time.Clock()

# Create surfaces
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
info_surface = pygame.Surface((SCREEN_WIDTH, 200))
nn_surface = pygame.Surface((600, window_height))  # Neural network visualization
x_pos = 50
y_pos = (window_height - SCREEN_HEIGHT) // 2
nn_x_pos = x_pos + SCREEN_WIDTH
nn_y_pos = 0

# Ensure neural network visualization fits within the screen
NN_WIDTH, NN_HEIGHT = 600, window_height
if nn_x_pos + NN_WIDTH + 50 > window_width:
    raise ValueError("Neural Network visualization window exceeds screen width. Adjust NN_WIDTH or window size.")

# Load game assets
load_sprites()

def clamp(value, min_value=0, max_value=255):
    """Clamp the value between min_value and max_value."""
    return max(min_value, min(int(value), max_value))

def draw_neural_network(screen, genome, config, position=(50, 50), scale=600):
    """
    Draws a neural network based on a NEAT genome.
    
    Args:
        screen (pygame.Surface): The surface to draw on.
        genome (neat.Genome): The NEAT genome.
        config (neat.Config): The NEAT configuration.
        position (tuple): Top-left position to start drawing.
        scale (int): Scaling factor for the visualization.
    """
    node_names = {
        -1: 'Bird Y',
        -2: 'Bird Velocity',
        -3: 'Pipe Distance X',
        -4: 'Pipe Gap Y',
        -5: 'Dist to Top Pipe',
        -6: 'Dist to Bottom Pipe',
        0: 'Flap',
        1: 'No Flap'
    }

    # Separate nodes by type
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]

    layers = {
        "input": input_nodes,
        "hidden": hidden_nodes,
        "output": output_nodes
    }

    total_layers = len([layer for layer in layers.values() if layer])
    if total_layers == 0:
        return

    layer_spacing = scale // (total_layers + 1)
    node_positions = {}
    current_layer = 0

    for layer_name, nodes in layers.items():
        if not nodes:
            continue
        current_layer += 1
        x = position[0] + layer_spacing * current_layer
        num_nodes = len(nodes)
        vertical_spacing = scale // (num_nodes + 1) if num_nodes > 0 else scale
        for i, node_id in enumerate(nodes):
            y = position[1] + vertical_spacing * (i + 1)
            node_positions[node_id] = (x, y)

    font = pygame.font.SysFont(None, 18)

    # Draw connections
    for conn_gene in genome.connections.values():
        if not conn_gene.enabled:
            continue
        from_pos = node_positions.get(conn_gene.key[0])
        to_pos = node_positions.get(conn_gene.key[1])
        if from_pos and to_pos:
            weight = conn_gene.weight
            # Determine color based on weight sign and magnitude
            if weight > 0:
                color = (0, clamp(int(abs(weight) * 255)), 0)  # Greenish for positive weights
            else:
                color = (clamp(int(abs(weight) * 255)), 0, 0)  # Reddish for negative weights
            # Line thickness based on weight magnitude
            thickness = max(1, int(abs(weight) * 2))
            pygame.draw.line(screen, color, from_pos, to_pos, thickness)

    # Draw nodes with labels
    node_radius = 10
    for layer_name, nodes in layers.items():
        for node_id in nodes:
            pos = node_positions[node_id]
            if layer_name == 'input':
                color = (255, 165, 0)  # Orange
            elif layer_name == 'output':
                color = (0, 0, 255)    # Blue
            else:
                color = (255, 255, 255)  # White
            pygame.draw.circle(screen, color, pos, node_radius)
            label = node_names.get(node_id, str(node_id))
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
    for layer_name, nodes in layers.items():
        if nodes:
            first_pos = node_positions[nodes[0]]
            title = layer_titles[layer_name]
            title_surface = title_font.render(title, True, (255, 255, 255))
            if layer_name == "input":
                x_pos_title = first_pos[0] - node_radius
            elif layer_name == "output":
                x_pos_title = first_pos[0] - title_surface.get_width() + node_radius
            else:
                x_pos_title = first_pos[0] - title_surface.get_width() // 2
            screen.blit(title_surface, (x_pos_title, position[1] - 30))

def draw_bird_inputs(screen, bird, game_state):
    """
    Draws visual representations of the bird's inputs, including a green strip indicating the reward zone near the pipe gap.
    
    Args:
        screen (pygame.Surface): The surface to draw on.
        bird (Bird): The Bird object.
        game_state (dict): The current game state.
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

def reset_game(sprites, genomes, neat_controller):
    """
    Resets the game by clearing sprites and initializing background, floor, columns, and birds.

    Args:
        sprites (pygame.sprite.LayeredUpdates): The sprite group.
        genomes (list): List of genomes in the current generation.
        neat_controller (NEATController): The NEATController instance.

    Returns:
        list: List of Bird objects.
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
    # Initialize birds
    birds = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # Reset fitness
        net = neat.nn.FeedForwardNetwork.create(genome, neat_controller.config)
        bird = Bird(sprites, controller=net, color=(255, 0, 0))  # Red bird for NEAT
        bird.genome = genome
        bird.score = 0
        bird.passed_columns = set()
        birds.append(bird)
    return birds, last_column_time

def main():
    # Initialize NEATController with the configuration
    neat_controller = NEATController(config_path='neat-config.txt')

    # Create a unique subfolder for this run based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_identifier = f"{timestamp}"
    folder_name = f"neat_{unique_identifier}"
    save_dir = os.path.join('neat_models', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Genomes will be saved to: {save_dir}")

    # Initialize generation count
    generation = [1]  # Using a list to make it mutable within eval_genomes

    def eval_genomes(genomes, config):
        """
        Evaluates the fitness of each genome.
        
        Args:
            genomes (list): List of tuples (genome_id, genome).
            config (neat.Config): NEAT configuration.
        """
        current_generation = generation[0]
        max_score = 20 + (current_generation * 2)
        print(f"Evaluating Generation {current_generation} with max score {max_score}")

        # Reset the game with current genomes
        sprites = pygame.sprite.LayeredUpdates()
        birds, last_column_time = reset_game(sprites, genomes, neat_controller)

        running = True
        while running and len(birds) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Spawn new columns based on time
            current_time = pygame.time.get_ticks()
            if current_time - last_column_time > 1500:
                sprites.add(Column(sprites))
                last_column_time = current_time

            if not birds:
                break  # No birds left to evaluate

            # Assuming all birds share the same game state
            game_state = get_game_state(sprites, birds[0])

            # Update each bird
            for bird in birds[:]:  # Copy the list to avoid modification during iteration
                genome = bird.genome
                inputs = np.array([
                    game_state['bird_y'] / SCREEN_HEIGHT,
                    game_state['bird_vel'] / 10,
                    game_state['pipe_dist'] / SCREEN_WIDTH,
                    game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
                    game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
                    game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
                ])
                output = bird.controller.activate(inputs)
                action = 'flap' if output[0] > output[1] else 'no_flap'
                if action == 'flap':
                    bird.flap = -6

                # Assign fitness
                genome.fitness += 0.1  # Small reward for staying alive

                # Check for passing columns and update score
                for sprite in sprites.get_sprites_from_layer(Layer.OBSTACLE):
                    if isinstance(sprite, Column):
                        if (sprite.rect.x + sprite.rect.width < bird.rect.x) and (sprite not in bird.passed_columns):
                            bird.score += 1
                            genome.fitness += 5  # Reward for passing a column
                            bird.passed_columns.add(sprite)
                            # Check if bird has reached max score
                            if bird.score >= max_score:
                                genome.fitness += 10  # Additional reward for reaching max score
                                birds.remove(bird)
                                bird.kill()
                            break

                # Check for collision
                if bird.check_collision(sprites):
                    genome.fitness -= 100  # Penalty for dying
                    birds.remove(bird)
                    bird.kill()

            # Update all sprites
            sprites.update()

            # Draw everything
            game_surface.fill((0, 0, 0))
            sprites.draw(game_surface)

            if birds:
                bird = birds[0]
                draw_bird_inputs(game_surface, bird, game_state)
                draw_bird_info_text(
                    info_surface, 
                    bird, 
                    game_state, 
                    fitness=bird.genome.fitness, 
                    score=bird.score, 
                    generation=current_generation, 
                    max_score=max_score
                )
                
                # Clear the nn_surface before drawing the neural network
                nn_surface.fill((0, 0, 0))  # Black background

                # Draw neural network visualization for the best genome
                best_genome = max(genomes, key=lambda g: g[1].fitness)[1]
                draw_neural_network(
                    nn_surface, 
                    best_genome, 
                    config, 
                    position=(0, 100), 
                    scale=600
                )

            # Render everything on the main screen
            screen.fill((0, 0, 0))
            screen.blit(info_surface, (x_pos, 0))
            screen.blit(game_surface, (x_pos, y_pos))
            if birds:
                screen.blit(nn_surface, (nn_x_pos, nn_y_pos))
            pygame.display.flip()

            # Control the frame rate
            clock.tick(FPS)

        # After evaluation, increment generation
        generation[0] += 1

        # Save the best genome of the current generation
        if generation % 10 == 0:
            best_genome = max(genomes, key=lambda g: g[1].fitness)[1]
            save_path = os.path.join(save_dir, f'best_genome_gen_{current_generation}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(best_genome, f)
            print(f"Best genome of generation {current_generation} saved to '{save_path}'")

    # Run NEAT
    winner = neat_controller.run(eval_genomes, n_generations=200)

    # Save the final best genome
    final_save_path = os.path.join(save_dir, 'best_neat_genome.pkl')
    with open(final_save_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Training completed. Best genome saved as '{final_save_path}'.")

if __name__ == "__main__":
    main()
