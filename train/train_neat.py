import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import neat
import pickle
from ai.neat_controller import NEATController
from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.layer import Layer
from game.state import get_game_state

# Initialize Pygame
pygame.init()
display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird NEAT Training")
clock = pygame.time.Clock()

# Surfaces
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
info_surface = pygame.Surface((SCREEN_WIDTH, 200))
nn_surface = pygame.Surface((600, window_height))
x_pos = 50
y_pos = (window_height - SCREEN_HEIGHT) // 2
nn_x_pos = x_pos + SCREEN_WIDTH
nn_y_pos = 0

# Ensure the neural network visualization fits the screen
NN_WIDTH, NN_HEIGHT = 600, window_height
if nn_x_pos + NN_WIDTH + 50 > window_width:
    raise ValueError("Neural Network visualization window exceeds screen width. Adjust NN_WIDTH or window size.")

# Load game sprites
load_sprites()

def clamp(value, min_value=0, max_value=255):
    return max(min_value, min(int(value), max_value))

def draw_neural_network(screen, genome, config, position=(50, 50), scale=600):
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
    for conn_gene in genome.connections.values():
        if not conn_gene.enabled:
            continue
        from_pos = node_positions.get(conn_gene.key[0])
        to_pos = node_positions.get(conn_gene.key[1])
        if from_pos and to_pos:
            weight = conn_gene.weight
            color = (0, clamp(int(abs(weight) * 255)), 0) if weight > 0 else (clamp(int(abs(weight) * 255)), 0, 0)
            pygame.draw.line(screen, color, from_pos, to_pos, max(2, int(abs(weight) * 2)))

    node_radius = 10
    for layer_name, nodes in layers.items():
        for node_id in nodes:
            pos = node_positions[node_id]
            if layer_name == 'input':
                color = (255, 165, 0)
            elif layer_name == 'output':
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            pygame.draw.circle(screen, color, pos, node_radius)
            label = node_names.get(node_id, str(node_id))
            label_surface = font.render(label, True, (255, 255, 255))
            label_rect = label_surface.get_rect(center=(pos[0], pos[1] - node_radius - 10))
            screen.blit(label_surface, label_rect)

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

    if next_pipe_x < SCREEN_WIDTH:
        top_of_reward_zone = max(0, next_pipe_gap_y - reward_zone_margin)
        bottom_of_reward_zone = min(SCREEN_HEIGHT, next_pipe_gap_y + reward_zone_margin)
        reward_zone_surface = pygame.Surface((SCREEN_WIDTH, bottom_of_reward_zone - top_of_reward_zone), pygame.SRCALPHA)
        reward_zone_surface.fill((0, 255, 0, 50))
        screen.blit(reward_zone_surface, (0, top_of_reward_zone))

    pygame.draw.circle(screen, red, (int(bird.rect.x), int(bird_y)), dot_radius, 1)
    vel_display_y = max(0, min(int(bird_y + bird_vel * 10), SCREEN_HEIGHT))
    pygame.draw.circle(screen, red, (int(bird.rect.x), vel_display_y), dot_radius)
    pygame.draw.line(screen, red, (int(bird.rect.x), int(bird_y)), (int(bird.rect.x), vel_display_y), 1)

    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.circle(screen, yellow, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius, 1)
        pipe_gap_half = pipe_gap_size / 2
        top_pipe_y = next_pipe_gap_y - pipe_gap_half
        bottom_pipe_y = next_pipe_gap_y + pipe_gap_half
        pygame.draw.line(screen, red, (int(next_pipe_x), 0), (int(next_pipe_x), int(top_pipe_y)), 5)
        pygame.draw.line(screen, red, (int(next_pipe_x), SCREEN_HEIGHT), (int(next_pipe_x), int(bottom_pipe_y)), 5)
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
        font.render(f"Bird Y Position: {bird_y}", True, (255, 255, 255)),
        font.render(f"Bird Velocity: {bird_vel:.2f}", True, (255, 255, 255)),
        font.render(f"Next Pipe Distance X: {next_pipe_x}", True, (255, 255, 255)),
        font.render(f"Next Pipe Gap Y: {next_pipe_gap_y}", True, (255, 255, 255)),
        font.render(f"Next Pipe End X: {next_pipe_end_x}", True, (255, 255, 255))
    ]

    info_surface.fill((0, 0, 0))
    for idx, text in enumerate(texts):
        info_surface.blit(text, (10, 10 + idx * 30))

def reset_game(sprites, genomes, neat_controller):
    sprites.empty()
    for i in range((SCREEN_WIDTH // get_sprite("floor").get_width()) + 2):
        Background(i, sprites)
        Floor(i, sprites)
    sprites.add(Column(sprites))
    last_column_time = pygame.time.get_ticks()
    birds = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, neat_controller.config)
        bird = Bird(sprites, controller=net, color=(255, 0, 0))
        bird.genome = genome
        birds.append(bird)
    return birds, last_column_time

def main():
    neat_controller = NEATController(config_path='neat-config.txt')

    def eval_genomes(genomes, config):
        sprites = pygame.sprite.LayeredUpdates()
        birds, last_column_time = reset_game(sprites, genomes, neat_controller)
        running = True
        while running and birds:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s and birds:
                    save_path = os.path.join('neat_models', 'current_neat_genome.pkl')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    try:
                        with open(save_path, 'wb') as f:
                            pickle.dump(birds[0].genome, f)
                        print(f"Current NEAT genome saved to '{save_path}'")
                    except Exception as e:
                        print(f"Failed to save genome: {e}")

            current_time = pygame.time.get_ticks()
            if current_time - last_column_time > 1500:
                sprites.add(Column(sprites))
                last_column_time = current_time

            for bird in birds[:]:
                game_state = get_game_state(sprites, bird)
                inputs = [
                    game_state['bird_y'] / SCREEN_HEIGHT,
                    game_state['bird_vel'] / 10,
                    game_state['pipe_dist'] / SCREEN_WIDTH,
                    game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
                    game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
                    game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
                ]
                output = bird.controller.activate(inputs)
                action = 'flap' if output[0] > output[1] else 'no_flap'
                if action == 'flap':
                    bird.flap = -6
                bird.genome.fitness += 0.1

                for sprite in sprites.get_sprites_from_layer(Layer.OBSTACLE):
                    if isinstance(sprite, Column) and sprite.is_passed():
                        bird.genome.fitness += 10
                        break

                if bird.check_collision(sprites):
                    bird.genome.fitness -= 100
                    birds.remove(bird)
                    bird.kill()
                    continue

            sprites.update()
            game_surface.fill((0, 0, 0))
            sprites.draw(game_surface)

            if birds:
                nn_surface.fill((0, 0, 0))
                draw_neural_network(nn_surface, birds[0].genome, neat_controller.config, position=(0, 100))
                draw_bird_inputs(game_surface, birds[0], get_game_state(sprites, birds[0]))
                draw_bird_info_text(info_surface, birds[0], get_game_state(sprites, birds[0]))

            screen.fill((0, 0, 0))
            screen.blit(info_surface, (x_pos, 0))
            screen.blit(game_surface, (x_pos, y_pos))
            screen.blit(nn_surface, (nn_x_pos, nn_y_pos))
            pygame.display.flip()
            clock.tick(FPS)

    winner = neat_controller.run(eval_genomes, n_generations=50)
    os.makedirs('neat_models', exist_ok=True)
    with open(os.path.join('neat_models', 'best_neat_genome.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    print("Training completed. Best genome saved as 'neat_models/best_neat_genome.pkl'.")

if __name__ == "__main__":
    main()
