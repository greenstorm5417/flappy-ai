# main.py

import pygame
import sys
import os
import torch
import torch.nn as nn  # For accessing the DQN model's layers
import numpy as np
import neat
import pickle  # Import pickle to load the best genome

from ai.dql_controller import DQNController
from ai.neat_controller import NEATController
from ai.rule_based_controller import RuleBasedController
from ai.genetic_controller import GeneticController

from game.assets import load_sprites, get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from game.background import Background
from game.floor import Floor
from game.column import Column
from game.bird import Bird
from game.state import get_game_state
from game.layer import Layer

pygame.init()

BUTTON_WIDTH = 120
BUTTON_HEIGHT = 30
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
BUTTON_TEXT_COLOR = (255, 255, 255)
BUTTON_FONT = pygame.font.SysFont(None, 24)
INFO_AREA_HEIGHT = 150

display_info = pygame.display.Info()
window_width, window_height = display_info.current_w, display_info.current_h
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("Flappy Bird Multi-Agent Test")
clock = pygame.time.Clock()

game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
x_pos = 50
y_pos = (window_height - SCREEN_HEIGHT) // 2

load_sprites()


def clamp(value, min_value=0, max_value=255):
    """Clamp the value between min_value and max_value."""
    return max(min_value, min(int(value), max_value))


def create_button_text(text):
    return BUTTON_FONT.render(text, True, BUTTON_TEXT_COLOR)


def init_controllers():
    controllers = {}

    # Rule-Based Controller
    controllers['Rule-Based'] = RuleBasedController()

    # DQN Controller
    dqn_models_dir = 'dqn_models'
    os.makedirs(dqn_models_dir, exist_ok=True)
    dqn_controller = DQNController(is_training=False)
    model_path = os.path.join(dqn_models_dir, 'current_dql_model.pth')
    if not os.path.exists(model_path):
        print(f"Trained DQL model not found at '{model_path}'. Please train the model first.")
        sys.exit(1)
    dqn_controller.load_model(model_path)
    controllers['DQL'] = dqn_controller

    # NEAT Controller
    neat_controller = NEATController(config_path='neat-config.txt', genome_path='neat_models/best_neat_genome.pkl')
    if neat_controller.genome is None:
        print("No genome loaded. Please ensure 'neat_models/best_neat_genome.pkl' exists and is a valid NEAT genome.")
        sys.exit(1)
    controllers['NEAT'] = neat_controller

    # Genetic Controller
    genetic_controller = GeneticController()
    best_genome_path = 'genetic_models\\pipegap_150_20241009_222924\\best_genome_gen_50.pkl'  # Update the path if needed
    if not os.path.exists(best_genome_path):
        print(f"Best genome not found at '{best_genome_path}'. Please provide the correct path.")
        sys.exit(1)
    with open(best_genome_path, 'rb') as f:
        best_genome = pickle.load(f)
    controllers['Genetic'] = (genetic_controller, best_genome)

    return controllers


def init_sprites():
    sprites = pygame.sprite.LayeredUpdates()
    floor_image_width = get_sprite("floor").get_width()
    num_floors = (SCREEN_WIDTH // floor_image_width) + 2
    for i in range(num_floors):
        Background(i, sprites)
        Floor(i, sprites)
    sprites.add(Column(sprites))
    return sprites


def create_bird(bird_instance, controller, name, color):
    return {
        'bird': bird_instance,
        'controller': controller,
        'name': name,
        'color': color,
        'score': 0,
        'passed_columns': set(),
        'inputs_visible': True
    }


def create_birds(sprites, controllers):
    birds = []
    # Rule-Based Bird
    birds.append(create_bird(Bird(sprites, color=(255, 255, 0)), controllers['Rule-Based'], 'Rule-Based', (255, 255, 0)))

    # DQL Bird
    birds.append(create_bird(Bird(sprites, color=(0, 0, 255)), controllers['DQL'], 'DQL', (0, 0, 255)))

    # NEAT Bird
    birds.append(create_bird(Bird(sprites, color=(255, 0, 0)), controllers['NEAT'], 'NEAT', (255, 0, 0)))

    # Genetic Bird
    birds.append(create_bird(Bird(sprites, color=(0, 255, 0)), controllers['Genetic'], 'Genetic', (0, 255, 0)))

    return birds


def draw_bird_info_text(info_surface, bird_dict, game_state, name='', color=(255, 255, 255), x_offset=0, bird_idx=None, button_rects=None):
    if not game_state:
        return
    bird_y = bird_dict['bird'].rect.y
    bird_vel = bird_dict['bird'].flap
    score = bird_dict['score']
    font = pygame.font.SysFont(None, 24)
    texts = [
        font.render(f"{name} Bird Info:", True, color),
        font.render(f"Bird Y Position: {bird_y}", True, color),
        font.render(f"Bird Velocity: {bird_vel:.2f}", True, color),
        font.render(f"Score: {score}", True, color)
    ]
    y = 10
    for text in texts:
        info_surface.blit(text, (10 + x_offset, y))
        y += 30
    button_x = 10 + x_offset
    button_y = y + 20
    absolute_button_rect = pygame.Rect(button_x + x_pos, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    if button_rects is not None and bird_idx is not None:
        button_rects.append((absolute_button_rect, bird_idx))


def draw_bird_inputs(screen, bird, game_state, color=(255, 255, 255)):
    if not game_state:
        return
    bird_y = bird.rect.y
    bird_vel = bird.flap
    next_pipe_x = game_state['next_pipe_x']
    next_pipe_gap_y = game_state['next_pipe_gap_y']
    pipe_gap_size = game_state['pipe_gap_size']
    dot_radius = 5
    pygame.draw.circle(screen, color, (int(bird.rect.x), int(bird_y)), dot_radius)
    vel_display_y = int(bird_y + bird_vel * 10)
    vel_display_y = max(0, min(vel_display_y, SCREEN_HEIGHT))
    pygame.draw.circle(screen, color, (int(bird.rect.x), vel_display_y), dot_radius)
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.circle(screen, color, (int(next_pipe_x), int(next_pipe_gap_y)), dot_radius)
        pipe_gap_half = pipe_gap_size / 2
        top_pipe_y = next_pipe_gap_y - pipe_gap_half
        bottom_pipe_y = next_pipe_gap_y + pipe_gap_half
        pygame.draw.line(screen, color, (int(next_pipe_x), 0), (int(next_pipe_x), int(top_pipe_y)), 5)
        pygame.draw.line(screen, color, (int(next_pipe_x), SCREEN_HEIGHT), (int(next_pipe_x), int(bottom_pipe_y)), 5)
    pygame.draw.line(screen, color, (int(bird.rect.x), int(bird_y)), (int(bird.rect.x), vel_display_y), 1)
    if next_pipe_x < SCREEN_WIDTH:
        pygame.draw.line(screen, color, (int(bird.rect.x), int(bird.rect.y)), (int(next_pipe_x), int(next_pipe_gap_y)), 1)


def handle_button_clicks(mouse_pos, buttons, birds):
    for button_rect, bird_idx in buttons:
        if button_rect.collidepoint(mouse_pos):
            birds[bird_idx]['inputs_visible'] = not birds[bird_idx]['inputs_visible']


def process_bird(player, sprites, columns):
    bird = player['bird']
    controller = player['controller']
    game_state = get_game_state(sprites, bird)
    if isinstance(controller, DQNController):
        state = np.array([
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ])
        action = controller.select_action(state)
        bird_action = 'flap' if action == 1 else 'no_flap'
    elif isinstance(controller, NEATController):
        net = controller.net
        inputs = [
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ]
        output = net.activate(inputs)
        bird_action = 'flap' if output[0] > output[1] else 'no_flap'
    elif isinstance(controller, RuleBasedController):
        bird_action = controller.decide_action(bird, game_state)
    elif isinstance(controller, tuple) and isinstance(controller[0], GeneticController):
        genetic_controller, genome = controller
        inputs = np.array([
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ])
        action = genetic_controller.decide_action(genome, inputs)
        bird_action = 'flap' if action == 0 else 'no_flap'  # Assuming 0: Flap, 1: No Flap
    else:
        bird_action = 'no_flap'
    if bird_action == 'flap':
        bird.flap = -6
    for column in columns:
        if (column.rect.x + column.rect.width < bird.rect.x) and (column not in player['passed_columns']):
            player['score'] += 1
            player['passed_columns'].add(column)
            break
    done = bird.check_collision(sprites)
    if done:
        bird.reset()
        player['score'] = 0
        player['passed_columns'] = set()


def draw_dqn_neural_network(screen, model, input_labels, output_labels, position=(50, 50), scale=600, weight_threshold=0.5):
    """
    Draws the neural network architecture and weights from a PyTorch model.
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
        label_surface = font.render(label, True, (255, 255, 255))
        label_rect = label_surface.get_rect(center=(pos[0], pos[1] - node_radius - 10))
        screen.blit(label_surface, label_rect)


def draw_neat_neural_network(screen, genome, config, position=(50, 50), scale=600):
    """
    Draws a neural network based on a NEAT genome.
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
            color = (0, clamp(int(abs(weight) * 255)), 0) if weight > 0 else (clamp(int(abs(weight) * 255)), 0, 0)
            pygame.draw.line(screen, color, from_pos, to_pos, max(2, int(abs(weight) * 2)))

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
            x_pos_title = first_pos[0] - title_surface.get_width() // 2
            screen.blit(title_surface, (x_pos_title, position[1]+20))


def draw_genetic_neural_network(screen, genome, input_size, hidden_sizes, output_size, input_labels, output_labels, position=(50, 50), scale=600):
    """
    Draws a neural network based on a GeneticController genome with labeled inputs and outputs.
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


def main():
    controllers = init_controllers()
    sprites = init_sprites()
    birds = create_birds(sprites, controllers)
    info_surface = pygame.Surface((window_width, INFO_AREA_HEIGHT))

    # Initialize Neural Network Surface
    NN_WIDTH = 600
    NN_HEIGHT = SCREEN_HEIGHT
    nn_surface = pygame.Surface((NN_WIDTH, NN_HEIGHT))
    nn_x_pos = x_pos + SCREEN_WIDTH + 50
    nn_y_pos = y_pos
    if nn_x_pos + NN_WIDTH > window_width:
        nn_x_pos = window_width - NN_WIDTH - 50  # Adjust if it exceeds window width

    # List of network names and selection index
    network_names = [player['name'] for player in birds]
    selected_network_index = 0

    last_column_time = pygame.time.get_ticks()
    running = True
    while running:
        buttons = []
        current_time = pygame.time.get_ticks()
        if current_time - last_column_time > 1500:
            sprites.add(Column(sprites))
            last_column_time = current_time
        columns = sorted(sprites.get_sprites_from_layer(Layer.OBSTACLE), key=lambda c: c.rect.x)
        for player in birds:
            process_bird(player, sprites, columns)
        sprites.update()
        game_surface.fill((0, 0, 0))
        sprites.draw(game_surface)
        info_surface.fill((0, 0, 0))
        for idx, player in enumerate(birds):
            bird = player['bird']
            name = player['name']
            color = player['color']
            game_state = get_game_state(sprites, bird)
            x_offset = idx * 250
            draw_bird_info_text(
                info_surface,
                player,
                game_state,
                name=name,
                color=color,
                x_offset=x_offset,
                bird_idx=idx,
                button_rects=buttons
            )
            if player['inputs_visible']:
                draw_bird_inputs(game_surface, bird, game_state, color=color)
        screen.fill((0, 0, 0))
        screen.blit(info_surface, (x_pos, 0))
        screen.blit(game_surface, (x_pos, y_pos))

        # Prepare Neural Network Surface
        nn_surface.fill((0, 0, 0))
        # Draw title and arrows
        title_font = pygame.font.SysFont(None, 36)
        selected_network_name = network_names[selected_network_index]
        title_text = title_font.render(f"{selected_network_name}", True, (255, 255, 255))
        title_text_rect = title_text.get_rect(center=(NN_WIDTH // 2, 20))
        nn_surface.blit(title_text, title_text_rect)

        # Define arrow dimensions
        arrow_width = 30
        arrow_height = 30

        left_arrow_x = title_text_rect.left - arrow_width - 10
        left_arrow_y = title_text_rect.centery - arrow_height // 2
        right_arrow_x = title_text_rect.right + 10
        right_arrow_y = title_text_rect.centery - arrow_height // 2

        left_arrow_rect = pygame.Rect(left_arrow_x, left_arrow_y, arrow_width, arrow_height)
        right_arrow_rect = pygame.Rect(right_arrow_x, right_arrow_y, arrow_width, arrow_height)

        arrow_buttons = [(left_arrow_rect, 'left'), (right_arrow_rect, 'right')]

        # Get relative mouse position
        mouse_pos = pygame.mouse.get_pos()
        relative_mouse_pos = (mouse_pos[0] - nn_x_pos, mouse_pos[1] - nn_y_pos)

        is_left_hover = left_arrow_rect.collidepoint(relative_mouse_pos)
        is_right_hover = right_arrow_rect.collidepoint(relative_mouse_pos)

        pygame.draw.rect(nn_surface, BUTTON_HOVER_COLOR if is_left_hover else BUTTON_COLOR, left_arrow_rect)
        pygame.draw.rect(nn_surface, BUTTON_HOVER_COLOR if is_right_hover else BUTTON_COLOR, right_arrow_rect)

        # Draw '<' and '>' on the arrows
        arrow_font = pygame.font.SysFont(None, 36)
        left_arrow_text = arrow_font.render('<', True, (255, 255, 255))
        right_arrow_text = arrow_font.render('>', True, (255, 255, 255))
        left_arrow_text_rect = left_arrow_text.get_rect(center=left_arrow_rect.center)
        right_arrow_text_rect = right_arrow_text.get_rect(center=right_arrow_rect.center)
        nn_surface.blit(left_arrow_text, left_arrow_text_rect)
        nn_surface.blit(right_arrow_text, right_arrow_text_rect)

        # Draw the neural network for the selected controller
        if selected_network_name == 'Rule-Based':
            # Leave it blank
            pass
        elif selected_network_name == 'DQL':
            # Get the DQL model
            dqn_controller = controllers['DQL']
            model = dqn_controller.policy_net
            # Define input and output labels
            input_labels = [
                "Bird Y Position",
                "Bird Velocity",
                "Pipe Distance X",
                "Pipe Gap Y",
                "Dist to Top Pipe",
                "Dist to Bottom Pipe"
            ]
            output_labels = ["No Flap", "Flap"]  # Assuming output 0: No Flap, 1: Flap
            draw_dqn_neural_network(nn_surface, model, input_labels, output_labels)
        elif selected_network_name == 'NEAT':
            # Get the genome and config
            neat_controller = controllers['NEAT']
            genome = neat_controller.genome
            config = neat_controller.config
            draw_neat_neural_network(nn_surface, genome, config)
        elif selected_network_name == 'Genetic':
            # Get the genome and controller
            genetic_controller, best_genome = controllers['Genetic']
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
            draw_genetic_neural_network(
                nn_surface,
                best_genome,
                input_size=genetic_controller.input_size,
                hidden_sizes=genetic_controller.hidden_sizes,
                output_size=genetic_controller.output_size,
                input_labels=input_labels,
                output_labels=output_labels
            )

        # Blit the nn_surface onto the main screen
        screen.blit(nn_surface, (nn_x_pos, nn_y_pos))

        # Handle buttons
        mouse_pos = pygame.mouse.get_pos()
        for button_rect, bird_idx in buttons:
            is_hover = button_rect.collidepoint(mouse_pos)
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR, button_rect)
            text_surf = create_button_text("Toggle Vis")
            text_rect = text_surf.get_rect(center=button_rect.center)
            screen.blit(text_surf, text_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                handle_button_clicks(mouse_pos, buttons, birds)
                # Check for arrow clicks
                relative_mouse_pos = (mouse_pos[0] - nn_x_pos, mouse_pos[1] - nn_y_pos)
                for arrow_rect, direction in arrow_buttons:
                    if arrow_rect.collidepoint(relative_mouse_pos):
                        if direction == 'left':
                            selected_network_index = (selected_network_index - 1) % len(network_names)
                        elif direction == 'right':
                            selected_network_index = (selected_network_index + 1) % len(network_names)
                        break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
