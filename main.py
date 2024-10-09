# main.py

import pygame
import sys
import os
import torch
import numpy as np
import neat

from ai.dql_controller import DQNController
from ai.neat_controller import NEATController
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


def create_button_text(text):
    return BUTTON_FONT.render(text, True, BUTTON_TEXT_COLOR)


def init_controllers():
    rule_based_controller = RuleBasedController()
    dqn_models_dir = 'dqn_models'
    os.makedirs(dqn_models_dir, exist_ok=True)
    dqn_controller = DQNController(is_training=False)
    model_path = os.path.join(dqn_models_dir, 'current_dql_model.pth')
    if not os.path.exists(model_path):
        print(f"Trained DQL model not found at '{model_path}'. Please train the model first.")
        sys.exit(1)
    dqn_controller.load_model(model_path)
    neat_controller = NEATController(config_path='neat-config.txt', genome_path='neat_models/current_neat_genome.pkl')
    if neat_controller.genome is None:
        print("No genome loaded. Please ensure 'ai/current_neat_genome.pkl' exists and is a valid NEAT genome.")
        sys.exit(1)
    neat_net = neat.nn.FeedForwardNetwork.create(neat_controller.genome, neat_controller.config)
    return rule_based_controller, dqn_controller, neat_net


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
    rule_based_controller, dqn_controller, neat_net = controllers
    birds.append(create_bird(Bird(sprites, color=(255, 255, 0)), rule_based_controller, 'Rule-Based', (255, 255, 0)))
    birds.append(create_bird(Bird(sprites, color=(0, 0, 255)), dqn_controller, 'DQL', (0, 0, 255)))
    birds.append(create_bird(Bird(sprites, color=(255, 0, 0)), neat_net, 'NEAT', (255, 0, 0)))
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
    vel_display_y = max(0, min(int(vel_display_y), SCREEN_HEIGHT))
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
    elif isinstance(controller, neat.nn.FeedForwardNetwork):
        inputs = [
            game_state['bird_y'] / SCREEN_HEIGHT,
            game_state['bird_vel'] / 10,
            game_state['pipe_dist'] / SCREEN_WIDTH,
            game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
            game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
            game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
        ]
        output = controller.activate(inputs)
        bird_action = 'flap' if output[0] > output[1] else 'no_flap'
    elif isinstance(controller, RuleBasedController):
        bird_action = controller.decide_action(bird, game_state)
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


def main():
    controllers = init_controllers()
    sprites = init_sprites()
    birds = create_birds(sprites, controllers)
    info_surface = pygame.Surface((SCREEN_WIDTH, INFO_AREA_HEIGHT * len(birds)))
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
