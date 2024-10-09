import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np

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

# Load sprites
load_sprites()

def clamp(value, min_value=0, max_value=255):
    return max(min_value, min(int(value), max_value))

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

def train_dql():
    dqn_controller = DQNController()
    dqn_models_dir = 'dqn_models'
    os.makedirs(dqn_models_dir, exist_ok=True)
    NUM_EPISODES = 1000

    for episode in range(1, NUM_EPISODES + 1):
        sprites = pygame.sprite.LayeredUpdates()
        bird, last_column_time, score = reset_game(sprites, dqn_controller)
        total_reward = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    model_path = os.path.join(dqn_models_dir, 'current_dql_model.pth')
                    dqn_controller.save_model(model_path)
                    print(f"DQL model saved to '{model_path}'")

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
            bird.flap = -6 if action == 1 else bird.flap

            sprites.update(game_state)
            next_game_state = get_game_state(sprites, bird)
            next_state = np.array([
                next_game_state['bird_y'] / SCREEN_HEIGHT,
                next_game_state['bird_vel'] / 10,
                next_game_state['pipe_dist'] / SCREEN_WIDTH,
                next_game_state['next_pipe_gap_y'] / SCREEN_HEIGHT,
                next_game_state['dist_to_top_pipe'] / SCREEN_HEIGHT,
                next_game_state['dist_to_bottom_pipe'] / SCREEN_HEIGHT
            ])

            done = bird.check_collision(sprites)
            if done:
                reward = -40
                dqn_controller.store_transition(state, action, reward, next_state, done)
                dqn_controller.optimize_model()
                running = False
            else:
                reward = -0.25
                for sprite in sprites.get_sprites_from_layer(Layer.OBSTACLE):
                    if isinstance(sprite, Column) and sprite.is_passed():
                        reward += 5
                        score.value += 1
                        break
                y_diff = abs(bird.rect.y - next_game_state['next_pipe_gap_y'])
                if y_diff <= 30:
                    reward += 20
                elif y_diff > 50:
                    punishment = (y_diff - 30) * 0.02
                    reward -= punishment
                dqn_controller.store_transition(state, action, reward, next_state, done)
                dqn_controller.optimize_model()

            total_reward += reward
            game_surface.fill((0, 0, 0))
            sprites.draw(game_surface)
            draw_bird_inputs(game_surface, bird, game_state)
            draw_bird_info_text(info_surface, bird, game_state)

            screen.fill((0, 0, 0))
            screen.blit(info_surface, (x_pos, 0))
            screen.blit(game_surface, (x_pos, y_pos))
            pygame.display.flip()
            clock.tick(FPS)

        if episode % 50 == 0:
            model_path = os.path.join(dqn_models_dir, f'dqn_model_episode_{episode}.pth')
            dqn_controller.save_model(model_path)
            print(f"DQL model saved to '{model_path}'")

    final_model_path = os.path.join(dqn_models_dir, 'dqn_model_final.pth')
    dqn_controller.save_model(final_model_path)
    print("Training completed.")

def main():
    train_dql()

if __name__ == "__main__":
    main()
