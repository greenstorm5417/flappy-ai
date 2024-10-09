# game/state.py

from game.layer import Layer
from game.column import Column
from game.assets import get_sprite
from game.configs import SCREEN_WIDTH, SCREEN_HEIGHT

def get_game_state(sprites, bird):
    """
    Extracts and returns the current game state in a standardized format.

    Args:
        sprites (pygame.sprite.Group): The group containing all sprites.
        bird (Bird): The Bird object controlled by AI.

    Returns:
        dict: A dictionary containing the game state.
    """
    # Find all pipes that are ahead of the bird
    pipes = [
        sprite for sprite in sprites.get_sprites_from_layer(Layer.OBSTACLE)
        if sprite.rect.right > bird.rect.left
    ]
    
    if pipes:
        next_pipe = min(pipes, key=lambda pipe: pipe.rect.x)
        pipe_dist = next_pipe.rect.x - bird.rect.x
        pipe_gap_y = next_pipe.get_gap_center_y()
        pipe_gap_size = next_pipe.gap  # Assuming 'gap' is an attribute representing the size

        # Calculate distances to top and bottom pipes
        dist_to_top_pipe = bird.rect.top - (pipe_gap_y - pipe_gap_size / 2)
        dist_to_bottom_pipe = (pipe_gap_y + pipe_gap_size / 2) - bird.rect.bottom

        return {
            'screen_width': SCREEN_WIDTH,
            'screen_height': SCREEN_HEIGHT,
            'next_pipe_x': next_pipe.rect.x,
            'next_pipe_gap_y': pipe_gap_y,
            'pipe_gap_size': pipe_gap_size,
            'bird_y': bird.rect.y,
            'bird_vel': bird.flap,
            'pipe_dist': pipe_dist,
            'dist_to_top_pipe': dist_to_top_pipe,
            'dist_to_bottom_pipe': dist_to_bottom_pipe
        }
    else:
        default_gap_size = 150
        return {
            'screen_width': SCREEN_WIDTH,
            'screen_height': SCREEN_HEIGHT,
            'next_pipe_x': SCREEN_WIDTH,
            'next_pipe_gap_y': SCREEN_HEIGHT / 2,
            'pipe_gap_size': default_gap_size,
            'bird_y': bird.rect.y,
            'bird_vel': bird.flap,
            'pipe_dist': SCREEN_WIDTH - bird.rect.x,
            'dist_to_top_pipe': -bird.rect.top,
            'dist_to_bottom_pipe': SCREEN_HEIGHT - bird.rect.bottom
        }