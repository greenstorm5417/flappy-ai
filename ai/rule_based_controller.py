class RuleBasedController:
    def __init__(self, rules=None):
        """
        Initialize the Rule-Based Controller with customizable rules.

        Args:
            rules (dict): A dictionary containing the rules for decision-making.
        """
        # Default rules if none are provided
        if rules is None:
            self.rules = {
                'flap_if_below_gap': True,
                'flap_if_descending': False,
                'flap_if_close_to_pipe': False,
                'distance_threshold': 50,  # Pixels
            }
        else:
            self.rules = rules

    def decide_action(self, bird, game_state):
        """
        Decide whether to flap or not based on the current game state and rules.

        Args:
            bird: The Bird object.
            game_state: The game state dictionary.

        Returns:
            'flap' or 'no_flap'
        """
        if not game_state:
            return 'no_flap'

        # Extract necessary information from the game state
        bird_y = bird.rect.y
        bird_vel = bird.flap
        next_pipe_x = game_state['next_pipe_x']
        next_pipe_gap_y = game_state['next_pipe_gap_y']
        pipe_dist = next_pipe_x - bird.rect.x

        action = 'no_flap'

        # Apply rules
        if self.rules.get('flap_if_below_gap', False):
            if bird_y > next_pipe_gap_y:
                action = 'flap'

        if self.rules.get('flap_if_descending', False):
            if bird_vel > 0:  # Bird is descending
                action = 'flap'

        if self.rules.get('flap_if_close_to_pipe', False):
            distance_threshold = self.rules.get('distance_threshold', 50)
            if 0 < pipe_dist < distance_threshold:
                action = 'flap'

        return action

    def reset(self):
        pass  # No action needed for reset in this implementation
