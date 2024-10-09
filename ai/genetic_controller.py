import random
import numpy as np

class GeneticController:
    def __init__(self, input_size=4, output_size=1, population_size=50):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.genomes = [self.initialize_genome() for _ in range(self.population_size)]
        self.fitness_scores = [0] * self.population_size
        self.current_genome = 0

    def initialize_genome(self):
        # Simple genome: weights for a single-layer neural network
        return np.random.uniform(-1, 1, (self.output_size, self.input_size))

    def decide_action(self, bird, game_state):
        genome = self.genomes[self.current_genome]
        if not game_state:
            return 'no_flap'
        
        bird_y = bird.rect.y / game_state['screen_height']
        bird_vel = bird.flap / 10
        pipe_dist = (game_state['next_pipe_x'] - bird.rect.x) / game_state['screen_width']
        pipe_gap = game_state['next_pipe_gap_y'] / game_state['screen_height']
        
        inputs = np.array([bird_y, bird_vel, pipe_dist, pipe_gap])
        output = np.dot(genome, inputs)
        
        return 'flap' if output[0] > 0 else 'no_flap'

    def update_fitness(self, score):
        self.fitness_scores[self.current_genome] += score

    def next_genome(self):
        self.current_genome += 1
        if self.current_genome >= self.population_size:
            self.evolve()
            self.current_genome = 0

    def evolve(self):
        # Selection: Select top 20% genomes
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        top_indices = sorted_indices[:int(0.2 * self.population_size)]
        top_genomes = [self.genomes[i] for i in top_indices]

        # Crossover and mutation to create new population
        new_genomes = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_genomes, 2)
            child = (parent1 + parent2) / 2
            # Mutation
            mutation = np.random.uniform(-0.1, 0.1, child.shape)
            child += mutation
            new_genomes.append(child)

        self.genomes = new_genomes
        self.fitness_scores = [0] * self.population_size

    def reset(self):
        # Reset fitness scores if needed
        self.fitness_scores = [0] * self.population_size
