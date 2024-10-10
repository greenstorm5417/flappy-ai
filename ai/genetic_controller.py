import random
import numpy as np

class GeneticController:
    def __init__(self, input_size=6, hidden_sizes=[10, 10], output_size=2, population_size=50, mutation_rate=0.1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        # Initialize population: list of neural networks (as weight matrices)
        self.population = [self.initialize_genome() for _ in range(self.population_size)]
        self.fitness_scores = [0.0 for _ in range(self.population_size)]
        self.current_genome = 0

    def initialize_genome(self):
        """Initialize a genome as a list of weight matrices for each layer."""
        genome = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layer_sizes)-1):
            # Initialize weights with small random values
            weights = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            genome.append(weights)
        return genome

    def decide_action(self, genome, inputs):
        """Feedforward neural network to decide action based on inputs."""
        activation = inputs
        for idx, weights in enumerate(genome):
            z = np.dot(activation, weights)
            if idx < len(genome) -1:
                # Hidden layers use ReLU activation
                activation = np.maximum(0, z)
            else:
                # Output layer uses softmax
                exp_z = np.exp(z - np.max(z))  # For numerical stability
                activation = exp_z / exp_z.sum()
        return np.argmax(activation)

    def evaluate_fitness(self, fitness_function):
        """Evaluate fitness for each genome using the provided fitness function."""
        for idx, genome in enumerate(self.population):
            self.fitness_scores[idx] = fitness_function(genome)

    def select_parents(self, num_parents):
        """Select the top genomes as parents based on fitness."""
        parents_idx = np.argsort(self.fitness_scores)[-num_parents:]
        return [self.population[idx] for idx in parents_idx]

    def crossover(self, parent1, parent2):
        """Crossover two parent genomes to produce a child genome."""
        child = []
        for w1, w2 in zip(parent1, parent2):
            mask = np.random.rand(*w1.shape) < 0.5
            child_weights = np.where(mask, w1, w2)
            child.append(child_weights)
        return child

    def mutate(self, genome):
        """Mutate a genome by adding Gaussian noise."""
        for i in range(len(genome)):
            mutation_mask = np.random.rand(*genome[i].shape) < self.mutation_rate
            gaussian_noise = np.random.randn(*genome[i].shape) * 0.5
            genome[i] += mutation_mask * gaussian_noise
        return genome

    def evolve(self, num_parents=10):
        """Evolve the population to produce the next generation."""
        parents = self.select_parents(num_parents)
        next_generation = parents.copy()

        # Crossover to produce offspring
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_generation.append(child)

        self.population = next_generation
        self.fitness_scores = [0.0 for _ in range(self.population_size)]

    def reset_fitness(self):
        """Reset fitness scores for the next generation."""
        self.fitness_scores = [0.0 for _ in range(self.population_size)]
