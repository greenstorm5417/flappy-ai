import neat
import pickle
import os

class NEATController():
    def __init__(self, config_path, genome_path=None):
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.population = neat.Population(self.config)
        self.genome = None
        self.net = None  # Neural network

        if genome_path:
            if not os.path.exists(genome_path):
                raise ValueError(f"Genome file '{genome_path}' does not exist.")
            try:
                with open(genome_path, 'rb') as f:
                    self.genome = pickle.load(f)
                self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
                print(f"Loaded genome from '{genome_path}' and created neural network.")
            except Exception as e:
                raise ValueError(f"Failed to load genome from '{genome_path}': {e}")
        else:
            print("No genome path provided. NEATController will manage the population.")

    def run(self, eval_genomes_function, n_generations):
        # Add reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        # Run NEAT
        winner = self.population.run(eval_genomes_function, n_generations)
        self.genome = winner
        self.net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        return winner

    def activate(self, inputs):
        if self.net is None:
            raise ValueError("Neural network not initialized. Ensure a genome is loaded.")
        return self.net.activate(inputs)

    def save_genome(self, filepath):
        if self.genome is None:
            raise ValueError("No genome to save.")
        with open(filepath, 'wb') as f:
            pickle.dump(self.genome, f)
        print(f"Genome saved to '{filepath}'.")

    def reset(self):
        # If you have any reset logic for NEATController, implement it here
        pass
