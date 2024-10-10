# Flappy Bird AI Controllers

This project is an implementation of various AI controllers that play a Flappy Bird-like game. The controllers include:

- **Deep Q-Learning (DQL)**: A reinforcement learning approach.
- **NEAT (NeuroEvolution of Augmenting Topologies)**: A genetic algorithm approach.
- **Genetic Algorithm**: A simple evolutionary algorithm for evolving neural networks.
- **Rule-Based Controller**: A set of predefined rules that controls the bird.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [AI Controllers](#ai-controllers)
- [License](#license)

## Features

- Train a bird using various AI techniques to play the Flappy Bird game.
- Visualize the neural network as the bird plays.
- Predefined rules for controlling the bird (for comparison).
- Logging of Q-values and training progress for reinforcement learning models.

## Installation

### Prerequisites

Ensure you have `Python 3.8+` installed on your machine.

### Install Dependencies

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

This will install the necessary Python libraries such as `pygame`, `torch`, `numpy`, and `neat-python`.

### Additional Setup

Ensure that the assets required for the game (like images for the bird, pipes, background, etc.) are located in the `assets/` directory. If these are missing, the game may not function properly.

## Usage

There are multiple controllers that can be used to run the game. Each controller is located in the `train/` folder, and they are implemented as Python scripts.

### Running the Game with a Predefined Controller

To run the game with different controllers:

1. **DQL Controller**:
   
   Train the bird using Deep Q-Learning by running:

   ```bash
   python train/train_dql.py
   ```

2. **NEAT Controller**:

   Train the bird using the NEAT algorithm by running:

   ```bash
   python train/train_neat.py
   ```

3. **Genetic Algorithm**:

   Train the bird using a genetic algorithm by running:

   ```bash
   python train/genetic_train.py
   ```

4. **Rule-Based Controller**:

   Run the game with a predefined rule-based controller by running:

   ```bash
   python tests/test_rule_based.py
   ```

### Testing

Test scripts are available for each controller under the `tests/` folder:

- **Test DQL**: `tests/test_dql.py`
- **Test NEAT**: `tests/test_neat.py`
- **Test Genetic Algorithm**: `tests/test_genetic.py`
- **Test Rule-Based**: `tests/test_rule_based.py`

## File Structure

- **ai/**: Contains the AI controllers (DQL, NEAT, Genetic, Rule-Based).
- **game/**: Contains the game components (bird, pipes, background, assets).
- **train/**: Scripts for training AI controllers.
- **tests/**: Test scripts for evaluating the controllers.

## AI Controllers

1. **Deep Q-Learning (DQL)**: Uses a neural network to estimate Q-values for actions (flap or no flap) based on the current game state. The model is trained through reinforcement learning.

2. **NEAT (NeuroEvolution)**: Evolves neural networks through genetic algorithms. NEAT evolves both the network structure and weights simultaneously.

3. **Genetic Algorithm**: A more traditional genetic algorithm approach that evolves fixed neural networks based on fitness scores.

4. **Rule-Based Controller**: A controller that uses a set of predefined rules to decide when to make the bird flap.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
