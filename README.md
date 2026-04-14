# 🐍 Snake AI with Genetic Algorithms (NumPy + Pygame)

## Overview

End-to-end AI project implementing a genetic algorithm to train an autonomous agent to play the Snake game. The system includes a custom neural network built from scratch using NumPy, a full game environment, and a real-time visualization using Pygame.

The project focuses on evolutionary computation, optimization, and emergent behavior without relying on high-level machine learning frameworks such as TensorFlow or PyTorch.

This project was originally developed during high school, reflecting a (very) early exploration of AI and evolutionary algorithms. 
---

## Features

* Genetic algorithm for training (selection, crossover, mutation)
* Fully custom neural network (NumPy-based)
* Complete Snake game implementation
* Real-time graphical visualization using Pygame
* Autonomous agent learning via fitness-based evolution
* No external ML frameworks

---

## How It Works

### Model Architecture

Each individual in the population is a neural network with the following structure:

* Input layer: 8 features
* Hidden layers: 2 layers of 16 neurons each
* Output layer: 4 neurons (movement directions)

**Activations:**

* ReLU for hidden layers
* Sigmoid for output

The network maps the current game state to a movement decision.

---

### Input Representation

The agent receives a compact, hand-crafted state representation:

* Relative position of food:

  * Food left / right / up / down
* Current movement direction:

  * Moving left / right / up / down

This results in an 8-dimensional binary input vector.

---

### Output

The network outputs 4 values corresponding to movement directions:

* Up
* Left
* Down
* Right

The chosen action is the one with the highest activation (argmax).

---

## Genetic Algorithm

### Population

* Size: 800 individuals per generation

### Selection

* Top-performing individuals (top 4%) are selected based on fitness

### Crossover

* Offspring inherit different weight matrices from two parents:

  * Input → Hidden1 from parent 1
  * Hidden1 → Hidden2 from parent 2
  * Hidden2 → Output from parent 1

### Mutation

* Gaussian noise added to weights:

```python
weights += np.random.randn(...) * mutation_rate
```

* Mutation rate: 0.33

---

## Fitness Function

Fitness is defined as the game score:

* +1 per food consumed

Additional mechanics:

* Eating food resets step counter
* Each move consumes a limited number of steps
* Extra steps are granted when food is eaten

This implicitly rewards:

* Efficient navigation
* Survival
* Food acquisition

---

## Game Environment

* Grid-based world (15x15)
* Snake starts at center with random direction
* Food spawns randomly (excluding snake body)
* Game ends on:

  * Collision with wall
  * Self-collision
  * Running out of moves

---

## Visualization (Pygame)

* Real-time rendering of the snake and environment
* Visual feedback during training
* Adjustable simulation speed via frame limiting
* Simple but expressive UI (including animated snake body and eyes)

---

## Installation

```bash
pip install numpy pygame
```

---

## Usage

Run the script:

```bash
python ViperBrain.py
```

The system will:

1. Initialize a population of neural networks
2. Evaluate each agent in the Snake environment
3. Select top performers
4. Generate a new population via crossover and mutation
5. Repeat indefinitely

---

## Results

* Progressive improvement across generations
* Emergence of basic navigation strategies
* Increasing ability to reach food efficiently

Note: Performance depends on randomness and parameter tuning.

---

## Project Structure

```
.
├── ViperBrain.py              # Full implementation (model + game + training loop)
├── ViperBrain Optimiado.py    # Optimized version for testing
├── Juego con red entrenada.py # Game with the fully trained AI model
├── pesos.json                 # Trained neural net weights
```

---

## Design Decisions

* **NumPy over ML frameworks**: full control over implementation
* **Compact input space**: reduces search complexity for GA
* **No backpropagation**: focuses purely on evolutionary strategies
* **Decoupled weight inheritance**: encourages diversity in offspring

---

## Limitations

* No obstacle awareness beyond boundaries and self-collision
* Limited state representation (no full grid perception)
* No model persistence (best agent is not saved)
* Fitness function is relatively simple

---

## Future Improvements

* Add model saving/loading
* Improve state representation (e.g., vision rays or grid encoding)
* Introduce more advanced selection strategies (elitism, tournament)
* Hybrid approach: GA + gradient-based fine-tuning
* Performance metrics and training visualization

---

## Summary

This project demonstrates how intelligent behavior can emerge from simple rules and evolutionary pressure, using a fully custom implementation pipeline. It highlights core concepts in genetic algorithms, neural networks, and simulation-based optimization.
