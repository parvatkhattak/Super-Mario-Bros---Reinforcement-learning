# Super Mario Bros RL Project

This project demonstrates the application of reinforcement learning to train an agent to play the classic *Super Mario Bros*. The model uses deep Q-learning to navigate the game environment and improve performance over time.

## Features

- Implements a reinforcement learning agent with deep Q-learning.
- Includes utilities for environment preprocessing and result visualization.
- Configurable neural network architecture for the agent.
- Log generation and plotting for performance metrics.

## Folder Structure

- **`main.py`**: Entry point for running the training/testing process.
- **`agent.py`**: Core RL agent implementation.
- **`agent_nn.py`**: Defines the neural network for the RL agent.
- **`utils.py`**: Contains utility functions.
- **`wrappers.py`**: Provides custom wrappers for game environment preprocessing.
- **`plot.py`**: Generates performance plots for training metrics.
- **`generate_logfiles.py`**: Helps in generating logs for debugging or analysis.
- **`requirements.txt`**: Contains the list of dependencies required to run the project.

## Setup

### Prerequisites

- Python 3.8+
- A working installation of [Gym](https://www.gymlibrary.dev/) and [Super Mario Bros](https://pypi.org/project/gym-super-mario-bros/).

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Parvatkhattak/Super-Mario-Bros-RL.git
   cd Super-Mario-Bros-RL
2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
3. Run the project
```bash
python main.py
```






