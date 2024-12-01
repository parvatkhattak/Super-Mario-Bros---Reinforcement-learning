import torch
import csv
import os
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from agent import Agent
from wrappers import apply_wrappers
from utils import *

# Set up model path for saving
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# Set up logging to CSV
log_file = 'training_log.csv'
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Episode Length', 'Loss', 'Epsilon'])  # CSV header row

# Check for CUDA availability
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# Environment and agent configuration
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL =500
NUM_OF_EPISODES = 1000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load pre-trained model if not training
if not SHOULD_TRAIN:
    folder_name = "2024-11-14-12_28_35"
    ckpt_name = "model_2000_iter.pt"
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

# Training loop
for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    episode_length = 0
    episode_loss = 0  # Initialize loss

    # Run episode
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward
        episode_length += 1

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()
            episode_loss = agent.calculate_loss() if hasattr(agent, 'calculate_loss') else 0  # Update loss

        state = new_state

    # Log metrics to CSV after each episode
    epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 'N/A'
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, total_reward, episode_length, episode_loss, epsilon])

    # Print episode summary
    print("Total reward:", total_reward, "Epsilon:", epsilon, 
          "Replay buffer size:", len(agent.replay_buffer), 
          "Learn step counter:", agent.learn_step_counter)

    # Save model checkpoint periodically
    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

env.close()
