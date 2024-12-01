import pandas as pd
import matplotlib.pyplot as plt


# Reload the data with manually specified column names
data = pd.read_csv("training_log.csv", header=None, names=['Episode', 'Total Reward', 'Episode Length', 'Loss', 'Epsilon'])

# Display the first few rows to verify
data.head()


# Plot Total Reward per Episode
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Total Reward'], color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid()
plt.show()

# Plot Episode Length over Episodes
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Episode Length'], color='orange')
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Episode Length per Episode')
plt.grid()
plt.show()

# Plot Loss over Episodes (if applicable)
if 'Loss' in data.columns and data['Loss'].notna().any():
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Loss'], color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.grid()
    plt.show()

# Plot Epsilon over Episodes (if applicable)
if 'Epsilon' in data.columns and data['Epsilon'].notna().any():
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Epsilon'], color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.grid()
    plt.show()
