import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class CustomMazeEnv(gym.Env):
    def __init__(self, maze):
        super(CustomMazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=max(self.maze.shape), shape=(2,), dtype=np.int32)
        self.agent_pos = self.start_pos

    def reset(self):
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos, dtype=np.int32)

    def step(self, action):
        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        new_pos = (self.agent_pos[0] + directions[action][0], self.agent_pos[1] + directions[action][1])
        if (0 <= new_pos[0] < self.maze.shape[0] and
            0 <= new_pos[1] < self.maze.shape[1] and
            self.maze[new_pos] != 1):
            self.agent_pos = new_pos
        done = self.agent_pos == self.goal_pos
        reward = 1 if done else -0.1
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {}

    def render(self):
        maze_render = self.maze.copy()
        maze_render[self.agent_pos] = 4
        plt.imshow(maze_render, cmap='hot', interpolation='nearest')
        plt.title("Maze Environment")
        plt.show()

# Define the maze
maze = [
    [2, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 3]
]

# Create the environment
env = CustomMazeEnv(maze)

# Test the environment
state = env.reset()
done = False
max_steps = 50  # Limit the number of steps to prevent infinite loops

for _ in range(max_steps):
    if done:
        break
    action = env.action_space.sample()  # Random action
    state, reward, done, _ = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    env.render()


# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.maze.shape[0], env.maze.shape[1], env.action_space.n))  # Q-table

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit: best action

    def update_q_value(self, state, action, reward, next_state, done):
        # Update Q-value using the Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + (self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action] * (1 - done))
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_error

    def decay_epsilon(self):
        # Decay epsilon after each episode
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Train the Q-learning agent
def train_agent(env, agent, episodes=1000, max_steps=100):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    return rewards

# Plot the rewards
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()

# Define the maze
maze = [
    [2, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 3]
]

# Create the environment
env = CustomMazeEnv(maze)

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
rewards = train_agent(env, agent, episodes=1000, max_steps=100)

# Plot the training rewards
plot_rewards(rewards)



# Train the Q-learning agent
def train_agent(env, agent, episodes=1000, max_steps=100):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    return rewards

# Plot the rewards to monitor learning progress
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()

# Visualize the learned policy
def visualize_policy(agent, env):
    policy = np.zeros_like(env.maze, dtype=str)
    directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for i in range(env.maze.shape[0]):
        for j in range(env.maze.shape[1]):
            if env.maze[i, j] == 1:  # Wall
                policy[i, j] = "█"
            elif env.maze[i, j] == 2:  # Start
                policy[i, j] = "S"
            elif env.maze[i, j] == 3:  # Goal
                policy[i, j] = "G"
            else:
                best_action = np.argmax(agent.q_table[i, j])
                policy[i, j] = directions[best_action]

    print("Learned Policy:")
    for row in policy:
        print(" ".join(row))

# Simulate the agent's trajectory through the maze
def simulate_trajectory(agent, env):
    state = env.reset()
    trajectory = [tuple(state)]

    for _ in range(100):  # Limit steps to avoid infinite loops
        action = np.argmax(agent.q_table[state[0], state[1]])  # Always take the best action
        next_state, _, done, _ = env.step(action)
        trajectory.append(tuple(next_state))
        state = next_state

        if done:
            break

    # Visualize the trajectory
    maze_with_trajectory = env.maze.copy()
    for pos in trajectory:
        if maze_with_trajectory[pos] == 0:  # Mark trajectory on empty spaces
            maze_with_trajectory[pos] = 4

    plt.imshow(maze_with_trajectory, cmap="hot", interpolation="nearest")
    plt.title("Agent's Trajectory")
    plt.show()

# Define the maze
maze = np.array([
    [2, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 3]
])

# Create the environment
env = CustomMazeEnv(maze)

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
rewards = train_agent(env, agent, episodes=1000, max_steps=100)

# Plot the training rewards
plot_rewards(rewards)

# Visualize the learned policy
visualize_policy(agent, env)

# Simulate and visualize the agent's trajectory
simulate_trajectory(agent, env)


def evaluate_agent(agent, env, episodes=100, max_steps=100):
    success_count = 0

    for episode in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = np.argmax(agent.q_table[state[0], state[1]])  # Always take the best action
            next_state, _, done, _ = env.step(action)
            state = next_state

            if done:  # Goal reached
                success_count += 1
                break

    success_rate = success_count / episodes * 100
    print(f"Agent Success Rate: {success_rate:.2f}% ({success_count}/{episodes} episodes)")
    return success_rate


def analyze_agent_performance(agent, maze_variations, episodes=100, max_steps=100):
    results = {}

    for maze_name, maze in maze_variations.items():
        print(f"Testing on {maze_name}...")
        env = CustomMazeEnv(maze)
        success_rate = evaluate_agent(agent, env, episodes, max_steps)
        results[maze_name] = success_rate

    return results

# Original maze
maze_original = np.array([
    [2, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 3]
])

# Larger maze
maze_large = np.array([
    [2, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 3, 0, 0],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

# Maze with more obstacles
maze_obstacles = np.array([
    [2, 0, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 3]
])

# Define maze variations
maze_variations = {
    "Original Maze": maze_original,
    "Larger Maze": maze_large,
    "Maze with More Obstacles": maze_obstacles
}

# Create the environment for the original maze
env = CustomMazeEnv(maze_original)

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
rewards = train_agent(env, agent, episodes=1000, max_steps=100)

# Plot the training rewards
plot_rewards(rewards)

# Evaluate the agent on the original maze
evaluate_agent(agent, env, episodes=100, max_steps=100)

# Analyze the agent's performance on different maze configurations
results = analyze_agent_performance(agent, maze_variations, episodes=100, max_steps=100)

# Print the results
print("\nPerformance Results:")
for maze_name, success_rate in results.items():
    print(f"{maze_name}: {success_rate:.2f}%")

def experiment_hyperparameters(env, learning_rates, discount_factors, episodes=500, max_steps=100):
    results = {}

    for alpha in learning_rates:
        for gamma in discount_factors:
            print(f"Training with Learning Rate: {alpha}, Discount Factor: {gamma}")
            agent = QLearningAgent(env, learning_rate=alpha, discount_factor=gamma)
            rewards = train_agent(env, agent, episodes=episodes, max_steps=max_steps)
            success_rate = evaluate_agent(agent, env, episodes=100, max_steps=max_steps)
            results[(alpha, gamma)] = {
                "rewards": rewards,
                "success_rate": success_rate
            }
            print(f"Success Rate: {success_rate:.2f}%\n")

    return results

# Define hyperparameter ranges
learning_rates = [0.1, 0.5, 0.9]
discount_factors = [0.8, 0.9, 0.99]

# Run experiments
env = CustomMazeEnv(maze_original)  # Use the original maze
hyperparameter_results = experiment_hyperparameters(env, learning_rates, discount_factors)

# Visualize the results
for (alpha, gamma), result in hyperparameter_results.items():
    plt.plot(result["rewards"], label=f"α={alpha}, γ={gamma}")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Impact of Hyperparameters on Learning")
plt.legend()
plt.show()


# Define the neural network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the DQN agent
class DQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n).to(self.device)
        self.target_model = DQN(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update the model
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Train the DQN agent
def train_dqn_agent(env, agent, episodes=500, max_steps=100, target_update_freq=10):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_model()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    return rewards

# Example usage
env = CustomMazeEnv(maze_original)
agent = DQNAgent(env)
rewards = train_dqn_agent(env, agent, episodes=500, max_steps=100)

# Plot the training rewards
plot_rewards(rewards)