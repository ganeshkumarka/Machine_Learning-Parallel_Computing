import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# ======================
# Custom Maze Environment
# ======================

class MazeEnv(gym.Env):
    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.rows, self.cols = self.maze.shape
        self.start = tuple(np.argwhere(self.maze == 'S')[0])
        self.goal = tuple(np.argwhere(self.maze == 'G')[0])
        self.action_space = spaces.Discrete(4)  # 0:up, 1:down, 2:left, 3:right
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.state = None

    def reset(self):
        self.state = self.start
        return self._state_to_idx(self.state)
    
    def step(self, action):
        moves = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        new_row = self.state[0] + moves[action][0]
        new_col = self.state[1] + moves[action][1]
        
        # Check valid move
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            if self.maze[new_row, new_col] != '#':
                self.state = (new_row, new_col)
        
        done = (self.state == self.goal)
        reward = 10 if done else -1  # Reward for reaching goal vs step penalty
        return self._state_to_idx(self.state), reward, done, {}
    
    def render(self, mode='human'):
        grid = np.copy(self.maze)
        grid[self.state] = 'A'
        print(grid)
        print()
    
    def _state_to_idx(self, state):
        return state[0] * self.cols + state[1]

# ======================
# Q-Learning Implementation
# ======================

# Hyperparameters
alpha = 0.1   # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# Maze configuration
maze = [
    ['S', ' ', ' ', ' ', ' '],
    [' ', '#', '#', '#', ' '],
    [' ', '#', ' ', ' ', ' '],
    [' ', '#', ' ', '#', ' '],
    [' ', ' ', ' ', '#', 'G']
]

# Initialize environment and Q-table
env = MazeEnv(maze)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, _ = env.step(action)
        
        # Q-value update
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state
    
    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# ======================
# Visualization of Learned Policy
# ======================

# Convert Q-table to policy
policy = np.argmax(q_table, axis=1)

# Create grid for visualization
grid_size = (env.rows, env.cols)
arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

# Create figure
fig, ax = plt.subplots()
ax.set_xticks(np.arange(-.5, grid_size[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, grid_size[0], 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
ax.set_xticks([])
ax.set_yticks([])

# Plot policy arrows
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        state_idx = i * grid_size[1] + j
        if (i, j) == env.goal:
            ax.text(j, i, 'G', ha='center', va='center', fontsize=20)
        elif env.maze[i, j] == '#':
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='black'))
        else:
            arrow = arrows[policy[state_idx]]
            ax.text(j, i, arrow, ha='center', va='center', fontsize=15)

plt.title("Learned Policy")
plt.show()

# ======================
# Evaluation and Testing
# ======================

success = 0
test_episodes = 10

print("\nTesting Agent:")
for _ in range(test_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        env.render()
    success += (reward == 10)

print(f"\nSuccess Rate: {success/test_episodes * 100}%")