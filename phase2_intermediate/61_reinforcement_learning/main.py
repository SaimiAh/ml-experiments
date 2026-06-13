import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Q-learning basics: Reinforcement Learning
# We will use a simple grid world to demonstrate Q-learning

# Define the grid world
grid_size = 5
actions = ['up', 'down', 'left', 'right']

# Define the reward function
def reward(state):
    if state == (0, 0):
        return -10
    elif state == (grid_size - 1, grid_size - 1):
        return 10
    else:
        return -1

# Define the Q-learning algorithm
def q_learning(num_iterations, learning_rate, discount_factor):
    q_values = {}
    for i in range(num_iterations):
        state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        action = np.random.choice(actions)
        next_state = state
        if action == 'up' and state[0] > 0:
            next_state = (state[0] - 1, state[1])
        elif action == 'down' and state[0] < grid_size - 1:
            next_state = (state[0] + 1, state[1])
        elif action == 'left' and state[1] > 0:
            next_state = (state[0], state[1] - 1)
        elif action == 'right' and state[1] < grid_size - 1:
            next_state = (state[0], state[1] + 1)

        reward_value = reward(next_state)
        if (state, action) not in q_values:
            q_values[(state, action)] = 0
        if next_state not in [s for (s, a) in q_values]:
            q_values[(next_state, 'up')] = 0
            q_values[(next_state, 'down')] = 0
            q_values[(next_state, 'left')] = 0
            q_values[(next_state, 'right')] = 0

        q_values[(state, action)] += learning_rate * (reward_value + discount_factor * max(q_values[(next_state, a)] for a in actions) - q_values[(state, action)])
    return q_values

if __name__ == "__main__":
    num_iterations = 1000
    learning_rate = 0.1
    discount_factor = 0.9

    q_values = q_learning(num_iterations, learning_rate, discount_factor)
    for state in [(i, j) for i in range(grid_size) for j in range(grid_size)]:
        for action in actions:
            if (state, action) in q_values:
                print(f"State: {state}, Action: {action}, Q-value: {q_values[(state, action)]}")
```
Note that due to the random nature of Q-learning, the exact output will vary with each run. This will give a basic understanding of the Q-learning process.