import numpy as np
import random
from tqdm import tqdm
from time import sleep


class FrozenLake:
    def __init__(self, nrow: int, ncol: int, holes: list[int]):
        self.nrow = nrow
        self.ncol = ncol
        self.n_states = nrow * ncol + 1
        # 0 for a empty slot, 1 for the terminal, -1 for a hole, 2 for an absorption
        self.states = np.zeros(self.n_states, dtype=np.int32)
        self.states[holes] = -1
        self.states[self.n_states - 2] = 1
        self.states[self.n_states - 1] = 2
        self.absorption = self.n_states - 1
        # for action: 0 for up, 1 for down, 2 for left, 3 for right.
        self.n_actions = 4
        # Initialize transition matrix and reward
        self.transition = np.zeros(
            (self.n_states, self.n_actions, self.n_states), dtype=np.float64
        )
        self.reward = np.zeros(
            (self.n_states, self.n_actions), dtype=np.float64
        )
        for action in range(self.n_actions):
            self.transition[self.absorption, action, self.absorption] = 1
        
        for x in range(nrow):
            for y in range(ncol):
                state = x * ncol + y
                if self.states[state] != 0:
                    if self.states[state] == -1:
                        reward = -100
                    elif self.states[state] == 1:
                        reward = 1000
                    # Terminate
                    for action in range(self.n_actions):
                        self.transition[state, action, self.absorption] = 1
                        self.reward[state, action] = reward
                    continue
                can_up = can_down = can_left = can_right = True
                if x == 0:
                    can_up = False
                if x == nrow - 1:
                    can_down = False
                if y == 0:
                    can_left = False
                if y == ncol - 1:
                    can_right = False
                for action in range(self.n_actions):
                    if can_up and action != 1:
                        next_state = (x - 1) * ncol + y
                        self.transition[state, action, next_state] = 1
                    if can_down and action != 0:
                        next_state = (x + 1) * ncol + y
                        self.transition[state, action, next_state] = 1
                    if can_left and action != 3:
                        next_state = x * ncol + y - 1
                        self.transition[state, action, next_state] = 1
                    if can_right and action != 2:
                        next_state = x * ncol + y + 1
                        self.transition[state, action, next_state] = 1
                    if (s := np.sum(self.transition[state, action])) != 0:
                        self.transition[state, action] *= 1 / s
    
    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """
        return (state, reward, is_end)
        """
        reward = self.reward[state, action]
        if self.states[state] != 0:
            return (-1, reward, True)
        distribution = self.transition[state, action]
        next_state = random.choices(np.arange(self.n_states), distribution)[0]
        return (next_state, reward, False)

env = FrozenLake(4, 4, [5, 7, 11, 12])
n_states, n_actions = env.n_states, env.n_actions

def epsilon_greedy(state: int, epsilon: float, q_table: np.ndarray):
    if np.random.random() < epsilon:
        action = np.random.choice(n_actions)
    else:
        action = np.argmax(q_table[state])
    return action

# Q-Learning

q_table = np.random.random((n_states, n_actions))
epsilon = 0.2
gamma = 0.95
alpha = 0.1
n_epochs = 10000
for epoch in tqdm(range(n_epochs)):
    state = 0
    is_end = False
    for t in range(1000):
        if is_end:
            break
        action = epsilon_greedy(state, epsilon, q_table)
        next_state, reward, is_end = env.step(state, action)
        max_action = np.argmax(q_table[next_state])
        q_table[state, action] += alpha * (
            reward + gamma * q_table[next_state, max_action] - q_table[state, action]
        )
        state = next_state
