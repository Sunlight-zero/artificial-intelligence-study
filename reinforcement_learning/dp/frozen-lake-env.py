import numpy as np
from typing import Optional
import random


class FrozenLake:
    def __init__(self, nrow: int, ncol: int, holes: list[int]):
        self.nrow = nrow
        self.ncol = ncol
        self.n_states = nrow * ncol
        # 0 for a empty slot, 1 for the terminal, -1 for a hole
        self.states = np.zeros(self.n_states, dtype=np.int32)
        self.states[holes] = -1
        self.states[self.n_states - 1] = 1
        # for action: 0 for up, 1 for down, 2 for left, 3 for right.
        self.n_actions = 4
        # Initialize transition matrix and reward
        self.transition = np.zeros(
            (self.n_states, self.n_actions, self.n_states), dtype=np.float64
        )
        self.reward = np.zeros(
            (self.n_states, self.n_actions), dtype=np.float64
        )
        for x in range(nrow):
            for y in range(ncol):
                can_up = can_down = can_left = can_right = True
                if x == 0:
                    can_up = False
                if x == nrow - 1:
                    can_down = False
                if y == 0:
                    can_left = False
                if y == ncol - 1:
                    can_right = False
                state = x * ncol + y
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
                if self.states[state] == -1:
                    self.reward[state, action] = -100
                elif self.states[state] == 1:
                    self.reward[state, action] = 1000
    
    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        """
        return (state, reward, is_end)
        """
        reward = self.reward[state, action]
        if self.states[state] != 0:
            return (-1, reward, True)
        distribution = self.transition[state, action]
        next_state = random.choices(np.arange(self.n_states), distribution)
        return (next_state, reward, False)


if __name__ == "__main__":
    env = FrozenLake(4, 4, [6, 9])
