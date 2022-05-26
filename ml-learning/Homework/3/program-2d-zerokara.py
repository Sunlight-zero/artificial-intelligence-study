"""
这一程序只对正方形的迷宫有用
长方形迷宫会出错。
"""

import numpy as np
from time import sleep


class MazeEnv2D:

    def __init__(self, width, height, epsilon, lr, gamma, terminal):
        """
        An environment containing all the parameters, as well as Q table.
        """
        # Q table: 第 0 列为向左，第 1 列为向右
        self.size = (width, height)
        self.q_table = np.zeros((width, height, 4))
        self.epsilon = epsilon # ε-greedy
        self.lr = lr # learning rate
        self.rewards = np.zeros((width, height))
        self.gamma = gamma # Decline of the rewards.
        self.terminal = np.array(terminal)
    
    @property
    def width(self): return self.size[1]

    @property
    def height(self): return self.size[0]
    
    def set_rewards(self, list_rewards):
        for coordinate, reward in list_rewards:
            self.rewards[coordinate] = reward
    
    def choose(self, state: int, greedy=False):
        """
        选择动作：根据 Q table 选择一个动作
        方法：以 ε 的概率选择 Q table 的最大值，1 - ε 的概率选择 
        """
        # np.random.uniform 将返回 0-1 均匀分布
        if greedy or self.q_table[tuple(state)].any() and np.random.uniform() < self.epsilon:
            choices: np.ndarray = self.q_table[tuple(state)]
            action = int(choices.argmax())
        else:
            action = np.random.randint(0, 4)
        return action
    
    def env_interact(self, state: np.ndarray, action: int):
        """
        Update state and get a reward according to the environment of the maze.
        """
        # If the agent is at the edge of the maze
        # action: 0 for up, 1 for down, 2 for left and 3 for right.
        if action == 0: # up
            # Use positive operator to deepcopy.
            new_state = state + np.array([0, +1]) if state[1] != self.height - 1 else + state
        elif action == 1: # down
            new_state = state + np.array([0, -1]) if state[1] != 0 else + state
        elif action == 2: # left
            new_state = state + np.array([-1, 0]) if state[0] != 0 else + state
        elif action == 3: # right
            new_state = state + np.array([+1, 0]) if state[0] != self.width - 1 else + state

        reward = self.rewards[tuple(new_state)]
        return new_state, reward

    def visualize(self, state):
        """
        Return a group of characters to visualize the training. 
        """
        def map_symbol(reward):
            if reward > 0: return 'T'
            elif reward < 0: return 'U'
            else: return '-'
        
        for i in range(self.height):
            line = list(map(map_symbol, self.rewards[i]))
            if state[0] == i:
                line[state[1]] = 'O'
            print(''.join(line))
        
        print('')
        sleep(0.2)
    
    def train(self, epochs):
        
        for epoch in range(epochs):
            counter = 0
            state = np.array([0, 0])
            flag = True
            while flag:
                action = self.choose(state)
                new_state, reward = self.env_interact(state, action)
                counter += 1
                q_predict = self.q_table[(*state, action)]
                if not (new_state == self.terminal).all():
                    q_target = reward + self.gamma * self.q_table[tuple(new_state)].max()
                else:
                    q_target = reward
                    flag = False
                self.q_table[(*state, action)] += self.lr * (q_target - q_predict)
                state = new_state
                # self.visualize(state)
            
            print('Epoch: {}, Number of steps: {}'.format(epoch, counter))
            # sleep(0.5)
    
    def test(self):
        """
        Test what will the agent do.
        """
        counter = 0
        state = np.array([0, 0])
        self.visualize(state)
        while not (state == self.terminal).all():
            action = self.choose(state, greedy=True)
            state, _ = self.env_interact(state, action)
            self.visualize(state)
            counter += 1
        print('Test steps: {}'.format(counter))
            

if __name__ == '__main__':
    epsilon = 0.9
    lr = 0.1
    gamma = 0.9
    EPOCHS = 30

    maze = MazeEnv2D(4, 4, epsilon, lr, gamma, terminal=[2, 2])
    maze.set_rewards([((1, 2), -1), ((2, 1), -1), ((2, 2), 1)])
    maze.train(epochs=EPOCHS)

    maze.test()
