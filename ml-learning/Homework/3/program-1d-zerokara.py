from copy import deepcopy
import numpy as np
from time import sleep


class MazeEnv:

    def __init__(self, num_states, epsilon, lr, gamma):
        # Q table: 第 0 列为向左，第 1 列为向右
        self.num_states = num_states
        self.q_table = np.zeros((num_states - 1, 2))
        self.epsilon = epsilon
        self.lr = lr
        self.is_successful = False
        self.rewards = np.zeros(num_states)
        self.rewards[-1] = 1
        self.visualized_states = ['-'] * (num_states - 1) + ['T']
        self.gamma = gamma
    
    def choose(self, state: int, greedy=False):
        """
        选择动作：根据 Q table 选择一个动作
        方法：以 ε 的概率选择 Q table 的最大值，1 - ε 的概率选择 
        """
        # np.random.uniform 将返回 0-1 均匀分布
        if greedy or self.q_table[state].any() and np.random.uniform() < self.epsilon:
            choices: np.ndarray = self.q_table[state]
            action = int(choices.argmax())
        else:
            action = np.random.randint(0, 2)
        return action
    
    def env_interact(self, state, action):
        """
        根据当前的状态和动作，更新环境参数，并获得奖励
        """
        # 在最左边，且往左
        if state == 0 and action == 0:
            new_state = 0
        elif state == self.num_states and action == 1:
            raise Exception("The agent has arrived the terminal!")
        else:
            new_state = state - 1 if action == 0 else state + 1
        reward = self.rewards[new_state]
        return new_state, reward

    def visualize(self, state):
        """
        Return a group of characters to visualize the training. 
        """
        v_states = deepcopy(self.visualized_states)
        v_states[state] = 'O'
        print(''.join(v_states))
        sleep(0.2)
    
    def train(self, epochs):
        
        for epoch in range(epochs):
            counter = 0
            state = 0
            flag = True
            while flag:
                action = self.choose(state)
                new_state, reward = self.env_interact(state, action)
                counter += 1
                q_predict = self.q_table[state, action]
                if new_state < self.num_states - 1:
                    q_target = reward + self.gamma * self.q_table[new_state].max()
                else:
                    q_target = reward
                    flag = False
                self.q_table[state, action] += self.lr * (q_target - q_predict)
                state = new_state
                self.visualize(state)
            
            print('Epoch: {}, Number of steps: {}'.format(epoch, counter))
    
    def test(self):
        """
        Test what will the agent do.
        """
        counter = 0
        state = 0
        while state != self.num_states - 1:
            action = self.choose(state, greedy=True)
            state, _ = self.env_interact(state, action)
            self.visualize(state)
            counter += 1
        print('Test steps: {}'.format(counter))
            

if __name__ == '__main__':
    num_states = 6
    epsilon = 0.9
    lr = 0.1
    gamma = 0.9
    EPOCHS = 13

    maze = MazeEnv(num_states, epsilon, lr, gamma)
    maze.train(epochs=EPOCHS)

    maze.test()
