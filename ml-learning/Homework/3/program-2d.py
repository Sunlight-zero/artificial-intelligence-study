import numpy as np
from time import sleep


class MazeEnv2D:

    def __init__(self, width, height, epsilon, lr, gamma, terminal):
        """
        An environment containing all the parameters, as well as Q table.
        """

        self.size = (width, height)
        self.num_actions = 4 # Number of actions
        # A 3D Q table, with x and y the first two dimension.
        self.q_table = np.zeros((width, height, self.num_actions))
        self.epsilon = epsilon # ε-greedy
        self.lr = lr # learning rate
        self.rewards = np.full((width, height), -0.02)
        self.gamma = gamma # Decline of the rewards.
        self.terminal = np.array(terminal) # The coordinate of the endpoint.
    
    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]
    
    def set_rewards(self, list_rewards):
        for coordinate, reward in list_rewards:
            self.rewards[coordinate] = reward
    
    def choose(self, state: int, greedy=False):
        """
        Choose an action according to the state and Q table.
        Method: Choose the maximum of Q table with probability ε,
                and ramdonly choose an action with probability 1 - ε.
                If `greedy` is True, we will choose the maximum every time.
        """
        # np.random.uniform will get the uniform distribution.
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

    def visualize(self, state: np.ndarray):
        """
        Return a group of characters to visualize the training. 
        """
        rewards = self.rewards.T

        def map_symbol(reward: int or float) -> str:
            if reward > 0: return 'T'
            elif reward < -0.5: return 'U'
            else: return '-'
        
        for i in range(self.height):
            # Contruct the map of the i-th line.
            line = list(map(map_symbol, rewards[i]))
            if state[1] == i: # If the agent is at this line
                line[state[0]] = 'O'
            print(''.join(line))
        print('')
        sleep(0.2)
    
    def train(self, num_episodes):
        """
        Start training.
        """
        for episode in range(num_episodes):
            counter = 0 # Connt steps.
            state = np.array([0, 0]) # Start at (0, 0)
            flag = True
            # The main circulation of reinforcement learning. 
            while flag:
                action = self.choose(state)
                new_state, reward = self.env_interact(state, action)
                counter += 1
                q_predict = self.q_table[(*state, action)]
                # Judge if the agent arrives the terminal.
                if not (new_state == self.terminal).all():
                    q_target = reward + self.gamma * self.q_table[tuple(new_state)].max()
                else:
                    q_target = reward
                    flag = False
                self.q_table[(*state, action)] += self.lr * (q_target - q_predict)
                state = new_state
                # self.visualize(state)
            
            print('Episode: {}, Number of steps: {}'.format(episode, counter))
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
    np.random.seed(114514 + 1919810)
    epsilon = 0.7
    lr = 0.1
    gamma = 0.9
    episodes = 200

    maze = MazeEnv2D(7, 8, epsilon, lr, gamma, terminal=[6, 7])
    maze.set_rewards([
        ((0, 1), -1),
        ((1, 4), -1),
        ((2, 1), -1),
        ((2, 2), -1),
        ((2, 3), -1),
        ((3, 0), -1),
        ((4, 3), -1),
        ((4, 4), -1),
        ((4, 5), -1),
        ((4, 6), -1),
        ((5, 3), -1),
        ((5, 7), -1),
        ((6, 5), -1),
        ((6, 7), 1)])
    maze.train(num_episodes=episodes)

    maze.test()
