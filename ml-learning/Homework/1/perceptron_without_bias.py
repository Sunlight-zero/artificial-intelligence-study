from re import X
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DATASET = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

class Perceptron:

    def __init__(self, training_data, input_dim, eta, epochs):
        random.seed(0)
        np.random.seed(0)
        # weights of the perceptron
        self.weight = np.random.randn(input_dim)
        self.epochs = epochs
        self.eta = eta
        self.dataset = [(np.array(p[0]), p[1]) for p in training_data]
    
    def activation(self, x):
        '''
        activation function
        '''

        return 1 if x > 0 else 0
    
    @property
    def wrong_points(self):
        '''
        Search for all the wrong points
        '''
        return [(x, y) for x, y in self.dataset
                if self.activation(self.weight @ x) != y
        ]

    @property
    def is_successful(self):
        '''
        Return true/false which indicates whether the classification is entirely correct.
        '''
        return self.wrong_points == []
    
    def SGD(self, wrong_point):
        '''
        SGD algorithm.
        Update weight according to a wrong point.
        '''
        x, y = wrong_point
        a = self.activation(self.weight @ x)
        dw = self.eta * (y - a) * x
        self.weight += dw
    
    def train(self):
        '''
        Start training.
        '''
        for _ in range(self.epochs):
            if self.is_successful:
                break
            else:
                # Randomly choose a point
                wrong_point = random.sample(self.wrong_points, 1)[0]
                self.SGD(wrong_point)
    
    def plot(self):
        '''
        plot the plane and the data points on 3D coordinates.
        '''

        try:
            a, b, c = self.weight
        except ValueError:
            raise('The dataset must be 3-dimension!')
        else:
            set1 = [p[0] for p in self.dataset if p[1] == 0]
            set2 = [p[0] for p in self.dataset if p[1] == 1]
            
            x = np.array([0, 1])
            y = x
            x, y =np.meshgrid(x, y)

            fig = plt.figure()
            ax = Axes3D(fig)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            ax.set_zlim3d(-0.1, 1.1)
            ax.scatter(*zip(*set1), color='r')
            ax.scatter(*zip(*set2), color='g')
            z = - (a * x + b * y) / c

            ax.plot_surface(x, y, z, color='yellow', alpha=0.5)

            plt.show()
            


if __name__ == '__main__':
    eta = 0.1
    epochs = 200
    # Data processing, adding a extra dimension.
    num = 0
    for i in range(0, 16):
        # Use binary number to get all the combinations of parameters.
        c = [i // 2 ** k % 2 for k in range(0, 4)]
        # print(c)
        # Insert the parameters c into the dataset
        dataset = [(x + [c], y) for (x, y), c in zip(DATASET, c)]
        # print(dataset)
        perceptron = Perceptron(dataset, input_dim=3, eta=eta, epochs=epochs)
        perceptron.train()
        if perceptron.is_successful:
            perceptron.plot()
            print(c) 
            num += 1
    print('The total number is {}'.format(num))
