"""
萤火虫问题
当一个萤火虫点亮时，萤火虫会影响它周围的 K 个萤火虫
当一个萤火虫熄灭时，它会观察周围的 K 个萤火虫
"""

"""
Modeling Assumption:
1. The time is continuous, i.e. we have float time value here.
2. Since time is continuous, we think that it is impossible 
   for two fireflies to fire simutaneously.
3. 
"""

import numpy as np
import matplotlib.pyplot as plt

FIRING_SPAN = 25
RESTING_SPAN = 75
WEIGHT = 0.7
NUMBERS = 100
K = 5
MAX_NUM_LOOPING = 120


class Fireflies:

    def __init__(self, num_fireflies, rectangle_size=(1, 1), positions=None):
        """
        A class to store all the fireflies.
        """
        self.size = np.array(rectangle_size)
        self.num_fireflies = num_fireflies
        self.k = K
        self.max = 0
        # Initial the spans
        init_spans = np.random.rand(num_fireflies) * (FIRING_SPAN + RESTING_SPAN)
        # decide which is fired and which is extinguished.
        self.statuses = (init_spans < FIRING_SPAN)
        # Initial the first lasting span
        self.next_event_spans = \
            np.array(list(map(lambda t: t if t < FIRING_SPAN else t - FIRING_SPAN, init_spans)))
        # positions of the fireflies.
        self.positions = positions or np.random.rand(num_fireflies, 2) * np.array(rectangle_size)
        # adjacency matrix storing all the neighbors.
        self.adjacency = np.zeros((num_fireflies, num_fireflies), dtype=np.int8)
        for i in range(num_fireflies - 1):
            position = self.positions[i]
            # Sort all the nodes by distance.
            neighbors_by_dis = np.argsort([np.linalg.norm(position - self.positions[j]) 
            for j in range(num_fireflies)])
            # Pick out the k-nearest nodes.
            kneighbors = neighbors_by_dis[1:self.k+1]
            # Update the adjacency matrix
            self.adjacency[i, kneighbors] = 1
    
    def get_k_neighbors(self, firefly_id):
        """
        Pick out the k-nearest neighbors according to the adjacency matrix.
        """
        return [i for i, label in enumerate(self.adjacency[firefly_id]) if label == 1]
    
    def change(self, firefly_id):
        """
        Changes the status of a firefly, and affecting others.
        """
        if self.statuses[firefly_id]:
            # If it is firing, turns into resting.
            self.statuses[firefly_id] = 0
            self.next_event_spans[firefly_id] = RESTING_SPAN
            # Search for k_neighbors. If lighting, diminish its own resting span.
            for neighbor_id in self.get_k_neighbors(firefly_id):
                if self.statuses[neighbor_id]:
                    self.next_event_spans[firefly_id] *= WEIGHT
        else:
            # If it is resting, turns into firing.
            self.statuses[firefly_id] = 1
            self.next_event_spans[firefly_id] = FIRING_SPAN
            # Search for k_neighbors. If resting, diminish their resting spans.
            for neighbor_id in self.get_k_neighbors(firefly_id):
                if not self.statuses[neighbor_id]:
                    self.next_event_spans[neighbor_id] *= WEIGHT

    def start_simulation(self):
        for _ in range(MAX_NUM_LOOPING * self.num_fireflies):
            # Find the next event:
            next_firefly = np.argmin(self.next_event_spans)
            time_span = self.next_event_spans[next_firefly]
            # Update all the event spans
            self.next_event_spans -= time_span
            self.change(next_firefly)
            # Update the maximum.
            self.max = max(np.sum(self.statuses), self.max)

if __name__ == '__main__':
    np.random.seed(100)
    num_fireflies = 100
    print('Current: configuration:')
    print('Number of fireflies:', num_fireflies)
    print('Weight:', WEIGHT)
    print('K', K)
    fireflies = Fireflies(100)
    fireflies.start_simulation()
    print('The max fireflies is', fireflies.max)
