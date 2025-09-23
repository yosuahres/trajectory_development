import collections
import numpy as np

class AverageFilter:
    def __init__(self, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)

    def update(self, value):
        self.history.append(value)
        return np.mean(self.history)

    def reset(self):
        self.history.clear()
