# Reference: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("Experience", field_names = ["ret", "saved_action"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def append(self, experience : Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # print (len(self.buffer), batch_size)
        selected_indicies = np.random.choice(len(self.buffer), size = batch_size, replace = False)
        sampled_data = [self.buffer[index] for index in selected_indicies]
        
        
        # Unfortunately, the way the data loader expects data and the way data is stored in the buffer is different
        # This causes below scuffness
        returns, actions = zip(*sampled_data)
        return (
            returns,
            actions,
        )




