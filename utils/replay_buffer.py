# Reference: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("experience", field_names = ["state", "action", "next_state", "return"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        selected_indicies = np.random.choice(len(self.buffer), size = batch_size, replace = False)
        sampled_data = [self.buffer[index] for index in selected_indicies]
        # Unfortunately, the way the data loader expects data and the way data is stored in the buffer is different
        # This causes below scuffness
        states, actions, next_states, returns = zip(*sampled_data)

        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(returns)
        )




