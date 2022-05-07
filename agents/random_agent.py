from torch.distributions import Categorical
import numpy as np
from utils import convert_to_char_index
import torch

class RandomAgent:
    def __init__(self, word_list):
        self.word_list = word_list

    def __call__(self, state):
        # print(action_log_probs.exp())
        action = np.random.choice(len(self.word_list))

        return convert_to_char_index(self.word_list[action]), None, None, None
