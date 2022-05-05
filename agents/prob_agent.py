from torch.distributions import Categorical
import numpy as np
from utils import convert_to_char_index
import torch

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.word_list = word_list

    def __call__(self, state, cheat_prob=0, cheat_word=0):
        action_log_probs, state_value = self.net(state)

        # print(action_log_probs.exp())
        dist = Categorical(probs = action_log_probs.exp())

        action = dist.sample()
        
        if (cheat_prob > np.random.random()):
            action = torch.tensor (cheat_word, dtype=torch.int)


        return convert_to_char_index(self.word_list[action.item()]), dist.log_prob(action), dist.entropy(), state_value
