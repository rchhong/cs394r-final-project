import numpy as np
import torch as torch
from ...utils import convert_to_char_index

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = convert_to_char_index(word_list)

    def __call__(self, states, device):
        action_log_probs, _ = self.net(torch.Tensor([states], device = device))
        action_probs = np.exp(action_log_probs)

        ret = []
        for prob_dist in action_probs:
            action = np.random.choice(self.env_actions, p = prob_dist)
            ret.append(action)
        return ret
