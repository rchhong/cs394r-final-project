import numpy as np
import torch as torch
from ...utils import convert_to_char_index

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = convert_to_char_index(word_list)

    def __call__(self, states):
        action_log_probs, _ = self.net(states)
        action_probs = action_log_probs.exp()

        ret = []
        for prob_dist in action_probs:
            action = np.random.choice(len(self.env_actions), p = prob_dist.detach().numpy())
            ret.append(self.env_actions[action])

        return ret
