import numpy as np
import torch as torch
from torch.distributions import Categorical
from utils import convert_to_char_index

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = convert_to_char_index(word_list)

    def __call__(self, states):
        action_log_probs, _ = self.net(states)
        action_probs = action_log_probs.exp()

        actions = []
        action_probs = []
        state_value = []

        for prob_dist in action_probs:
            dist = Categorical(probs = prob_dist.detach().numpy())
            action_index = dist.sample().detach().item()

            indicies.append(action_index)
            actions.append(self.env_actions[action_index])

        return actions, action_probs, state_value
