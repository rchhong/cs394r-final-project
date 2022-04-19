import imp
import numpy as np
import torch as torch
from ...utils import convert_to_char_index

class GreedyAgent():
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = convert_to_char_index(word_list)

    def __call__(self, states, device):
        action_log_probs, _ = self.net(torch.Tensor([states], device = device))
        best_action_index = np.argmax(action_log_probs, dim = 1)

        return np.take(self.env_actions, best_action_index, axis = 0)