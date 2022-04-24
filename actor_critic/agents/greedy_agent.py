import numpy as np
import torch as torch
from utils import convert_to_char_index

class GreedyAgent():
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = [0, 1] #convert_to_char_index(word_list)

    def __call__(self, states):
        action_log_probs, _ = self.net(states)
        best_action_index = np.argmax(action_log_probs.detach().numpy(), axis = 1)


        return list(best_action_index), list(np.take(self.env_actions, best_action_index, axis = 0))