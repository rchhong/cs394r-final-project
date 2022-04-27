from torch.distributions import Categorical
import numpy as np
from utils import convert_to_char_index

class GreedyAgent():
    def __init__(self, net, word_list):
        self.net = net
        self.word_list = word_list

    def __call__(self, state):
        action_log_probs, state_value = self.net(state)

        dist = Categorical(probs = action_log_probs.exp())
        action = np.argmax(action_log_probs.detach().numpy())

        return convert_to_char_index(self.word_list[action]), action_log_probs[action], dist.entropy(), state_value