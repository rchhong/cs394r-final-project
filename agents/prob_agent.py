from torch.distributions import Categorical
import numpy as np
from utils import convert_to_char_index
import torch

import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.word_list = word_list
        self.net_time = 0
        self.action_time = 0
        self.conversion_time = 0

    def __call__(self, state, cheat_prob=0, cheat_word=0):

        # start = time.time()

        action_log_probs, state_value = self.net(state)

        # self.net_time += time.time() - start

        # start = time.time()
        # print(action_log_probs.exp())
        dist = Categorical(probs = action_log_probs.exp())

        action = dist.sample()

        if (cheat_prob > np.random.random()):
            action = torch.tensor (cheat_word, dtype=torch.int).to(device)

        # self.action_time += time.time() - start

        # start = time.time()

        ret = convert_to_char_index(self.word_list[action.item()]), dist.log_prob(action), dist.entropy(), state_value

        # self.conversion_time += time.time() - start
        return ret

    def print_times(self):
        print ("Times: {} {} {}".format (self.net_time, self.action_time, self.conversion_time))
        self.net_time = 0
        self.action_time = 0
        self.conversion_time = 0
