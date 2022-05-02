from torch.distributions import Categorical

from utils import convert_to_char_index

class ProbabilisticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.word_list = word_list

    def __call__(self, state):
        action_log_probs, state_value = self.net(state)

        dist = Categorical(probs = action_log_probs.exp())
        action = dist.sample()

        return convert_to_char_index(self.word_list[action.item()]), dist.log_prob(action), dist.entropy(), state_value
