import numpy as np

class ProbabilisticAgent:
    def __init__(self, net):
        self.net = net

    def __call__(self, state, device):
        # TODO: move convert to one_hot to its own utility file, symlink word_data
        action_log_probs, _ = self.net(state)
        action_probs = np.exp(action_log_probs)

        return np.random.choice(p = action_probs)
