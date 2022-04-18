import numpy as np

class GreedyAgent():
    def __init__(self, net):
        self.net = net

    def __call__(self, state, device):
        action_log_probs, _ = self.net(state)
        return np.argmax(action_log_probs)