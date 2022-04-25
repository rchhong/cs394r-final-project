import numpy as np
import torch as torch
from utils import convert_to_char_index
from torch.distributions import Categorical

def select_action(state):
    state = torch.from_numpy(state).reshape(1,-1).float()
    probs, state_value = model(state)
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    # print (SavedAction(m.log_prob(action), state_value))
    return action.item(), SavedAction(m.log_prob(action), state_value)

class GreedyAgent():
    def __init__(self, net, word_list):
        self.net = net
        self.env_actions = convert_to_char_index(word_list)

    def __call__(self, states):
        action_log_probs, _ = self.net(states)
        m = Categorical(action_log_probs)

        # and sample an action using the distribution
        action = m.sample()
        # print (action_log_probs)
        # best_action_index = np.argmax(action_log_probs.detach().numpy(), axis = 1)


        return self.env_actions[action]