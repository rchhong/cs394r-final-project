import argparse
from audioop import avg
import gym
import gym_wordle
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from utils.experience_dataset import generate_dataset
from utils.utils import load_model, load_word_list, save_model
from utils import convert_to_char_index
from utils import STATE_SIZE
from collections import deque

from actor_critic.a2c import ActorCriticNet
from actor_critic.agents.greedy_agent import GreedyAgent
# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--words_dir')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-s', '--sample_size', type=int, default=256)
parser.add_argument('-p', '--capacity', type=int, default=1000)
parser.add_argument('-e', '--num_new_transitions', type=int, default=128)
parser.add_argument('-m', '--embedding_size', type=int, default=32)

args = parser.parse_args()


env = gym.make('Wordle-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x), inplace=False)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

word_list = load_word_list(args.words_dir)
conv_word_list = convert_to_char_index(word_list)
model = ActorCriticNet(STATE_SIZE, word_list, args.embedding_size)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_values = model(state)

    results = []
    for i, prob in enumerate (probs):

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(prob)
        # and sample an action using the distribution
        action = m.sample()
        results.append((action.item(), SavedAction(m.log_prob(action), state_values[i])))

    # save to action buffer
    # model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    # print (SavedAction(m.log_prob(action), state_value))
    return results

def finish_episode(returns, saved_actions):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    # saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    # returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    # for r in model.rewards[::-1]:
    #     # calculate the discounted value
    #     R = r + args.gamma * R
    #     returns.insert(0, R)
    # print ("RETURNS:", returns)
    # print ("ACTIONS:", saved_actions)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        # print (value.shape, torch.tensor([R]).shape)
        value_losses.append(F.smooth_l1_loss(value.reshape((1,)), torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = (torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()) / 50.0
    # perform backprop
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    return loss.detach().cpu().item()

actions_total = np.zeros((51,))

# For Actor-Critic
def generate_a2c_data(num_transitions, env):
    games_data = []

    states = []
    actions = []
    total_returns = []
    next_states = []

    # curr_state = env.reset()
    done = False

    # Statistics
    total_played = 0
    total_rewards = 0
    env.reset()

    states = np.zeros((num_transitions, STATE_SIZE))    
    environments = []
    for i in range (num_transitions):
        env_copy = gym.make('Wordle-v0')
        new_state = env_copy.reset()
        environments.append(env_copy)
        states[i] = new_state

        env_copy.storedrewards = []
        env_copy.storedactions = []
        env_copy.isFin = False

    total_rewards = 0
    for i in range (6):
        results = select_action(states)

        for j, (res, envs) in enumerate (zip (results, environments)):
            if envs.isFin:
                continue
            state, reward, done, _ = envs.step(conv_word_list[res[0]])
            envs.storedrewards.append(reward)
            envs.storedactions.append(res[1])
            envs.isFin = done

            #assign new state
            states[j] = state

            total_rewards += reward

    #all games are finished
    for envs in environments:
        returns = []

        R = 0
        for r in envs.storedrewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)
        actions.extend(envs.storedactions)
        total_returns.extend(returns)
    
    return total_returns, actions, total_rewards / num_transitions
    
def main():
    running_reward = 10
    replay_buffer, dataset = generate_dataset(args.capacity, args.sample_size, args.batch_size)
    
    # returns, actions = generate_a2c_data(10, env)
    # replay_buffer.buffer = deque(list (zip (returns, actions)), args.capacity)
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        model.train()
        state = env.reset()
        returns, actions, avg_rewards = generate_a2c_data(50, env)

        loss_val = finish_episode(returns, actions)

        model.eval()
        if (i_episode % 10 == 0):
            agent = GreedyAgent(model, word_list)
            print(play_game_a2c(agent, False, env))
            print ("Epoch: {}, Avg Rewards: {}, Loss: {}".format (i_episode, avg_rewards, loss_val))
        # print (actions_total)
        # if i_episode % 10 == 0:
        #     # print ("Winrate: {}, GamePlayed: {}, Average_Reward: {}".format (num_wins / num_played, num_played, average_rewards))
        #     games = [play_game_a2c(agent, False) for i in range(100)]
        #     print ("Current Win Rate: {}".format(sum ([len (x[0]) != 6 for x in games]) / 100.0))
        #     print ("Actions: {}, Goal: {}".format (games[0][0], games[0][1]))
            save_model(model, "a2c")


def play_game_a2c(a2c_agent, visualize, env):

    obs = env.reset()
    done = False

    actions = []
    while not done:
        pred = a2c_agent(torch.Tensor(obs.reshape(1,-1)))
        # actions.append(convert_encoded_array_to_human_readable(env_action))
        actions.append (pred)

        obs, reward, done, _ = env.step(pred)
        # print (reward)

    if visualize:
        env.render()

    return actions
if __name__ == '__main__':
    main()