# from collections import deque
# from pickle import FALSE

# from matplotlib.pyplot import text
# import torch.utils.tensorboard as tb
# from os import path
# import torch

# import gym_wordle
# from actor_critic.a2c import ActorCriticNet
# from actor_critic.agents.prob_agent import ProbabilisticAgent
# from utils.generate_data import generate_data
# from utils.play_game import play_game_a2c
# from utils.utils import load_model, load_word_list, save_model
# from utils.experience_dataset import generate_dataset
# from utils import STATE_SIZE

from os import path
import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np

MODEL_NAME = "a2c"

class ActorCriticCartpoleNet(torch.nn.Module):
    def __init__(self):
        super(ActorCriticCartpoleNet, self).__init__()
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

        return action_prob, state_values

class ProbabilisticAgentCartpole:
    def __init__(self, net):
        self.net = net

    def __call__(self, state):
        action_probs, state_value = self.net(state)

        dist = Categorical(probs = action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy(), state_value

def generate_a2c_data(agent, num_transitions, gamma, env):
    games_data = []

    states = []
    log_prob_actions = []
    state_values = []
    total_returns = []
    entropies = []
    next_states = []

    # curr_state = env.reset()
    done = False

    # Statistics
    total_played = 0
    total_rewards = 0

    for _ in range(num_transitions):
        state = env.reset()

        ep_reward = 0
        rewards = []

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action, log_prob_action, entropy, state_value = agent(torch.Tensor(state))

            # take the action
            # env.render()
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            log_prob_actions.append(log_prob_action)
            state_values.append(state_value)
            entropies.append(entropy)

            # model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        total_rewards += ep_reward

        returns = []

        R = 0
        for r in rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

        total_returns.extend(returns)
    print (total_rewards / num_transitions)
    return total_returns, log_prob_actions, entropies, state_values

# TECHICALLY REINFORCE WITH BASELINE FOR NOW
def train(args):
    # word_list = load_word_list(args.words_dir)
    model = ActorCriticCartpoleNet()
    agent = ProbabilisticAgentCartpole(model)
    env = gym.make("CartPole-v1")

    # train_logger = None
    # valid_logger = None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        load_model(model, MODEL_NAME)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, )

    # Statistics
    # num_wins = 0
    # num_played = 0
    # average_rewards = 0

    # replay_buffer, dataset = generate_dataset(args.capacity, args.sample_size, args.batch_size)

    global_step = 0
    for num_episodes in range(args.num_episodes):
        model.train()

        # Play a game and append to the replay buffer
        total_returns, log_prob_actions, entropies, state_values = generate_a2c_data(agent, args.capacity, args.gamma, env)

        # if train_logger:
        #     train_logger.add_scalar("win_rate", num_wins / num_played, global_step=global_step)
        #     train_logger.add_scalar("num_played", num_played, global_step=global_step)
        #     train_logger.add_scalar("average_rewards", average_rewards, global_step=global_step)

        # If we have enough data, begin training
        # if len(replay_buffer) == args.capacity:
        # # Train on minibatch
        #     for (states, actions, next_states, returns) in dataset:
        optimizer.zero_grad()
        loss_val = loss(total_returns, log_prob_actions, entropies, state_values, args.entropy_beta)

                # if(train_logger):
                #         train_logger.add_scalar("loss", loss_val, global_step=global_step)

                # if(global_step % 100 == 0):
                #     if(train_logger):
                #         # Make the model play a game
                #         games = [play_game_a2c(agent, False) for i in range(1)]
                #         for game in games:
                #             valid_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(game[0]) + " GOAL: " + str(game[1]), global_step=global_step)
                #     # SAVE MODEL EVERY 100 STEPS
                #     save_model(model, MODEL_NAME)


        loss_val.backward()
        optimizer.step()

        global_step += 1


        # new_data, new_num_wins, new_num_played, new_average_rewards = generate_a2c_data(args.num_new_transitions, args.gamma, agent, model, env, device)
        # for experience in new_data:
        #     replay_buffer.append(experience)

        # average_rewards = (average_rewards * num_played  + new_average_rewards * new_num_played) / (num_played + new_num_played)
        # num_wins += new_num_wins
        # num_played += new_num_played


        # if train_logger:
        #     train_logger.add_scalar("win_rate", num_wins / num_played, global_step=global_step)
        #     train_logger.add_scalar("num_played", num_played, global_step=global_step)
        #     train_logger.add_scalar("average_rewards", average_rewards, global_step=global_step)


        save_model(model, MODEL_NAME)

def loss(total_returns, log_prob_actions, entropies, state_values, entropy_beta):
    # No divide by 0
    epsilon = torch.finfo(torch.float32).eps
    # No gradient necessary when normalizing

    returns = torch.tensor(total_returns)
    returns = (returns - returns.mean()) / (returns.std() + epsilon)

    actor_losses = []
    critic_losses = []

    for ret, log_prob, state_value in zip(total_returns, log_prob_actions, state_values):
        advantage = ret - state_value.item()

        # Actor Loss - based on REINFORCE update rule
        # Gradient ASCENT not DESCENT
        actor_losses.append(-advantage * log_prob)

        # Critic Loss - MSE
        critic_losses.append(F.smooth_l1_loss(state_value, torch.tensor([ret])))

    loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum() - torch.stack(entropies).sum()
    # print("loss:", loss)

    return loss

def save_model(model, name):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name))

def load_model(model, name):
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('--log_dir')
    # parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_episodes', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-2)
    parser.add_argument('-g', '--gamma', type=float, default=.99)
    parser.add_argument('-m', '--embedding_size', type=int, default=32)

    parser.add_argument('--critic_beta', type=float, default=1)
    parser.add_argument('--entropy_beta', type=float, default=.05)

    # As dumb as this looks, make sure that all of these are the same for now
    parser.add_argument('-b', '--batch_size', type=float, default=64)
    parser.add_argument('-s', '--sample_size', type=float, default=64)
    parser.add_argument('-p', '--capacity', type=int, default=64)
    parser.add_argument('-e', '--num_new_transitions', type=int, default=64)


    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
