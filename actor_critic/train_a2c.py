# from utils.play_game import play_game_a2c
from ensurepip import bootstrap
from os import path
from random import sample
from sched import scheduler
import gym
import gym_wordle
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import numpy as np

from actor_critic.a2c import ActorCriticNet
from agents.prob_agent import ProbabilisticAgent

from utils import STATE_SIZE, load_model, load_word_list, save_model
from utils.play_game import play_game_reinforce
from utils.utils import convert_encoded_array_to_human_readable
<<<<<<< HEAD
from tqdm import tqdm
=======
from datetime import datetime
from torch.distributions import Categorical
import torch.nn as nn


>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034

MODEL_NAME = "a2c"
EMBEDDING_SIZE = 64
rng = np.random.default_rng(12345)
now = datetime.now()

# Statistics
num_wins = 0
num_played = 0
average_rewards_per_batch = 0
sample_game = []
num_wins_batch = 0

<<<<<<< HEAD
#Starts with all ones. Add one if model fails on that word (make it more likely the model returns that word)
word_weights = None
=======
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
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034

# TECHICALLY REINFORCE WITH BASELINE FOR NOW
def train(args):
    word_list = load_word_list(args.words_dir)
<<<<<<< HEAD
    global word_weights
    word_weights = np.ones((len(word_list)))

    model = ActorCriticNet(STATE_SIZE, word_list, EMBEDDING_SIZE)
    agent = ProbabilisticAgent(model, word_list)
    env = gym.make("Wordle-v0")
=======
    model = ActorCriticCartpoleNet()
    agent = ProbabilisticAgentCartpole(model)
    env = gym.make("CartPole-v1")
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034

    # train_logger = None
    # valid_logger = None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'A2C', now.strftime("%Y%m%d-%H%M%S"), 'train'), flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'A2C', now.strftime("%Y%m%d-%H%M%S"), 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        load_model(model, MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    global_step = 0
<<<<<<< HEAD
    # TODO: LR Scheduler
    # TODO: Kaiming initialization
    # TODO: Lower LR
    for num_episodes in tqdm (range(args.num_episodes)):

        # if (num_episodes % 200 == 0):
        #     print("Running Benchmark")
        #     win_amt = 0
        #     #Reset Word_weights here
        #     with torch.no_grad():
        #         for idx, word in enumerate (tqdm(word_list)):
                    
        #             state = env.reset()
        #             env.hidden_word = [ord(c) - 97 for c in word]
        #             for i in range (6):
        #                 action, log_prob_action, entropy, state_value = agent(torch.Tensor(state))
        #                 state, reward, done, __ = env.step(action)

        #                 if (done):
        #                     break
        #             if (reward == 10):
        #                 word_weights[idx] = 1
        #                 win_amt += 1
        #             else:
        #                 word_weights[idx] += 1
                        
        #     print ("Benchmark completed: Win Rate of all words: {}".format(win_amt / len(word_list)))
=======
    for num_episodes in range(args.num_episodes):
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034
        model.train()

        # Play some games, gather experiences
        td_errors, log_prob_actions, entropies, state_values = generate_a2c_data(agent, args.batch_size, args.gamma, env)

        # if train_logger:
        #     train_logger.add_scalar("cumulative_win_rate", num_wins / num_played, global_step=global_step)
        #     train_logger.add_scalar("batch_win_rate", num_wins_batch / args.batch_size, global_step=global_step)
        #     train_logger.add_scalar("num_played", num_played, global_step=global_step)
        #     train_logger.add_scalar("average_rewards", average_rewards_per_batch, global_step=global_step)

        #     actions = [convert_encoded_array_to_human_readable(encoded_action) for encoded_action in sample_game[:-1]]
        #     goal_word = convert_encoded_array_to_human_readable(sample_game[-1])
        #     train_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(actions) + " GOAL: " + goal_word, global_step=global_step)

            # train_logger.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)

        optimizer.zero_grad()
        loss_val = loss(td_errors, log_prob_actions, entropies, args.critic_beta, args.entropy_beta)
        loss_val.backward()
        optimizer.step()
        # scheduler.step()

<<<<<<< HEAD
        if(train_logger):
            train_logger.add_scalar("loss", loss_val, global_step=global_step)

            if(global_step % 100 == 0):
                if(train_logger):
                    # Make the model play a game
                    games = [play_game_a2c(agent, False) for i in range(5)]
                    for game in games:
                        valid_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(game[0]) + " GOAL: " + str(game[1]), global_step=global_step)
                # SAVE MODEL EVERY 100 STEPS
                save_model(model, MODEL_NAME)
=======
        # if(train_logger):
        #         train_logger.add_scalar("loss", loss_val, global_step=global_step)

        #         if(global_step % 100 == 0):
        #             if(train_logger):
        #                 # Make the model play a game
        #                 games = [play_game_reinforce(agent, False) for i in range(5)]
        #                 for game in games:
        #                     valid_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(game[0]) + " GOAL: " + str(game[1]), global_step=global_step)
        #             # SAVE MODEL EVERY 100 STEPS
        #             save_model(model, MODEL_NAME)
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034




        global_step += 1

        save_model(model, MODEL_NAME)

def loss(td_errors, log_prob_actions, entropies, critic_beta, entropy_beta):
    # No divide by 0
    epsilon = torch.finfo(torch.float32).eps
    # No gradient necessary when normalizing

    actor_losses = []
    critic_losses = []

    for td_error, log_prob in zip(td_errors, log_prob_actions):
        # Actor Loss - based on REINFORCE update rule
        # Gradient ASCENT not DESCENT
        actor_losses.append(-(td_error.item()) * log_prob)

        # Critic Loss - TD Error
        critic_losses.append(td_error)

    loss = (torch.stack(actor_losses).sum() - entropy_beta * torch.stack(entropies).sum()) + critic_beta * torch.stack(critic_losses).sum()
    # print("loss:", loss)

    return loss

def save_model(model, name):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name))

def load_model(model, name):
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name)))


warm_start_brackets = [0.01]

def generate_a2c_data(agent, batch_size, gamma, env):
    global num_wins
    global num_played
    global average_rewards_per_batch
    global sample_game
    global num_wins_batch

    states = []
    log_prob_actions = []
    state_values = []
    td_errors = []
    entropies = []
    next_states = []

    # curr_state = env.reset()
    done = False
    total_rewards = 0

    record_data = True
    num_wins_batch = 0

    for _ in range(batch_size):
        state = env.reset(weight = word_weights)

        ep_reward = 0
        rewards = []
        curr_state_values = []

        if(record_data):
            del sample_game[:]

        for t in range(1, 10000):
            states.append(state)
            # select action from policy
<<<<<<< HEAD
            action, log_prob_action, entropy, state_value = agent(torch.Tensor(state), t * warm_start_brackets[min (num_played // 1000, len (warm_start_brackets) - 1) ] / 6, env.hidden_word_idx)
            if(record_data):
                sample_game.append(action)
=======
            action, log_prob_action, entropy, state_value = agent(torch.Tensor(state))
            # if(record_data):
            #     sample_game.append(action)
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034
            # print("action:", action)
            # take the action
            # env.render()
            state, reward, done, __ = env.step(action)

            rewards.append(reward)
            log_prob_actions.append(log_prob_action)
            curr_state_values.append(state_value)
            entropies.append(entropy)

            # model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # if(record_data):
        #     sample_game.append(env.hidden_word)
        # record_data = False

        total_rewards += ep_reward
        num_played += 1
<<<<<<< HEAD
        if(action == list(env.hidden_word)):
            num_wins += 1
            num_wins_batch += 1
            word_weights[env.hidden_word_idx] = 1
        else:
            word_weights[env.hidden_word_idx] = word_weights[env.hidden_word_idx] * 0.95 + 1
=======
        # if(action == list(env.hidden_word)):
        #     num_wins += 1
        #     num_wins_batch += 1
>>>>>>> 0553c277dd17f6fae4b89a02ca6102eae9cdf034

        curr_td_errors = []
        # print("rewards:", rewards)
        # print("state_values:", curr_state_values)
        for i in range(len(rewards)):
            # calculate the discounted value
            next_state_value = 0
            if(i + 1 < len(rewards)):
                next_state_value = curr_state_values[i + 1]

            td_error = torch.Tensor([rewards[i]]) + gamma * next_state_value - curr_state_values[i]
            curr_td_errors.append(td_error)

        # print("td_errors:", curr_td_errors)
        td_errors.extend(curr_td_errors)
        state_values.extend(curr_state_values)

    average_rewards_per_batch = total_rewards / batch_size
    print(average_rewards_per_batch)
    return td_errors, log_prob_actions, entropies, state_values

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_episodes', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=.9)
    parser.add_argument('-m', '--embedding_size', type=int, default=32)

    parser.add_argument('--critic_beta', type=float, default=1)
    parser.add_argument('--entropy_beta', type=float, default=.05)

    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
