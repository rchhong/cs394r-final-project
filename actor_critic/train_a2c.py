# from utils.play_game import play_game_a2c
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
from actor_critic.agents.prob_agent import ProbabilisticAgent

from utils import STATE_SIZE, load_model, load_word_list, save_model
from utils.play_game import play_game_a2c
from utils.utils import convert_encoded_array_to_human_readable

MODEL_NAME = "a2c"
EMBEDDING_SIZE = 32
rng = np.random.default_rng(12345)

# Statistics
num_wins = 0
num_played = 0
average_rewards_per_batch = 0
sample_game = []
num_wins_batch = 0

# TECHICALLY REINFORCE WITH BASELINE FOR NOW
def train(args):
    word_list = load_word_list(args.words_dir)
    model = ActorCriticNet(STATE_SIZE, word_list, EMBEDDING_SIZE)
    agent = ProbabilisticAgent(model, word_list)
    env = gym.make("Wordle-v0")

    train_logger = None
    valid_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        load_model(model, MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    global_step = 0
    # TODO: LR Scheduler
    # TODO: Kaiming initialization
    # TODO: Lower LR
    for num_episodes in range(args.num_episodes):
        model.train()

        # Play some games, gather experiences
        total_returns, log_prob_actions, entropies, state_values = generate_a2c_data(agent, args.batch_size, args.gamma, env)

        if train_logger:
            train_logger.add_scalar("cumulative_win_rate", num_wins / num_played, global_step=global_step)
            train_logger.add_scalar("batch_win_rate", num_wins_batch / args.batch_size, global_step=global_step)
            train_logger.add_scalar("num_played", num_played, global_step=global_step)
            train_logger.add_scalar("average_rewards", average_rewards_per_batch, global_step=global_step)

            actions = [convert_encoded_array_to_human_readable(encoded_action) for encoded_action in sample_game[:-1]]
            goal_word = convert_encoded_array_to_human_readable(sample_game[-1])
            train_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(actions) + " GOAL: " + goal_word, global_step=global_step)

            # train_logger.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)

        optimizer.zero_grad()
        loss_val = loss(total_returns, log_prob_actions, entropies, state_values, args.critic_beta, args.entropy_beta)
        loss_val.backward()
        optimizer.step()
        # scheduler.step()

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




        global_step += 1

        save_model(model, MODEL_NAME)

def loss(total_returns, log_prob_actions, entropies, state_values, critic_beta, entropy_beta):
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

    loss = (torch.stack(actor_losses).sum() - entropy_beta * torch.stack(entropies).sum()) + critic_beta * torch.stack(critic_losses).sum()
    # print("loss:", loss)

    return loss

def save_model(model, name):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name))

def load_model(model, name):
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name)))

def generate_a2c_data(agent, batch_size, gamma, env):
    global num_wins
    global num_played
    global average_rewards_per_batch
    global sample_game
    global num_wins_batch

    states = []
    log_prob_actions = []
    state_values = []
    total_returns = []
    entropies = []
    next_states = []

    # curr_state = env.reset()
    done = False
    total_rewards = 0

    record_data = True
    num_wins_batch = 0

    for _ in range(batch_size):
        state = env.reset()

        ep_reward = 0
        rewards = []

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        if(record_data):
            del sample_game[:]

        for t in range(6):

            # select action from policy
            action, log_prob_action, entropy, state_value = agent(torch.Tensor(state))
            if(record_data):
                sample_game.append(action)
            # print("action:", action)
            # take the action
            # env.render()
            state, reward, done, __ = env.step(action)

            rewards.append(reward)
            log_prob_actions.append(log_prob_action)
            state_values.append(state_value)
            entropies.append(entropy)

            # model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        if(record_data):
            sample_game.append(env.hidden_word)
        record_data = False

        # TODO: Change to win_rate per batch
        total_rewards += ep_reward
        num_played += 1
        if(action == list(env.hidden_word)):
            num_wins += 1
            num_wins_batch += 1

        returns = []
        R = 0
        for r in rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

        total_returns.extend(returns)

    average_rewards_per_batch = total_rewards / batch_size
    # print(average_rewards_per_batch)
    return total_returns, log_prob_actions, entropies, state_values

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_episodes', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=float, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-g', '--gamma', type=float, default=.9)
    parser.add_argument('-m', '--embedding_size', type=int, default=32)

    parser.add_argument('--critic_beta', type=float, default=1)
    parser.add_argument('--entropy_beta', type=float, default=.05)

    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
