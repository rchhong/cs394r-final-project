from collections import deque
from pickle import FALSE
import gym
from matplotlib.pyplot import text
import torch.utils.tensorboard as tb
from os import path
import torch

import gym_wordle
from actor_critic.a2c import ActorCriticNet
from actor_critic.agents.prob_agent import ProbabilisticAgent
from utils.generate_data import generate_a2c_data
from utils.play_game import play_game_a2c
from utils.utils import load_model, load_word_list, save_model
from utils.experience_dataset import generate_dataset
from utils import STATE_SIZE
import numpy as np

MODEL_NAME = "a2c"


def train(args):
    word_list = load_word_list(args.words_dir)
    model = ActorCriticNet(STATE_SIZE, word_list, args.embedding_size)
    agent = ProbabilisticAgent(model, word_list)
    env = gym.make('Wordle-v0')

    train_logger = None
    valid_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        load_model(model, MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Statistics
    num_wins = 0
    num_played = 0
    average_rewards = 0

    replay_buffer, dataset = generate_dataset(args.capacity, args.sample_size, args.batch_size)
    # Generate data to place in the buffer
    inital_data, num_wins, num_played, average_rewards = generate_a2c_data(args.capacity, args.gamma, agent, model, env, device)
    replay_buffer.buffer = deque(inital_data, args.capacity)

    global_step = 0
    if train_logger:
        train_logger.add_scalar("win_rate", num_wins / num_played, global_step=global_step)
        train_logger.add_scalar("num_played", num_played, global_step=global_step)
        train_logger.add_scalar("average_rewards", average_rewards, global_step=global_step)

    for epoch in range(args.num_epoch):
        model.train()

        for (states, actions, next_states, returns) in dataset:
            loss_val = loss(states, actions, returns, model, args.critic_beta, args.entropy_beta)

            if(train_logger):
                    train_logger.add_scalar("loss", loss_val, global_step=global_step)

            if(global_step % 100 == 0):
                if(train_logger):
                    # Make the model play a game
                    games = [play_game_a2c(agent, False) for i in range(1)]
                    for game in games:
                        valid_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(game[0]) + " GOAL: " + str(game[1]), global_step=global_step)
                # SAVE MODEL EVERY 100 STEPS
                save_model(model, MODEL_NAME)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        new_data, new_num_wins, new_num_played, new_average_rewards = generate_a2c_data(args.num_new_transitions, args.gamma, agent, model, env, device)
        for experience in new_data:
            replay_buffer.append(experience)

        average_rewards = (average_rewards * num_played  + new_average_rewards * new_num_played) / (num_played + new_num_played)
        num_wins += new_num_wins
        num_played += new_num_played


        if train_logger:
            train_logger.add_scalar("win_rate", num_wins / num_played, global_step=global_step)
            train_logger.add_scalar("num_played", num_played, global_step=global_step)
            train_logger.add_scalar("average_rewards", average_rewards, global_step=global_step)


        save_model(model, MODEL_NAME)

def loss(states, actions, returns, net, critic_beta, entropy_beta):

    log_probs, values = net(states)
    # No divide by 0
    epsilon = torch.finfo(torch.float32).eps
    # No gradient necessary when normalizing
    with torch.no_grad():
        # ISSUE: values and returns may be in different units, need to make sure they're the same
        advantages = returns - ((values * (returns.std() + epsilon)) - returns.mean())

        advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)
        targets = (returns - returns.mean()) / (returns.std() + epsilon)


    # Actor Loss - based on REINFORCE update rule
    # How did we get this action
    action_log_probs = log_probs[np.arange(len(actions)), actions]
    actor_loss = (advantages * action_log_probs).mean()

    # Critic Loss - MSE
    critic_loss = critic_beta * torch.square(targets - values).mean()

    # According the original paper on A2C (https://arxiv.org/pdf/1602.01783.pdf) entropy regularlization improves exploration
    probs = log_probs.exp()
    entropy = entropy_beta * (-(probs * log_probs)).sum()

    return actor_loss + critic_loss - entropy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-g', '--gamma', type=float, default=.99)
    parser.add_argument('-m', '--embedding_size', type=int, default=32)

    parser.add_argument('--critic_beta', type=float, default=.5)
    parser.add_argument('--entropy_beta', type=float, default=.01)

    # As dumb as this looks, make sure that all of these are the same for now
    parser.add_argument('-b', '--batch_size', type=float, default=32)
    parser.add_argument('-s', '--sample_size', type=float, default=256)
    parser.add_argument('-p', '--capacity', type=int, default=1000)
    parser.add_argument('-e', '--num_new_transitions', type=int, default=128)


    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
