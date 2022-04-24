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
from actor_critic.agents.greedy_agent import GreedyAgent
from utils.generate_data import generate_a2c_data
from utils.play_game import play_game_a2c
from utils.utils import load_model, load_word_list, save_model
from utils.experience_dataset import generate_dataset
from utils import STATE_SIZE
import numpy as np
from tqdm import tqdm

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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

        avg_loss = []

        for (states, actions, next_states, returns) in dataset:
            optimizer.zero_grad()
            loss_val = loss(states, actions, returns, model, args.critic_beta, args.entropy_beta)
            avg_loss.append(loss_val.detach().numpy())
            if(train_logger):
                train_logger.add_scalar("loss", loss_val, global_step=global_step)
            
            if(global_step % 100 == 0):
                if(train_logger):
                    # Make the model play a game
                    games = [play_game_a2c(agent, False) for i in range(1)]
                    for game in games:
                        if valid_logger:
                            valid_logger.add_text(tag = "SAMPLE GAMES", text_string = "ACTIONS: " + str(game[0]) + " GOAL: " + str(game[1]), global_step=global_step)
                        
                # SAVE MODEL EVERY 100 STEPS
                save_model(model, MODEL_NAME)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            global_step += 1

        print (sum (avg_loss) / len (avg_loss))

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
        elif epoch % 10 == 0:
            print ("Winrate: {}, GamePlayed: {}, Average_Reward: {}".format (num_wins / num_played, num_played, average_rewards))
            games = [play_game_a2c(agent, False) for i in range(100)]
            print ("Current Win Rate: {}".format(sum ([len (x[0]) != 6 for x in games]) / 100.0))
            print ("Actions: {}, Goal: {}".format (games[0][0], games[0][1]))
        save_model(model, MODEL_NAME)

def loss(states, actions, returns, net, critic_beta, entropy_beta):

    log_probs, values = net(states)
    # No divide by 0
    epsilon = torch.finfo(torch.float32).eps
    # No gradient necessary when normalizing
    with torch.no_grad():
        # ISSUE: values and returns may be in different units, need to make sure they're the same
        advantages = returns - ((values * returns.std()) + returns.mean())

        advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)
        targets = (returns - returns.mean()) / (returns.std() + epsilon)


    # Actor Loss - based on REINFORCE update rule
    # How did we get this action
    action_log_probs = log_probs[np.arange(len(actions), dtype=np.int_), actions.long()]
    actor_loss = (advantages * action_log_probs).mean()

    # Critic Loss - MSE
    critic_loss = critic_beta * torch.square(targets - values).mean()

    # According the original paper on A2C (https://arxiv.org/pdf/1602.01783.pdf) entropy regularlization improves exploration
    probs = log_probs.exp()
    entropy = entropy_beta * (-(probs * log_probs)).sum()

    # print ("ACTOR LOSS: ", actor_loss)
    # print ("LOG PROBS: ", log_probs)

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
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-s', '--sample_size', type=int, default=256)
    parser.add_argument('-p', '--capacity', type=int, default=1000)
    parser.add_argument('-e', '--num_new_transitions', type=int, default=128)


    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
