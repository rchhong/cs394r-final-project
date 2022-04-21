import gym
import torch.utils.tensorboard as tb
from os import path
import torch

from actor_critic.a2c import ActorCriticNet
from actor_critic.agents.prob_agent import ProbabilisticAgent
from utils.generate_data import generate_a2c_data
from utils.utils import load_model, load_word_list, save_model

env = gym.make('Wordle-v0')
MODEL_NAME = "a2c"


# TODO: Training loop
def train(args):
    word_list = load_word_list(args.words_dir)
    model = ActorCriticNet()
    train_agent = ProbabilisticAgent(model, word_list)

    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        load_model(model, MODEL_NAME)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    global_step = 0

    for epoch in range(args.num_epoch):
        model.train()

        train_data = generate_a2c_data(args.batch_size, args.gamma, train_agent, model)

        loss_val = loss(states, actions, returns)

        if(global_step % 100 == 0):
            if(train_logger):
                pass
            # SAVE MODEL EVERY 100 STEPS
            save_model(model)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        global_step += 1

        save_model(model, MODEL_NAME)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-g', '--gamma', type=float, default=.99)
    parser.add_argument('-b', '--batch_size', type=float, default=48)
    parser.add_argument('-c', '--continue_training', action='store_true')


    args = parser.parse_args()
    train(args)
