from click import pass_context
import gym
from pkg_resources import require
import torch.utils.tensorboard as tb
from os import path
import torch

from actor_critic.a2c import ActorCriticNet
from actor_critic.agents.greedy_agent import GreedyAgent
from actor_critic.agents.prob_agent import ProbabilisticAgent
from utils.utils import load_word_list

env = gym.make('Wordle-v0')

def generate_batch_data(batch_size, gamma, agent, net, env, device):
    assert(batch_size % 6 == 0)

    games_data = []

    num_wins = 0
    num_losses = 0
    avg_reward_per_game = 0

    for _ in range(batch_size / 6):

        states = []
        actions = []
        rewards = []
        next_states = []
        # Generate episodes
        curr_state = env.reset()
        done = False
        while not done:
            action = agent(curr_state)
            next_state, reward, done, _ = env.step()

            states.append(curr_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            curr_state = next_state

        returns = []
        _, bootstrapped_value = net(torch.Tensor([curr_state], device = device))

        ret = bootstrapped_value
        requires_bootstrap = False
        for reward in rewards[::-1]:
            if requires_bootstrap:
                ret = reward + gamma * ret
                returns.append(ret)
            else:
                returns.append(reward)
            requires_bootstrap = True

        game_data = list(zip(states, actions, returns[::-1], next_states))

        games_data.append(games_data)

    return games_data

def save_model(model):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % "imitation"))



def generate_dataloader(batched_data):
    pass

# TODO: Custom loss, simply weighted average of the critic and actor minus entropy for some reasonf


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
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'a2c.th')))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    global_step = 0

    for epoch in range(args.num_epoch):
        model.train()

        train_data = generate_batch_data(args.batch_size, args.gamma, train_agent, model)

        # TODO: FIX THIS
        loss_val = loss(states, actions, returns)

        if(global_step % 100 == 0):
            if(train_logger):
                # TODO: LOG LOSS, PRINT OUT WINRATE, SAMPLE GAMES, AVERAGE REWARD PER GAME, GLOBAL STEP?
                pass
            # SAVE MODEL EVERY 100 STEPS
            save_model(model)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        global_step += 1

        # TODO: Do we want validation
        # model.eval()
        # for img, heatmap, sizes in valid_data:
        #     img, heatmap = img.to(device), heatmap.to(device)

        #     output = model(img)
        #     if(valid_logger):
        #         log(valid_logger, img, heatmap, output, global_step)

        save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--words_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num-epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4)
    parser.add_argument('-g', '--gamma', type=float, default=.99)
    parser.add_argument('-b', '--batch-size', type=float, default=48)
    parser.add_argument('-c', '--continue-training', action='store_true')


    args = parser.parse_args()
    train(args)
