from copy import deepcopy
import gym
import gym_wordle
import torch
from gym_wordle.exceptions import InvalidWordException
from actor_critic.a2c import ActorCriticNet
from actor_critic.agents.prob_agent import ProbabilisticAgent
from utils import load_model
from utils.const import STATE_SIZE
from utils.utils import load_word_list, convert_encoded_array_to_human_readable
from actor_critic.agents import GreedyAgent


def run_trial(a2c_agent, verbose):
    env = gym.make('Wordle-v0')



    NUM_ITER = 100

    games = []
    num_wins = 0
    for i in range(NUM_ITER):
        obs = env.reset()
        done = False

        actions = []
        while not done:
            while True:
                try:
                    action, log_prob_action, entropy, state_value = a2c_agent(torch.Tensor(obs))
                    actions.append(convert_encoded_array_to_human_readable(action))

                    obs, reward, done, _ = env.step(action)

                    break
                except InvalidWordException:
                    pass
        if(action == list(env.hidden_word)):
            num_wins += 1

        games.append((deepcopy(actions), convert_encoded_array_to_human_readable(env.hidden_word)))

    print(f"WIN RATE: {num_wins / NUM_ITER}")
    if(verbose):
        for game in games:
            print(str(game))

    return num_wins / NUM_ITER, games



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--words_dir')
    parser.add_argument('-m', '--embedding_size', type=int, default=32)

    args = parser.parse_args()

    word_list = load_word_list(args.words_dir)

    model = ActorCriticNet(STATE_SIZE, word_list, args.embedding_size)
    load_model(model, "a2c")

    agent = GreedyAgent(model, word_list)

    print(run_trial(agent, True))
