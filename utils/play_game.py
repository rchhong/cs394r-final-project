import gym
import gym_wordle
import torch
from gym_wordle.exceptions import InvalidWordException
from actor_critic.a2c import ActorCriticNet
from utils import load_model
from utils.const import STATE_SIZE
from utils.utils import load_word_list, convert_encoded_array_to_human_readable
from actor_critic.agents import GreedyAgent


def play_game_a2c(a2c_agent, visualize):
    env = gym.make('Wordle-v0')

    obs = env.reset()
    done = False

    actions = []
    while not done:
        while True:
            try:
                pred = a2c_agent(torch.Tensor(obs.reshape(1,-1)))
                
                # actions.append(convert_encoded_array_to_human_readable(env_action))
                actions.append (pred)

                obs, reward, done, _ = env.step(pred)
                # print (reward)
                break
            except InvalidWordException:
                pass

    if visualize:
        env.render()

    return actions



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

    print(play_game_a2c(agent, True))
