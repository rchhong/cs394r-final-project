from copy import deepcopy
import gym
import gym_wordle
import torch
from gym_wordle.exceptions import InvalidWordException
from actor_critic.a2c import ActorCriticNet
from agents.prob_agent import ProbabilisticAgent
from agents.random_agent import RandomAgent
from utils import load_model
from utils.const import STATE_SIZE
from utils.utils import load_word_list, convert_encoded_array_to_human_readable
from agents import GreedyAgent
from tqdm import tqdm


def run_trial(a2c_agent, verbose, word_list):
    env = gym.make('Wordle-v0')
    games = []
    num_turns_distribution = [0 for _ in range(6)]
    num_wins = 0
    total_steps = 0
    for index in tqdm(range(len(word_list))):
        i = word_list[index]
        obs = env.reset()
        env.hidden_word = [ord(x) - 97 for x in i]
        done = False
        step = 0


        actions = []
        while not done:
            while True:
                try:
                    action, log_prob_action, entropy, state_value = a2c_agent(torch.Tensor(obs))
                    actions.append(convert_encoded_array_to_human_readable(action))

                    obs, reward, done, _ = env.step(action)
                    step += 1

                    break
                except InvalidWordException:
                    pass

        if(action == list(env.hidden_word)):
            num_wins += 1
            num_turns_distribution[step - 1] += 1
            total_steps += step

        games.append((deepcopy(actions), convert_encoded_array_to_human_readable(env.hidden_word)))

    # print(f"WIN RATE: {num_wins / NUM_ITER}")
    if(verbose):
        for game in games:
            print(str(game))

    return num_wins / len(word_list), num_turns_distribution, total_steps / num_wins if num_wins > 0 else 0, games


if __name__ == '__main__':
    WORDS_DIR = "./gym-wordle/gym_wordle/data/5_words.txt"
    SOLUTIONS_DIR = "./plots/entropy/possible_words.txt"
    EMBEDDING_SIZE = 64

    word_list = load_word_list(WORDS_DIR)
    allowed_solutions = load_word_list(SOLUTIONS_DIR)

    model = ActorCriticNet(STATE_SIZE, word_list, EMBEDDING_SIZE)
    load_model(model, "reinforceFINAL")
    optimal_agent = ProbabilisticAgent(model, word_list)
    win_rate_optimal, turn_distribution_optimal, average_amount_of_turns_optimal, games_optimal = run_trial(optimal_agent, False, allowed_solutions)
    print("OPTIMAL:")
    print("WIN RATE:", win_rate_optimal)
    print("TURN DISTRIBUTION:", str(turn_distribution_optimal))
    print("AVERAGE AMOUNT OF TURNS TAKEN:", average_amount_of_turns_optimal)

    random_agent = RandomAgent(word_list)
    win_rate_random, turn_distribution_random, average_amount_of_turns_random, games_random = run_trial(random_agent, False, allowed_solutions)
    print("RANDOM:")
    print("WIN RATE:", win_rate_random)
    print("TURN DISTRIBUTION:", str(turn_distribution_random))
    print("AVERAGE AMOUNT OF TURNS TAKEN:", average_amount_of_turns_random)