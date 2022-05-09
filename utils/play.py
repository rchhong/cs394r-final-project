from copy import deepcopy
import gym
import gym_wordle
import torch
from gym_wordle.exceptions import InvalidWordException
from agents.prob_agent import ProbabilisticAgent
from reinforce.reinforce import REINFORCEWithBaseline
from utils import load_model
from utils.const import STATE_SIZE
from utils.utils import load_word_list, convert_encoded_array_to_human_readable
from agents import GreedyAgent
import numpy as np

def run_trial(a2c_agent, verbose, word_list=[]):
    done = False
    
    action_replay = []
    state_space = np.zeros((26, 5, 3), dtype=np.int0)
    state_space[:,:,1] = 1
    board = np.negative(
            np.ones(shape=(6, 5), dtype=int))
    board_row_idx = 0

    obs = np.zeros((26 + 5 * 26 * 3 + 6))
    alphabet = np.zeros(26)
    for i in range (6):
        actions = a2c_agent(torch.Tensor(obs))

        print (actions)
        ind = int(input())

        action = [ord(x) - 97 for x in actions[ind]]

        real = [int (x) for x in input()]
        action_replay.append (action)

        for idx, (char, value) in enumerate(zip (action, real)):

            if value == 2:
                encoding = 2

                #Set everything else to 0 except correct character
                state_space[:, idx, 0] = 1
                state_space[char, idx, :] = 0
                state_space[char, idx, 2] = 1
            else:
                encoding = 0

            board[board_row_idx, idx] = encoding
            alphabet[char] = 1

        for idx, (char, value) in enumerate(zip (action, real)):
            if value == 1:
                encoding = 1
                # state_space[char, idx] = 1
            else:
                encoding = 0

            if (board[board_row_idx, idx] != 2):
                board[board_row_idx, idx] = encoding
                res = np.where(state_space[char, :, 1] == 1, encoding, np.argmax (state_space[char], axis=1))
                b = np.zeros((5, 3))
                b[np.arange(5), res] = 1
                state_space[char, :] = b

        # actions.append(convert_encoded_array_to_human_readable(action))

        return_state = np.zeros((26 + 5 * 26 * 3 + 6))
        return_state[0:26] = alphabet
        return_state[26:-6] = state_space.flatten()
        return_state[6 * 26 + 5 - i] = 1
        obs = return_state
        for i in range (26):
            # print (self.state_space[i])
            print (chr(i + 97), np.argmax (state_space[i], axis=1))
        board_row_idx += 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--words_dir')
    parser.add_argument('-m', '--embedding_size', type=int, default=64)

    args = parser.parse_args()

    word_list = load_word_list(args.words_dir)

    model = REINFORCEWithBaseline(STATE_SIZE, word_list, args.embedding_size)
    load_model(model, "reinforce")

    agent = GreedyAgent(model, word_list)

    print(run_trial(agent, True, word_list=word_list))
