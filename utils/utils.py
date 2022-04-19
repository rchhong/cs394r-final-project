import torch as torch
import numpy as np

from const import NUM_CHARACTERS_ALPHABET

# For NN and embeddings
def convert_to_one_hot(word_list):
    ret = np.zeros((len(word_list), len(word_list[0]) * NUM_CHARACTERS_ALPHABET))

    for i, word in enumerate(word_list):
        for j, char in enumerate(word):
            index = ord(char) - ord('a')
            ret[i, NUM_CHARACTERS_ALPHABET * j + index] = 1

    return torch.tensor(ret)

# For agents
def convert_to_char_index(word_list):
    ret = np.zeroes((len(word_list), len(word_list[0])))

    for i, word in enumerate(word_list):
        for j, char in enumerate(word):
            val = ord(char) - ord('a')

            ret[i, j] = val

    return ret

def load_word_list(path):
    word_list = []

    with open(path, 'r') as f:
        for word in f:
            word_list.append(word)

    return word_list