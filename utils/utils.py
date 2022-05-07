import torch as torch
import numpy as np
from os import path

from utils.const import NUM_CHARACTERS_ALPHABET

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
    # print(word_list)
    if(type(word_list) != list):
        ret = []
        for i, char in enumerate(word_list):
            val = ord(char) - ord('a')
            ret.append(val)

        return ret
    else:
        ret = np.zeros((len(word_list), len(word_list[0])), dtype=int)

        for i, word in enumerate(word_list):
            for j, char in enumerate(word):
                val = ord(char) - ord('a')

                ret[i, j] = val
        return ret

def convert_encoded_array_to_human_readable(word):
    return ''.join(chr(c + ord('a')) for c in word)

def load_word_list(path):
    word_list = []

    with open(path, 'r') as f:
        for word in f:
            word_list.append(word.strip())

    return word_list

def save_model(model, name):
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name))

def load_model(model, name):
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '../models', '%s.th' % name), map_location=torch.device('cpu')))

def generate_dataloader(batched_data):
    pass