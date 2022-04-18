import torch as torch
import numpy as np

NUM_CHARACTERS_ALPHABET = 26

def convert_to_one_hot(word_list):
    ret = np.zeros((len(word_list), len(word_list[0]) * NUM_CHARACTERS_ALPHABET))

    for i, word in enumerate(word_list):
        for j, char in enumerate(word):
            index = ord(char) - ord('a')
            ret[i, NUM_CHARACTERS_ALPHABET * j + index] = 1

    return torch.tensor(ret)


class ActionEmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_size, word_list, hidden_size = 64):
        super().__init__()

        self.action_size = len(word_list[0]) * NUM_CHARACTERS_ALPHABET
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.word_list = word_list
        self.one_hot_words = convert_to_one_hot(word_list)

        self.net = torch.nn.Sequential(*[
            torch.nn.Linear(self.action_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.embedding_size)
        ])

    def forward(self, x):
        return self.net(self.one_hot_words.float()).to(x.get_device())