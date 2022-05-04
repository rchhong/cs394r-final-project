import torch as torch
import numpy as np
from utils import convert_to_one_hot, NUM_CHARACTERS_ALPHABET


class ActionEmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_size, word_list, hidden_size = 64):
        super().__init__()

        self.action_size = len(word_list[0]) * NUM_CHARACTERS_ALPHABET
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.word_list = word_list
        self.one_hot_words = convert_to_one_hot(word_list).float()

        self.net = torch.nn.Sequential(*[
            torch.nn.Linear(self.action_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.embedding_size),
        ])

    def forward(self, x):
        return self.net(self.one_hot_words).to(x.device)