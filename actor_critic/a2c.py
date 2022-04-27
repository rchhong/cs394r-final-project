import torch as torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .embeddings import StateEmbeddingLayer, ActionEmbeddingLayer

class ActorCriticNet(torch.nn.Module):
    def __init__(self, state_size, word_list, embedding_size, hidden_size=256, n_hidden=1):
        super().__init__()
        word_width = 26*5
        self.n_emb = embedding_size

        layers = [
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, self.n_emb))

        self.f_state = nn.Sequential(*layers)

        self.actor_head = nn.Linear(self.n_emb, self.n_emb)
        self.critic_head = nn.Linear(self.n_emb, 1)

        word_array = np.zeros((len(word_list), word_width))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                # print (j, (ord(c) - ord('a')))
                word_array[i, j*26 + (ord(c) - ord('a'))] = 1
        self.words = torch.Tensor(word_array)

        # W x word_width -> W x emb
        self.f_word = nn.Sequential(
            nn.Linear(word_width, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_emb),
        )

    def forward(self, x):
        fs = self.f_state(x.float())
        fw = self.f_word(
            self.words.to("cpu"),
        ).transpose(0, 1)

        # print(self.actor_head(fs).shape)
        # print(fw.shape)
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(fs), fw,
                            dims=((0,), (0,))),
            dim=-1)
        c = self.critic_head(fs)

        return a, c