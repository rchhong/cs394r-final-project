import torch as torch
import numpy as np
from torch import nn

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
        
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(fs), fw,
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(fs)
        return a, c
    #     self.state_size = state_size
    #     self.word_list = word_list
    #     self.embedding_size = embedding_size

    #     self.state_embedding = StateEmbeddingLayer(self.state_size, self.embedding_size)

    #     # Actions are possible words
    #     self.action_embedding = ActionEmbeddingLayer(self.embedding_size, word_list)

    #     self.actor_net = torch.nn.Sequential(*[
    #         # torch.nn.Linear(self.embedding_size, self.embedding_size),
    #         # torch.nn.ReLU(),
    #         torch.nn.Linear(self.embedding_size, self.embedding_size),
    #     ])

    #     self.critic_net = torch.nn.Sequential(*[
    #         # torch.nn.Linear(self.embedding_size, self.embedding_size),
    #         # torch.nn.ReLU(),
    #         torch.nn.Linear(self.embedding_size, 1),
    #     ])



    # def forward(self, x):
    #     state_embedded = self.state_embedding(x)
    #     action_embedded = self.action_embedding(x)

    #     pred_action = self.actor_net(state_embedded)
    #     # Select actions that are most similar by taking the dot product
    #     similarity_index = torch.tensordot(pred_action, action_embedded, dims = ((0,), (1,)))

    #     action_log_probs = torch.log_softmax(similarity_index, dim = -1)
    #     v_s = self.critic_net(state_embedded)
    #     return action_log_probs, v_s