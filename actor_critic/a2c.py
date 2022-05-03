import torch as torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from embeddings import StateEmbeddingLayer, ActionEmbeddingLayer

class ActorCriticNet(torch.nn.Module):
    def __init__(self, state_size, word_list, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.state_embedding = StateEmbeddingLayer(state_size, embedding_size)
        self.action_embedding = ActionEmbeddingLayer(embedding_size, word_list)

        self.actor_head = nn.Linear(self.embedding_size, self.embedding_size)
        self.critic_head = nn.Linear(self.embedding_size, 1)

    def forward(self, x):
        state_embedded = self.state_embedding(x)
        actions_embedded = self.action_embedding(x)
        # print(actions_embedded)


        # print(self.actor_head(fs).shape)
        # print(fw.shape)
        similiarity_metric = torch.tensordot(self.actor_head(state_embedded), actions_embedded, dims=((0,), (1,)))
        a = torch.log_softmax(similiarity_metric, dim=-1)
        c = self.critic_head(state_embedded)

        return a, c