import torch as torch
import numpy as np

from embeddings import StateEmbeddingLayer, ActionEmbeddingLayer

class ActorCriticNet(torch.nn.Module):
    def __init__(self, state_size, word_list, embedding_size):
        super().__init__()

        self.state_size = state_size
        self.word_list = word_list
        self.embedding_size = embedding_size

        self.state_embedding = StateEmbeddingLayer(self.state_size, self.embedding_size)

        # Actions are possible words
        self.action_embedding = ActionEmbeddingLayer(self.embedding_size, word_list)

        self.actor_net = torch.nn.Sequential(*[
            # torch.nn.Linear(self.embedding_size, self.embedding_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        ])

        self.critic_net = torch.nn.Sequential(*[
            # torch.nn.Linear(self.embedding_size, self.embedding_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, 1),
        ])



    def forward(self, x):
        state_embedded = self.state_embedding(x)
        action_embedded = self.action_embedding(x)

        pred_action = self.actor_net(state_embedded)
        # Select actions that are most similar by taking the dot product
        similarity_index = torch.tensordot(pred_action, action_embedded, dims = ([1], [1]))

        action_log_probs = torch.log_softmax(similarity_index, dim = -1)
        v_s = self.critic_net(state_embedded)

        return action_log_probs, v_s