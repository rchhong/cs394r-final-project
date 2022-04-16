import torch as torch
import numpy as np

from embeddings import StateEmbeddingLayer, ActionEmbeddingLayer

class ActorCriticNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass