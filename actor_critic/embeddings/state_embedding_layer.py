import torch as torch

class StateEmbeddingLayer(torch.nn.Module):
    def __init__(self, state_size, embedding_size, hidden_size = 256):
        super().__init__()

        self.state_size = state_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.net = torch.nn.Sequential(*[
            torch.nn.Linear(self.state_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.embedding_size)
        ])

    def forward(self, x):
        return self.net(x.float())