import torch as torch


class ActionEmbeddingLayer(torch.nn.Module):
    def __init__(self, input_size, embedding_size):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        HIDDEN_SIZE = 256

        layers = [
            torch.nn.Linear(self.input_size, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, self.embedding_size)
        ]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())