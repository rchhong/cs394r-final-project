# Reference: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html
from pyparsing import col
import torch
from torch.utils.data import IterableDataset, DataLoader
from utils.replay_buffer import ReplayBuffer

class ExperienceDataset(IterableDataset):
    def __init__(self, buffer, sample_size = 128):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        returns, actions = self.buffer.sample(self.sample_size)
        # print (returns, actions)
        for i in range(len(returns)):
            # print (returns[i], actions[i])
            yield returns[i], actions[i] 

def collate (batch):
    returns_list = []
    actions_list = []

    for (_ret, _action) in batch:
        # print (_ret, _action)
        returns_list.append(_ret)
        actions_list.append(_action)
    return returns_list, actions_list

def generate_dataset(capacity, num_samples, batch_size):
    buffer = ReplayBuffer(capacity)
    dataset = ExperienceDataset(buffer, num_samples)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate
    )

    return buffer, dataloader