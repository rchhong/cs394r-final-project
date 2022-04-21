# Reference: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html
import torch
from torch.utils.data import IterableDataset, DataLoader
from replay_buffer import ReplayBuffer

class ExperienceDataset(IterableDataset):
    def __init__(self, buffer, sample_size = 128):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_states, returns  = self.buffer.sample(self.sample_size)

        for i in range(len(states)):
            yield states[i], actions[i], next_states[i], returns[i]

def generate_dataset(capacity, num_samples, batch_size):
    buffer = ReplayBuffer(capacity)
    dataset = ExperienceDataset(buffer, num_samples)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )
    return dataloader