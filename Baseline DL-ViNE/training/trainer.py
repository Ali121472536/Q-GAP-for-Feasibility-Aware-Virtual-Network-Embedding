import torch
from training.replay_buffer import ReplayBuffer


class Trainer:

    def __init__(self, agent):
        self.agent = agent
        self.buffer = ReplayBuffer()

    def train_step(self):
        pass