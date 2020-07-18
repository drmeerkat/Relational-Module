import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MultiheadAttention import RelationalAttention

class GNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(GNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(64, 510, 4, stride=2)), nn.ReLU()
        )
        self.gnn = RelationalAttention(hidden_size, 256, 256, 4, maxout=True)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = self.main(inputs / 255.0)

        x = self.gnn(x)

        return self.critic_linear(x)