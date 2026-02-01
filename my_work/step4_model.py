import torch.nn as nn


class ShardedMLP(nn.Module):
    def __init__(self, hidden_dim, total_layers, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        # 1. Calculate how many layers THIS GPU is responsible for
        assert (
            total_layers % world_size == 0
        ), f"total_layers ({total_layers}) must be divisible by world_size ({world_size})"
        layers_per_rank = total_layers // world_size
        # 2. Build the local stack of layers
        layers = []
        for _ in range(0, layers_per_rank):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        if rank == world_size - 1:
            layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # Run the local chunk of the network
        x = self.net(x)

        # Only the last GPU calculates loss
        if self.rank == self.world_size - 1:
            assert targets is not None
            loss = self.loss_fn(x, targets)
            return loss
        # Everyone else just returns the hidden states (activations)
        return x
