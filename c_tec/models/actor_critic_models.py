import math

from torch import nn
from torch.distributions import Categorical


class ActorModel(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Orthogonal init: √2 gain for hidden (Tanh), 0.01 for output
        for layer in (self.net[0], self.net[2]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[4].weight, gain=0.01)  # near-uniform start
        nn.init.zeros_(self.net[4].bias)

    def forward(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)


class CriticModel(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net: nn.Module = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Orthogonal init: √2 gain for hidden (Tanh), 1.0 for value output
        for layer in (self.net[0], self.net[2]):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[4].weight, gain=1.0)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, state):
        value = self.net(state)  # (batch, 1)
        return value
