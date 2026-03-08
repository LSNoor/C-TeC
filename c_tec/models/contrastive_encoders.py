import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from .losses import CRLLoss


class CriticEncoder(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sa_hidden_dim: int,
        sf_hidden_dim: int,
        repr_dim: int,
        norm_type: Literal["l1", "l2"],
        lr: float,
        logsumexp_penalty: float,
        device: torch.device = torch.device("cpu"),
    ):

        super().__init__()

        self.phi: StateActionEncoder = StateActionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            repr_dim=repr_dim,
            hidden_dim=sa_hidden_dim,
        )
        self.psi: FutureStateEncoder = FutureStateEncoder(
            state_dim=state_dim, repr_dim=repr_dim, hidden_dim=sf_hidden_dim
        )
        self.ord = 2 if norm_type == "l2" else 1
        # CRLLoss must be registered as a submodule BEFORE .to(device)
        # so its learnable temperature parameter is also moved to device.
        # The encoder is NOT stored inside CRLLoss to avoid a circular
        # submodule reference; it is passed at forward() time instead.
        self.loss = CRLLoss(logsumexp_penalty=logsumexp_penalty)
        self.device = device
        self.to(device)
        # Optimizer is created AFTER .to(device) so all parameter tensors
        # (including CRLLoss.temperature) are already on the target device.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, future_states):

        phi = self.phi(states, actions)
        psi = self.psi(future_states)
        return -torch.linalg.norm(phi - psi, dim=-1, ord=self.ord)

    def update(self, states, actions, future_states):

        loss = self.loss(self, states, actions, future_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class StateActionEncoder(nn.Module):
    """Encoder φ_θ(s, a) mapping state-action pairs to normalized representations."""

    def __init__(self, state_dim: int, action_dim: int, repr_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([states, actions], dim=-1)
        return F.normalize(self.net(sa), dim=-1)


class FutureStateEncoder(nn.Module):
    """Encoder ψ_θ(s_f) mapping future states to normalized representations."""

    def __init__(self, state_dim: int, repr_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(states), dim=-1)
