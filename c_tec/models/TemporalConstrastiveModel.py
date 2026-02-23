from typing import Literal

import torch
import torch.nn as nn

from .encoders import FutureStateEncoder, StateActionEncoder


class TemporalContrastiveModel(nn.Module):
    """
    Pairs φ and ψ with a learnable temperature and critic function.

    The separable (φ, ψ) architecture is essential — the paper's
    ablation (Appendix C.4, Fig. 11) shows a monolithic critic
    f(s, a, s_f) fails to produce useful exploration signal.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        repr_dim: int = 32,
        hidden_dim: int = 128,
        norm: Literal["l1", "l2"] = "l2",
    ):
        super().__init__()
        self.phi = StateActionEncoder(state_dim, action_dim, repr_dim, hidden_dim)
        self.psi = FutureStateEncoder(state_dim, repr_dim, hidden_dim)
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # learnable temperature
        self.norm = norm

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def critic(self, phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Temporal similarity score (Eq. 3-4):
            C_θ = -||φ(s,a) - ψ(s_f)||_p

        Optimal critic approximates log p_T(s_f | s,a) / p_T(s_f).
        """
        p = 2 if self.norm == "l2" else 1
        return -torch.norm(phi - psi, p=p, dim=-1)

    def compute_logits(self, states, actions, future_states) -> torch.Tensor:
        """Pairwise critic matrix. logits[i,j] = C_θ((s_i,a_i), s_f^(j))."""
        phi = self.phi(states, actions)
        psi = self.psi(future_states)
        p = 2 if self.norm == "l2" else 1
        return -torch.cdist(phi, psi, p=p) / self.tau