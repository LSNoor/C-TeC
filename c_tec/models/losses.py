import torch
from torch import nn
import torch.nn.functional as F


class CRLLoss(nn.Module):
    """
    Contrastive Representation Learning loss (Eq. 2 from the paper).
    """

    def __init__(self, logsumexp_penalty: float):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # learned temperature
        self.logsumexp_penalty = logsumexp_penalty

    def forward(self, critic_encoder, states, actions, future_states):
        """
        Args:
            critic_encoder: CriticEncoder — computes pairwise distances.
            states:         (K, s_dim)
            actions:        (K, a_dim)
            future_states:  (K, s_dim) — future_states[i] is the positive for i,
                            all j ≠ i serve as negatives.
        Returns:
            loss: scalar CRL loss (Eq. 2 from the paper)
        """
        batch_size = states.size(0)
        tau = self.temperature.clamp(min=0.01)

        # Broadcast to all (i, j) pairs → (K*K, dim)
        states_exp = (
            states.unsqueeze(1)
            .expand(-1, batch_size, -1)
            .reshape(batch_size * batch_size, -1)
        )
        actions_exp = (
            actions.unsqueeze(1)
            .expand(-1, batch_size, -1)
            .reshape(batch_size * batch_size, -1)
        )
        future_exp = (
            future_states.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * batch_size, -1)
        )

        # Distances → negate → similarity matrix (K, K); diagonal = positives
        sim = (
            critic_encoder(states_exp, actions_exp, future_exp).reshape(
                batch_size, batch_size
            )
            / tau
        )
        labels = torch.arange(batch_size, device=sim.device)

        infonce_term = F.cross_entropy(sim, labels)
        logsumexp_vals = torch.logsumexp(sim, dim=1)  # reuse the (K,K) sim matrix
        reg_term = self.logsumexp_penalty * (logsumexp_vals**2).mean()

        return (infonce_term + reg_term).mean()
