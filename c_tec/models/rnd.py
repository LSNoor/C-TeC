import math

import torch
import torch.nn as nn
import torch.optim as optim


class ObsRunningMeanStd(nn.Module):
    """Running mean/std for observation normalization (torch version).

    Uses Welford's online algorithm.  Registered as an ``nn.Module`` so
    that its buffers are automatically included in ``state_dict()`` and
    follow ``.to(device)`` calls.

    Following the RND paper (Burda et al., 2018, Section 2.4),
    observations are whitened (subtract mean, divide by std) and then
    clipped to [-clip, clip] before being fed to the target and
    predictor networks.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Update running statistics with a new batch of observations."""
        batch = batch.double()
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean.double()
        total_count = self.count + batch_count

        new_mean = self.mean.double() + delta * batch_count / total_count
        m_a = self.var.double() * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean.copy_(new_mean.float())
        self.var.copy_((m2 / total_count).float())
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        """Whiten by running mean/std and clip to ``[-clip, clip]``."""
        return ((x - self.mean) / (self.var.sqrt() + self.epsilon)).clamp(-clip, clip)


class RNDTarget(nn.Module):
    """Fixed randomly-initialized target network f(s).

    Maps states to a fixed embedding. Never trained — its random weights
    define the target distribution that the predictor must learn to match.
    Prediction error on unseen states serves as the intrinsic novelty signal.

    A shallow (one hidden layer) architecture is used deliberately: the
    target only needs to define a stable, non-trivial embedding space.
    """

    def __init__(self, state_dim: int, hidden_dim: int, repr_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )
        # Orthogonal initialization for stable, well-conditioned random features
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Freeze all parameters — the target is never updated
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class RNDPredictor(nn.Module):
    """Trainable predictor network f_hat(s).

    Learns to predict the fixed target network's output. The prediction
    error ||f_hat(s) - f(s)||^2 is high for novel states and low for
    familiar ones, providing a natural novelty signal that decays as the
    agent revisits states.

    Slightly deeper than the target (two hidden layers vs. one) following
    the original RND paper's recommendation that the predictor should be
    more expressive than the target.
    """

    def __init__(self, state_dim: int, hidden_dim: int, repr_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(hidden_dim, repr_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class RNDModel(nn.Module):
    """Random Network Distillation model with observation normalization.

    Maintains a running mean/std of observations (``obs_rms``) and
    whitens + clips inputs to [-5, 5] before feeding them to the target
    and predictor, following Burda et al. 2018 §2.4.  This prevents the
    networks from being dominated by high-magnitude state components
    (e.g. ``x, y ∈ [0, 18]`` vs direction one-hot ``∈ {0, 1}``).

    ``obs_rms`` is an ``nn.Module`` whose buffers are part of the
    ``state_dict``, so they are saved/loaded with checkpoints and move
    with ``.to(device)`` automatically.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        repr_dim: int,
        lr: float,
        device: torch.device,
    ):
        super().__init__()
        self.obs_rms = ObsRunningMeanStd((state_dim,))
        # Registered as submodules automatically by nn.Module
        self.target = RNDTarget(state_dim, hidden_dim, repr_dim)
        self.predictor = RNDPredictor(state_dim, hidden_dim, repr_dim)
        self.device = device
        self.to(device)
        # Optimizer is created AFTER .to(device) so all parameter tensors
        # (predictor weights) are already on the target device.
        # Only predictor parameters are optimized; target is frozen.
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    @torch.no_grad()
    def compute_reward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute per-state intrinsic rewards (prediction errors).

        Observations are normalized using the *current* running statistics
        (no update here — statistics are only updated during training via
        ``update_obs_stats``).

        Runs under torch.no_grad() — inference only, no gradient tracking.

        Args:
            states: (B, state_dim) float tensor already on self.device.

        Returns:
            rewards: (B,) tensor of ||f_hat(s) - f(s)||^2 values.
        """
        self.predictor.eval()
        normed = self.obs_rms.normalize(states)
        target_features = self.target(normed)
        predicted_features = self.predictor(normed)
        return ((predicted_features - target_features) ** 2).sum(dim=-1)

    @torch.no_grad()
    def update_obs_stats(self, states: torch.Tensor) -> None:
        """Update the observation running mean/std with a batch of states.

        Call this once per episode with all collected states *before*
        the mini-batch predictor training loop.  Keeping statistics
        updates separate from gradient steps ensures that every
        mini-batch within an epoch sees the same normalization.

        Args:
            states: (B, state_dim) float tensor already on self.device.
        """
        self.obs_rms.update(states)

    def update(self, states: torch.Tensor) -> float:
        """Train the predictor on a (mini-)batch of already-collected states.

        Observations are normalized using the current running statistics
        (which should have been updated beforehand via
        ``update_obs_stats``).

        The target output is detached to make the gradient flow explicit:
        only the predictor is updated, never the target.

        Args:
            states: (B, state_dim) float tensor already on self.device.

        Returns:
            loss: scalar predictor MSE loss.
        """
        normed = self.obs_rms.normalize(states)
        # .detach() is redundant (target has requires_grad=False) but makes
        # the intent explicit: gradients flow only through the predictor.
        target_features = self.target(normed).detach()
        predicted_features = self.predictor(normed)
        loss = ((predicted_features - target_features) ** 2).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
