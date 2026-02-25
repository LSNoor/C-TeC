"""
Trajectory buffer storing complete episodes.

Needed later for C-TeC's geometric future state sampling.
For the random baseline, we simply store trajectories to verify
the buffer works correctly before plugging in contrastive learning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


class RunningMeanStd:
    """Welford's online algorithm for running mean / variance.

    Used to normalize intrinsic rewards across episodes so that
    PPO's value function sees a stable reward scale.

    Only the *standard deviation* is used for normalization (no mean
    subtraction) so that the rewards stay positive and the relative
    ordering is preserved.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon  # avoid division-by-zero on first call

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a new batch of values."""
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Divide by running std (no mean subtraction)."""
        return x / (self.std + epsilon)


@dataclass
class Trajectory:
    def __init__(self):
        self.current_idx: int = 0

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.cell_covered: list[int] = []
        self.cell_covered_pct: list[float] = []

    def __len__(self) -> int:
        return len(self.states)

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        cell_covered: int,
        cell_covered_pct: float,
    ):
        self.current_idx += 1

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.cell_covered.append(cell_covered)
        self.cell_covered_pct.append(cell_covered_pct)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (Schulman et al., 2016, Eq. 11):

            δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
            Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

        Returns advantages (normalized) and lambda-returns for the value loss.

        Args:
            last_value: V(s_T), set to 0.0 for terminal episodes,
                        or the critic's estimate for truncated episodes.

        Returns:
            advantages : [T] normalized advantage estimates
            returns    : [T] lambda-returns  (Â_t + V(s_t))
        """
        T = len(self.states)
        advantages = np.zeros(T, dtype=np.float32)

        # Append bootstrap value for the lookahead at T
        bootstrap_values = self.values + [last_value]

        gae = 0.0
        for t in reversed(range(T)):
            delta = (
                self.rewards[t] + gamma * bootstrap_values[t + 1] - bootstrap_values[t]
            )
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        adv_tensor = torch.tensor(advantages)
        # Normalize across the rollout for stable updates
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        ret_tensor = torch.tensor(returns)

        return adv_tensor, ret_tensor

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Pack raw lists into tensors on the target device."""
        return {
            "states": torch.tensor(
                np.array(self.states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(
                np.array(self.actions), dtype=torch.float32, device=device
            ),
            "log_probs": torch.tensor(
                np.array(self.log_probs), dtype=torch.float32, device=device
            ),
        }

    def sample_delta(
        self,
        t: int,
        gamma: float,
    ) -> int:
        T = len(self.states)

        # Geometric offset, clipped to remaining episode length
        max_delta = T - t - 1
        delta = min(np.random.geometric(1 - gamma), max_delta)
        delta = max(delta, 1)
        return delta  # ensure at least 1 step into future

    @torch.no_grad()
    def compute_intrinsic_rewards(
        self,
        critic_encoder,
        gamma: float,
    ):
        """
        Compute intrinsic rewards and optionally normalize them.


        """
        tensors = self.to_tensors(device=critic_encoder.device)
        states = tensors["states"]
        actions = tensors["actions"]
        H = actions.shape[0]

        for t in range(H - 1):
            # Sample single future state via geometric distribution
            delta = self.sample_delta(t, gamma)
            sf = states[t + delta]

            c = critic_encoder(
                states[t].unsqueeze(0), actions[t].unsqueeze(0), sf.unsqueeze(0)
            )
            self.rewards[t] = -c.item()  # -(-distance) = +distance

    @torch.no_grad()
    def compute_intrinsic_rewards_rnd(
        self,
        rnd_model,
        gamma: float,
        return_rms: RunningMeanStd | None = None,
    ) -> None:
        """Compute RND intrinsic rewards for the trajectory.

        The reward at step t is the prediction error on the next state s_{t+1}:
            r_i(t) = ||f_hat(s_{t+1}) - f(s_{t+1})||^2

        The last step gets reward 0.0 (no next state available).
        Args:
            rnd_model: RNDModel instance (holds target + predictor).
            gamma:     Discount factor for return-based normalization.
            return_rms: Optional RunningMeanStd for reward normalization.
        """
        T = len(self.states)
        if T < 2:
            return

        states = torch.tensor(
            np.array(self.states), dtype=torch.float32, device=rnd_model.device
        )
        # Reward for step t = prediction error on s_{t+1}
        next_states = states[1:]  # (T-1, state_dim)
        errors = rnd_model.compute_reward(next_states)  # (T-1,)

        raw = errors.cpu().numpy().astype(np.float64)
        for t in range(T - 1):
            self.rewards[t] = float(raw[t])
        self.rewards[T - 1] = 0.0  # no next state for the terminal step

        if return_rms is not None:
            # Compute discounted returns for this episode (backward pass)
            returns = np.zeros(T - 1, dtype=np.float64)
            G = 0.0
            for t in reversed(range(T - 1)):
                G = raw[t] + gamma * G
                returns[t] = G

            # Update running statistics with returns, normalize rewards by std
            return_rms.update(returns)
            normed = raw / (return_rms.std + 1e-8)
            for t in range(T - 1):
                self.rewards[t] = float(normed[t])


class TrajectoryBuffer:
    """Stores complete trajectories. Simple deque with a max size."""

    def __init__(self, max_trajectories: int = 5000):
        self.trajectories: deque[Trajectory] = deque(maxlen=max_trajectories)

    def add(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)

    def get_last(self) -> Trajectory:
        return self.trajectories[-1]

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_steps(self) -> int:
        return sum(len(t) for t in self.trajectories)

    def sample_with_futures(
        self,
        batch_size: int,
        gamma,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (s_t, a_t, s_f) tuples with geometric future offsets.

        The geometric distribution Δ ~ Geom(1 - γ) means:
            - P(Δ = 1) = 1 - γ       (most likely: immediate next state)
            - P(Δ = k) = γ^{k-1}(1-γ) (decays exponentially)
            - E[Δ] = 1 / (1 - γ)     (e.g., 100 for γ=0.99)

        This weighting ensures the contrastive model learns about
        both short-term and long-term temporal relationships.
        """
        n_trajs = len(self.trajectories)
        states, actions, futures = [], [], []

        for _ in range(batch_size):
            # Sample trajectory uniformly
            traj_idx = np.random.randint(n_trajs)
            traj = self.trajectories[traj_idx]
            T = len(traj)

            # Sample timestep (need at least one future step)
            t = np.random.randint(T - 1)

            # Geometric offset, clipped to remaining episode length
            max_delta = T - t - 1
            delta = min(np.random.geometric(1 - gamma), max_delta)
            delta = max(delta, 1)  # ensure at least 1 step into future

            states.append(traj.states[t])
            actions.append(traj.actions[t])
            futures.append(traj.states[t + delta])

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.tensor(np.array(futures), dtype=torch.float32, device=device),
        )
