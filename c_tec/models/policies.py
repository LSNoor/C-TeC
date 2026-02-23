import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal

from buffer import TrajectoryBuffer, Trajectory

from .actor_critic_models import ActorModel, CriticModel
from .encoders import CriticEncoder
from .loss import CRLLoss


class RandomPolicy:
    """Uniform random action selection."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def select_action(self, obs: np.ndarray):
        return np.random.randint(self.n_actions), None, None


class PPOPolicy:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = ActorModel(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticModel(state_dim, hidden_dim).to(device)

        # Separate optimizers — actor and critic are updated independently
        self.policy_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=policy_lr,
        )
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=critic_lr)

    # ------------------------------------------------------------------
    # Action selection (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """
        Sample an action from the current policy π_θ.

        This is the single point of contact with the environment loop

        Returns
        -------
        action   : int   — sampled discrete action
        log_prob : float — log π_θ(action | obs), stored for the PPO ratio
        value    : float — V_θ(obs), stored for GAE
        """
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(state)  # Categorical
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state).squeeze(-1)  # scalar
        return action.item(), log_prob.item(), value.item()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        rollout: Trajectory,
        last_value: float = 0.0,
    ) -> dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates over the collected rollout.

        Call this once per rollout (e.g. at the end of each episode or
        after every N steps), then call rollout.clear() before the next
        collection phase.

        Args
        ----
        rollout    : filled RolloutBuffer from collect_episode
        last_value : V(s_T) — 0.0 for terminal episodes; use
                     critic(last_obs) for time-limit truncations so GAE
                     can bootstrap beyond the episode horizon.

        Returns
        -------
        dict with per-update averages of:
            'policy_loss', 'value_loss', 'entropy', 'total_loss'
        """
        advantages, returns = rollout.compute_gae(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        tensors = rollout.to_tensors(self.device)
        states = tensors["states"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        T = len(rollout)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        n_updates = 0

        for _ in range(self.n_epochs):
            # Fresh shuffle each epoch prevents ordering bias
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                # Slice of the rollout for this mini-batch update
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # ── Actor (policy) update ──────────────────────────────
                distribution = self.actor(batch_states)
                new_log_probs = distribution.log_prob(batch_actions.argmax(dim=-1))
                entropy = distribution.entropy().mean()

                # Importance ratio r_t(θ)
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # Clipped surrogate objective (Eq. 7)
                surr1 = ratio * batch_advantages
                surr2 = (
                    ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # ── Critic (value) update ──────────────────────────────
                new_values = self.critic(batch_states).squeeze(-1)
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["total_loss"] += (
                    policy_loss.item()
                    + self.value_coef * value_loss.item()
                    - self.entropy_coef * entropy.item()
                )
                n_updates += 1

        # Report per-update averages
        for key in metrics:
            metrics[key] /= max(n_updates, 1)

        return metrics


class CTeCPolicy(PPOPolicy):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        sa_hidden_dim: int,
        sf_hidden_dim: int,
        repr_dim: int,
        contrastive_lr: float,
        contrastive_batch_size: int,
        policy_lr: float,
        critic_lr: float,
        logsumexp_penalty: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        n_epochs: int,
        batch_size: int,
        max_grad_norm: float,
        similarity_function: Literal["l1", "l2"],
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        self.critic_encoder = CriticEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            sa_hidden_dim=sa_hidden_dim,
            sf_hidden_dim=sf_hidden_dim,
            repr_dim=repr_dim,
            norm_type=similarity_function,
            lr=contrastive_lr,
            logsumexp_penalty=logsumexp_penalty,
            device=device,
        )

        self.contrastive_batch_size = contrastive_batch_size

    def update_contrastive(self, trajectory_buffer: TrajectoryBuffer):

        s, a, s_f = trajectory_buffer.sample_with_futures(
            batch_size=self.contrastive_batch_size, gamma=self.gamma, device=self.device
        )
        self.critic_encoder.update(s, a, s_f)
