import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from c_tec.buffer import Trajectory, TrajectoryBuffer
from c_tec.config import Config
from .actor_critic_models import ActorModel, CriticModel
from .contrastive_encoders import CriticEncoder
from .rnd import RNDModel

logger = logging.getLogger(__name__)


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
        policy_lr: float,
        critic_lr: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        n_epochs,
        batch_size,
        max_grad_norm,
        device,
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _base_checkpoint(self, episode: int, total_steps: int) -> dict:
        """Build the base checkpoint dict shared by all PPO-derived policies.

        Subclasses extend the returned dict with their own components
        before calling torch.save(), so the base keys are defined exactly
        once and never duplicated.
        """
        return {
            "episode": episode,
            "total_steps": total_steps,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def _load_base_checkpoint(self, checkpoint: dict) -> tuple[int, int]:
        """Restore the base PPO components from a checkpoint dict.

        Subclasses call this first, then restore their own extra keys.

        Returns
        -------
        episode, total_steps : metadata stored in the checkpoint.
        """
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        return checkpoint.get("episode", 0), checkpoint.get("total_steps", 0)

    def save(self, path: str | Path, episode: int = 0, total_steps: int = 0) -> None:
        """Save model and optimizer state dicts to a checkpoint file.

        Parameters
        ----------
        path        : Destination ``.pt`` file.
        episode     : Current episode number (stored as metadata for resuming).
        total_steps : Current total environment steps (stored as metadata).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._base_checkpoint(episode, total_steps), path)
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> tuple[int, int]:
        """Load model and optimizer weights from a checkpoint file.

        Parameters
        ----------
        path : Path to the ``.pt`` checkpoint produced by :meth:`save`.

        Returns
        -------
        episode     : Episode number stored in the checkpoint.
        total_steps : Total steps stored in the checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        episode, total_steps = self._load_base_checkpoint(checkpoint)
        logger.info(
            f"Checkpoint loaded ← {path}  (episode {episode}, steps {total_steps})"
        )
        return episode, total_steps


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
        sampling_strategy: Literal["geometric", "uniform"] = "geometric",
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
        self.sampling_strategy = sampling_strategy

    # ------------------------------------------------------------------
    # Persistence (extends PPOPolicy)
    # ------------------------------------------------------------------

    def save(self, path: str | Path, episode: int = 0, total_steps: int = 0) -> None:
        """Save all C-TeC components (actor, critic, contrastive encoder) to a checkpoint.

        Extends _base_checkpoint() with the critic_encoder weights and its
        optimizer state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self._base_checkpoint(episode, total_steps)
        checkpoint["critic_encoder_state_dict"] = self.critic_encoder.state_dict()
        checkpoint["contrastive_optimizer_state_dict"] = (
            self.critic_encoder.optimizer.state_dict()
        )
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> tuple[int, int]:
        """Load all C-TeC components from a checkpoint produced by :meth:`save`.

        Returns
        -------
        episode     : Episode number stored in the checkpoint.
        total_steps : Total steps stored in the checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        episode, total_steps = self._load_base_checkpoint(checkpoint)
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder_state_dict"])
        self.critic_encoder.optimizer.load_state_dict(
            checkpoint["contrastive_optimizer_state_dict"]
        )
        logger.info(
            f"Checkpoint loaded ← {path}  (episode {episode}, steps {total_steps})"
        )
        return episode, total_steps

    def update_contrastive(self, trajectory_buffer: TrajectoryBuffer):

        s, a, s_f = trajectory_buffer.sample_with_futures(
            batch_size=self.contrastive_batch_size,
            gamma=self.gamma,
            device=self.device,
            sampling_strategy=self.sampling_strategy,
        )
        self.critic_encoder.update(s, a, s_f)


class RNDPolicy(PPOPolicy):
    """PPO policy with Random Network Distillation intrinsic rewards.

    Extends PPOPolicy by adding an RNDModel that generates intrinsic
    exploration rewards based on the prediction error of a fixed random
    network. The predictor is updated once per episode on all collected
    states.

    Training loop (mirrors CTeCPolicy):
        1. collect_episode()                          -> fills Trajectory
        2. trajectory.compute_intrinsic_rewards_rnd() -> fills rewards
        3. update_rnd(trajectory)                     -> trains predictor
        4. update(trajectory, last_value)             -> PPO update
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        rnd_hidden_dim: int,
        rnd_repr_dim: int,
        rnd_lr: float,
        policy_lr: float,
        critic_lr: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        n_epochs: int,
        batch_size: int,
        max_grad_norm: float,
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

        self.rnd_model = RNDModel(
            state_dim=state_dim,
            hidden_dim=rnd_hidden_dim,
            repr_dim=rnd_repr_dim,
            lr=rnd_lr,
            device=device,
        )

    # ------------------------------------------------------------------
    # RND predictor update
    # ------------------------------------------------------------------

    def update_rnd(self, trajectory: Trajectory) -> float:
        """Train the RND predictor on the current episode's states.

        1. Update observation running mean/std with all episode states
           (once, before training, so every mini-batch sees the same
           normalization).
        2. Iterate over ``n_epochs`` epochs of shuffled mini-batches
           (matching PPO's update structure) for stable learning.

        This is called once per episode, before the PPO update.

        Args:
            trajectory: The current episode's Trajectory.

        Returns:
            loss: Average predictor MSE loss across all mini-batch updates.
        """
        tensors = trajectory.to_tensors(self.device)
        states = tensors["states"]

        # Update obs normalization statistics once with the full episode
        self.rnd_model.update_obs_stats(states)

        # Mini-batch training over multiple epochs
        T = states.shape[0]
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]
                loss = self.rnd_model.update(states[idx])
                total_loss += loss
                n_updates += 1

        return total_loss / max(n_updates, 1)

    # ------------------------------------------------------------------
    # Persistence (extends PPOPolicy)
    # ------------------------------------------------------------------

    def save(self, path: str | Path, episode: int = 0, total_steps: int = 0) -> None:
        """Save all RND components (actor, critic, RND model) to a checkpoint.

        Extends _base_checkpoint() with the RND model weights and its
        optimizer state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self._base_checkpoint(episode, total_steps)
        checkpoint["rnd_state_dict"] = self.rnd_model.state_dict()
        checkpoint["rnd_optimizer_state_dict"] = self.rnd_model.optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> tuple[int, int]:
        """Load all RND components from a checkpoint produced by save().

        Returns
        -------
        episode     : Episode number stored in the checkpoint.
        total_steps : Total steps stored in the checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        episode, total_steps = self._load_base_checkpoint(checkpoint)
        self.rnd_model.load_state_dict(checkpoint["rnd_state_dict"])
        self.rnd_model.optimizer.load_state_dict(checkpoint["rnd_optimizer_state_dict"])
        logger.info(
            f"Checkpoint loaded ← {path}  (episode {episode}, steps {total_steps})"
        )
        return episode, total_steps


def get_policy(
    method: Literal["random", "c-tec", "rnd"],
    state_dim: int,
    action_dim: int,
    device,
    CONFIG: Optional[Config] = None,
):

    if CONFIG is None and method != "random":
        raise ValueError(
            "A configuration is required to get a policy other than random"
        )

    match method:
        case "random":
            policy = RandomPolicy(action_dim)

        case "c-tec":
            policy = CTeCPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=CONFIG.hyperparameters.hidden_dim,
                sa_hidden_dim=CONFIG.c_tec.hidden_dim,
                sf_hidden_dim=CONFIG.c_tec.hidden_dim,
                repr_dim=CONFIG.c_tec.representation_dim,
                similarity_function=CONFIG.c_tec.similarity_function,
                sampling_strategy=CONFIG.c_tec.sampling_strategy,
                policy_lr=CONFIG.hyperparameters.policy_lr,
                critic_lr=CONFIG.hyperparameters.critic_lr,
                gamma=CONFIG.c_tec.gamma,
                gae_lambda=CONFIG.hyperparameters.gae_lambda,
                max_grad_norm=CONFIG.hyperparameters.max_grad_norm,
                clip_eps=CONFIG.hyperparameters.clip_epsilon,
                value_coef=CONFIG.hyperparameters.value_coef,
                entropy_coef=CONFIG.hyperparameters.entropy_coef,
                n_epochs=CONFIG.hyperparameters.update_epoch,
                batch_size=CONFIG.hyperparameters.minibatch_size,
                device=device,
                contrastive_lr=CONFIG.c_tec.contrastive_lr,
                contrastive_batch_size=CONFIG.c_tec.batch_size,
                logsumexp_penalty=CONFIG.c_tec.logsumexp_penalty,
            )

        case "rnd":
            if CONFIG.rnd is None:
                raise RuntimeError(
                    "The [rnd] section is missing from the config file. "
                    "Add it before running with --method rnd."
                )
            policy = RNDPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=CONFIG.hyperparameters.hidden_dim,
                rnd_hidden_dim=CONFIG.rnd.hidden_dim,
                rnd_repr_dim=CONFIG.rnd.representation_dim,
                rnd_lr=CONFIG.rnd.predictor_lr,
                policy_lr=CONFIG.hyperparameters.policy_lr,
                critic_lr=CONFIG.hyperparameters.critic_lr,
                gamma=CONFIG.hyperparameters.discount_factor,
                gae_lambda=CONFIG.hyperparameters.gae_lambda,
                max_grad_norm=CONFIG.hyperparameters.max_grad_norm,
                clip_eps=CONFIG.hyperparameters.clip_epsilon,
                value_coef=CONFIG.hyperparameters.value_coef,
                entropy_coef=CONFIG.hyperparameters.entropy_coef,
                n_epochs=CONFIG.hyperparameters.update_epoch,
                batch_size=CONFIG.hyperparameters.minibatch_size,
                device=device,
            )

        case _:
            raise RuntimeError(f"Unknown method: {method}")

    return policy
