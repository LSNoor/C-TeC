from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import trange

from c_tec.buffer import RunningMeanStd, Trajectory, TrajectoryBuffer
from c_tec.utils.logging import MetricsLogger

logger = logging.getLogger(__name__)


def collect_episode(
    env,
    policy,
    trajectory_buffer: TrajectoryBuffer,
    is_training: bool,
    seed: int | None = None,
) -> dict:
    """
    Run one episode and collect statistics.

    This single function handles both training and evaluation:

    Training (PPO)
        Pass a RolloutBuffer.  At each step the function stores
        (pre-step obs, action, log_prob, value, reward, done) so that
        PPOPolicy.update() can compute GAE and run the clipped-surrogate
        update immediately after this call.

    Evaluation / random policy
        Pass rollout_buffer=None.  Only environment statistics are
        collected; no gradient information is stored.  Works with both

    The full trajectory (post-step observations) is always added to
    trajectory_buffer for C-TeC's geometric future sampling and the
    coverage visualizations.

    Returns
    -------
    dict with keys:
        episode_length      : int
        episode_coverage    : int   (cells visited)
        episode_coverage_pct: float
        reached_count       : dict  (cell → visit count)
        starting_pos        : tuple
        last_obs            : np.ndarray  (terminal / truncated observation)
        terminated          : bool  (False if episode was truncated)
    """
    obs, _ = env.reset(seed=seed)
    trajectory = Trajectory()
    steps = 0
    starting_pos = (obs[0], obs[1])
    terminated = False
    truncated = False

    is_training = is_training
    done = False

    while not done:
        pre_step_obs = obs

        action, log_prob, value = policy.select_action(pre_step_obs)

        obs, _, _, truncated, info = env.step(action)
        done = truncated  # In our context, there is no goal to reach
        # -> the episode ends when max episode length is reached

        # Trajectory stores post-step obs for C-TeC future-state sampling
        trajectory.append(
            state=pre_step_obs.copy(),
            action=env.env.action_to_onehot(action),
            log_prob=log_prob,
            value=value,
            reward=0.0,  # No extrinsic reward in c-tec
            cell_covered=info["episode_coverage"],
            cell_covered_pct=info["episode_coverage_pct"],
        )

        steps += 1

    trajectory_buffer.add(trajectory)

    return {
        "episode_length": steps,
        "episode_coverage": info["episode_coverage"],
        "episode_coverage_pct": info["episode_coverage_pct"],
        "reached_count": info["reached_count"],
        "starting_pos": starting_pos,
        "last_obs": obs,  # used for GAE bootstrap on truncation
    }


def train(
    env,
    policy,
    trajectory_buffer: TrajectoryBuffer,
    n_episodes: int,
    seed: int,
    method: str,
    log_interval: int,
    eval_interval: int,
    n_eval_episodes: int,
    save_path: str | Path | None = None,
    checkpoint_interval: int = 0,
) -> tuple[MetricsLogger, dict]:
    """
    Run the full training loop.

    Parameters
    ----------
    env                  : Gymnasium environment (wrapped).
    policy               : RandomPolicy or PPOPolicy.
    trajectory_buffer    : Buffer for full trajectories (C-TeC / visualizations).
    n_episodes           : Total number of training episodes to run.
    seed                 : Base random seed.
    method               : "random" or "ppo".
    log_interval         : Print summary every N episodes.
    eval_interval        : Run evaluation every N episodes (PPO only; 0 to disable).
    n_eval_episodes      : Number of episodes per evaluation run.
    save_path            : Directory where checkpoints and the best/final model
                           are written.  Saving is skipped when None or when the
                           policy does not expose a ``save()`` method (e.g.
                           RandomPolicy).
    checkpoint_interval  : Save a periodic checkpoint every N episodes.
                           0 disables periodic checkpoints.

    Returns
    -------
    train_logger : MetricsLogger populated with per-episode stats.
    last_stats   : The raw stats dict from the final episode (used for plots).
    """
    train_logger = MetricsLogger()
    total_steps = 0
    ppo_metrics: dict = {}
    last_stats: dict = {}
    return_rms = RunningMeanStd()  # running return normalizer (std-only, RND-style)

    # ── Saving setup ─────────────────────────────────────────────────
    can_save = save_path is not None and hasattr(policy, "save")
    save_path = Path(save_path) if save_path is not None else None
    best_coverage_pct: float = -1.0
    checkpoints_dir: Path | None = None
    if can_save:
        assert save_path is not None
        checkpoints_dir = save_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for episode in trange(1, n_episodes + 1):
        stats = collect_episode(
            env,
            policy,
            trajectory_buffer,
            is_training=True,
            seed=seed,
        )
        total_steps += stats["episode_length"]
        last_stats = stats

        # ── PPO update (training episodes only) ──────────────────────────
        if method == "c-tec":
            # The method of termination is always max steps reached because we are
            # exploring without explicit goal. Therefore, we always bootstrap last value from critic network
            _, _, last_value = policy.select_action(stats["last_obs"])

            trajectory = trajectory_buffer.get_last()
            trajectory.compute_intrinsic_rewards(
                policy.critic_encoder, gamma=policy.gamma, return_rms=return_rms
            )
            policy.update_contrastive(trajectory_buffer)
            logger.info(
                f"mean reward (normalized): {np.array(trajectory.rewards).mean():.4f} "
                f"| return running std: {return_rms.std:.4f}"
            )

            ppo_metrics = policy.update(trajectory, last_value=last_value)

        elif method == "rnd":
            # Same episode-level structure as C-TeC:
            #   1. compute intrinsic rewards via RND prediction error
            #   2. update the predictor on the episode's states
            #   3. PPO update on the intrinsic rewards
            _, _, last_value = policy.select_action(stats["last_obs"])

            trajectory = trajectory_buffer.get_last()
            trajectory.compute_intrinsic_rewards_rnd(
                policy.rnd_model, gamma=policy.gamma, return_rms=return_rms
            )
            rnd_loss = policy.update_rnd(trajectory)
            logger.info(
                f"mean reward (normalized): {np.array(trajectory.rewards).mean():.4f} "
                f"| RND predictor loss: {rnd_loss:.4f} "
                f"| return running std: {return_rms.std:.4f}"
            )

            ppo_metrics = policy.update(trajectory, last_value=last_value)
            ppo_metrics["rnd_loss"] = rnd_loss

        # ── Logging ─────────────────────────────────────────────────────
        stats_to_save = deepcopy(stats)
        for key in ("reached_count", "starting_pos", "last_obs"):
            stats_to_save.pop(key, None)
        if method in ["c-tec", "rnd"]:
            stats_to_save.update(ppo_metrics)

        train_logger.log(episode=episode, total_steps=total_steps, **stats_to_save)

        # ── Checkpoint saving ────────────────────────────────────────
        if can_save:
            assert save_path is not None
            # Best model: save whenever coverage improves
            current_coverage_pct = stats["episode_coverage_pct"]
            if current_coverage_pct > best_coverage_pct:
                best_coverage_pct = current_coverage_pct
                policy.save(
                    save_path / "best_model.pt",
                    episode=episode,
                    total_steps=total_steps,
                )
                logger.info(
                    f"New best coverage {best_coverage_pct:.1%} — best_model.pt updated"
                )

            # Periodic checkpoint
            if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
                assert checkpoints_dir is not None
                policy.save(
                    checkpoints_dir / f"checkpoint_ep{episode:06d}.pt",
                    episode=episode,
                    total_steps=total_steps,
                )

        if episode % log_interval == 0:
            recent = train_logger.recent(log_interval)
            ppo_str = ""
            if method in ["c-tec", "rnd"]:
                ppo_str = (
                    f" | Loss {recent['total_loss']:.4f}"
                    f"  π {recent['policy_loss']:.4f}"
                    f"  V {recent['value_loss']:.4f}"
                    f"  H {recent['entropy']:.4f}"
                )
            logger.info(
                f"[Ep {episode:5d}] "
                f"Steps: {total_steps:8d} | "
                f"Avg coverage: {recent['episode_coverage']:.1f} cells "
                f"({recent['episode_coverage_pct']:.1%}) | "
                f"Avg len: {recent['episode_length']:.0f} | "
                f"Buffer: {trajectory_buffer.n_trajectories} traj" + ppo_str
            )

        # # ── Periodic evaluation (PPO only) ───────────────────────────────
        # # collect_episode with rollout_buffer=None → evaluation mode:
        # # no rollout data stored, no PPO update, policy runs @no_grad.
        # if method == "ppo" and eval_interval > 0 and episode % eval_interval == 0:
        #     eval_coverages = []
        #     for eval_ep in range(n_eval_episodes):
        #         eval_stats = collect_episode(
        #             env,
        #             policy,
        #             trajectory_buffer,
        #             rollout_buffer=None,  # ← evaluation mode
        #             seed=seed + episode + eval_ep,
        #         )
        #         eval_coverages.append(eval_stats["episode_coverage_pct"])
        #     print(
        #         f"  [Eval @ ep {episode:5d}] "
        #         f"Coverage: {np.mean(eval_coverages):.1%} "
        #         f"± {np.std(eval_coverages):.1%}"
        #         f" over {n_eval_episodes} episodes"
        #     )

    # ── Final model save ─────────────────────────────────────────────
    if can_save:
        assert save_path is not None
        policy.save(
            save_path / "final_model.pt",
            episode=n_episodes,
            total_steps=total_steps,
        )

    return train_logger, last_stats
