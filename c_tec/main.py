from __future__ import annotations

import argparse
import logging
from pathlib import Path

import minigrid  # noqa
import numpy as np
import torch

from c_tec.buffer import TrajectoryBuffer
from c_tec.envs import make_env
from c_tec.models import RandomPolicy, CTeCPolicy
from c_tec.train import train
from c_tec.utils.visualization import (
    plot_coverage_over_time,
    plot_heatmap_of_position,
    plot_heatmap_of_position_filtered,
    plot_reached_states,
)
from c_tec.config import get_config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default="c-tec", choices=["random", "c-tec"]
    )
    parser.add_argument(
        "--n-episodes", type=int, default=150, help="Number of training episodes."
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="[PPO] Run evaluation every N training episodes. 0 to disable.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="[PPO] Number of episodes per evaluation run.",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum number of steps per episode (truncation limit).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.yaml",
        help="path to the yaml configuration file",
    )

    args = parser.parse_args()

    CONFIG = get_config(args.config_file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Setup ---
    env = make_env(seed=args.seed, max_steps=args.max_steps)
    n_actions = 7
    state_dim = 6
    trajectory_buffer = TrajectoryBuffer()

    match args.method:

        case "random":
            policy = RandomPolicy(n_actions)
            rollout = None

        case "c-tec":
            policy = CTeCPolicy(
                state_dim=state_dim,
                action_dim=n_actions,
                hidden_dim=CONFIG.hyperparameters.hidden_dim,
                sa_hidden_dim=CONFIG.c_tec.hidden_dim,
                sf_hidden_dim=CONFIG.c_tec.hidden_dim,
                repr_dim=CONFIG.c_tec.representation_dim,
                similarity_function=CONFIG.c_tec.similarity_function,
                policy_lr=CONFIG.hyperparameters.policy_lr,
                critic_lr=CONFIG.hyperparameters.critic_lr,
                gamma=0.8,  # CONFIG.hyperparameters.discount_factor,
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

        case _:
            raise RuntimeError(f"Unknown method: {args.method}")

    logger.info(f"Selected Policy: {args.method}")
    logger.info("Environment: MiniGrid-FourRooms-v0")
    logger.info(f"Reachable cells: {env.n_reachable}")
    logger.info(f"Action space: {n_actions} actions")
    logger.info(f"Running {args.n_episodes} episodes with {args.method} policy\n")

    # --- Train ---
    train_logger, last_stats = train(
        env=env,
        policy=policy,
        trajectory_buffer=trajectory_buffer,
        n_episodes=args.n_episodes,
        seed=args.seed,
        method=args.method,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
    )

    # --- Save results ---
    out = Path(args.output_dir) / args.method
    train_logger.save(out / "metrics.json")

    plot_coverage_over_time(
        trajectory_buffer=trajectory_buffer,
        n_reachable=env.n_reachable,
        save_path=out / "coverage_over_time.png",
    )

    plot_heatmap_of_position(
        reached_count=last_stats["reached_count"],
        n_episodes=args.n_episodes,
        reachable_cells=env.compute_reachable(),
        starting_cell=last_stats["starting_pos"],
        save_path=out / "heatmap_of_positions.png",
    )

    plot_heatmap_of_position_filtered(
        reached_count=last_stats["reached_count"],
        n_episodes=args.n_episodes,
        reachable_cells=env.compute_reachable(),
        starting_cell=last_stats["starting_pos"],
        save_path=out / "heatmap_filtered.png",
        min_probability=0.1,
    )

    plot_reached_states(
        train_logger=train_logger,
        save_path=out / "reached_states.png",
    )

    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
