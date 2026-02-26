from __future__ import annotations

import argparse
import logging
from pathlib import Path

import minigrid  # noqa
import numpy as np
import torch

from c_tec.buffer import TrajectoryBuffer
from c_tec.config import get_config
from c_tec.envs import make_env
from c_tec.evaluate import evaluate
from c_tec.models import CTeCPolicy, RandomPolicy, RNDPolicy
from c_tec.train import train
from c_tec.utils.visualization import (
    plot_coverage_over_time,
    plot_cumulative_coverage,
    plot_cumulative_coverage_comparison,
    plot_heatmap_of_position,
    plot_heatmap_of_position_filtered,
    plot_reached_states,
)

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
        "--method", type=str, default="c-tec", choices=["random", "c-tec", "rnd"]
    )
    parser.add_argument("--seed", type=int, default=28)
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
        help="Number of episodes per evaluation run.",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save a periodic checkpoint every N episodes. 0 to disable.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="c-tec_config.yaml",
        help="path to the yaml configuration file",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation mode instead of training. Requires --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file to load for evaluation.",
    )

    args = parser.parse_args()

    CONFIG = get_config(args.config_file)
    RESULTS_DIR = Path(__file__).parent.parent / args.output_dir / args.method

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Setup ---
    env = make_env(seed=args.seed, max_steps=CONFIG.training.episode_length)
    n_actions = 7
    state_dim = 6
    trajectory_buffer = TrajectoryBuffer()

    match args.method:
        case "random":
            policy = RandomPolicy(n_actions)

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
                action_dim=n_actions,
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
            raise RuntimeError(f"Unknown method: {args.method}")

    logger.info(f"Selected Policy: {args.method}")
    logger.info("Environment: MiniGrid-FourRooms-v0")
    logger.info(f"Reachable cells: {env.n_reachable}")
    logger.info(f"Action space: {n_actions} actions")

    if args.evaluate:
        # --- Evaluate ---
        if args.method != "random":
            if args.checkpoint is None:
                raise RuntimeError(
                    "--checkpoint is required when running in --evaluate mode."
                )

            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            policy.load(checkpoint_path)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        logger.info(
            f"Running {args.n_eval_episodes} evaluation episodes "
            f"with {args.method} policy\n"
        )

        EVAL_DIR = RESULTS_DIR / "eval"
        eval_logger, last_stats = evaluate(
            env=env,
            policy=policy,
            n_episodes=args.n_eval_episodes,
            seed=args.seed,
        )

        # --- Save evaluation results ---
        eval_logger.save(EVAL_DIR / "eval_metrics.json")

        eval_trajectory_buffer = last_stats["trajectory_buffer"]
        eval_trajectory_buffer.save(EVAL_DIR / "trajectory_buffer.pkl")

        plot_coverage_over_time(
            trajectory_buffer=eval_trajectory_buffer,
            n_reachable=env.n_reachable,
            save_path=EVAL_DIR / "coverage_over_time.png",
        )

        plot_cumulative_coverage(
            trajectory_buffer=eval_trajectory_buffer,
            save_path=EVAL_DIR / "cumulative_visited_states.png",
            label=args.method,
            title=f"Cumulative Visited States ({args.method})",
        )

        logger.info(f"Evaluation results saved to {EVAL_DIR}")

    else:
        # --- Train ---
        logger.info(
            f"Running {CONFIG.training.num_episodes} episodes with {args.method} policy\n"
        )

        train_logger, last_stats = train(
            env=env,
            policy=policy,
            trajectory_buffer=trajectory_buffer,
            n_episodes=CONFIG.training.num_episodes,
            seed=args.seed,
            method=args.method,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            n_eval_episodes=args.n_eval_episodes,
            save_path=RESULTS_DIR,
            checkpoint_interval=args.checkpoint_interval,
        )

        # --- Save results ---
        train_logger.save(RESULTS_DIR / "metrics.json")
        trajectory_buffer.save(RESULTS_DIR / "trajectory_buffer.pkl")

        plot_coverage_over_time(
            trajectory_buffer=trajectory_buffer,
            n_reachable=env.n_reachable,
            save_path=RESULTS_DIR / "coverage_over_time.png",
        )

        plot_heatmap_of_position(
            reached_count=last_stats["reached_count"],
            n_episodes=CONFIG.training.num_episodes,
            reachable_cells=env.compute_reachable(),
            starting_cell=last_stats["starting_pos"],
            save_path=RESULTS_DIR / "heatmap_of_positions.png",
        )

        plot_heatmap_of_position_filtered(
            reached_count=last_stats["reached_count"],
            n_episodes=CONFIG.training.num_episodes,
            reachable_cells=env.compute_reachable(),
            starting_cell=last_stats["starting_pos"],
            save_path=RESULTS_DIR / "heatmap_filtered.png",
            min_probability=0.1,
        )

        plot_reached_states(
            train_logger=train_logger,
            save_path=RESULTS_DIR / "reached_states.png",
        )

        plot_cumulative_coverage(
            trajectory_buffer=trajectory_buffer,
            save_path=RESULTS_DIR / "cumulative_visited_states.png",
            label=args.method,
            title=f"Cumulative Visited States ({args.method})",
        )

        logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
