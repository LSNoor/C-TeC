import argparse
import logging
from pathlib import Path

import minigrid  # noqa
import numpy as np
import torch

from c_tec.buffer import TrajectoryBuffer
from c_tec.config import get_config
from c_tec.environment import make_env
from c_tec.models import get_policy
from train import run_training
from evaluate import run_evaluation

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
    parser.add_argument(
        "--seed", type=int, default=28, help="The seed to run the environment with."
    )
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval.")
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=50,
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
        "--mode",
        type=str,
        default="training",
        choices=["evaluation", "training"],
        help="Whether to run the training or evaluation mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file to load for evaluation.",
    )
    parser.add_argument(
        "--plot-rewards-interval",
        type=int,
        default=0,
        help="The interval between plotting rewards.",
    )
    parser.add_argument("--use-one-seed", action="store_true")

    args = parser.parse_args()

    CONFIG = get_config(args.config_file)
    RESULTS_DIR = Path(__file__).parent.parent / args.output_dir / args.method
    EVAL_DIR = RESULTS_DIR / "eval"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Setup ---
    env = make_env(seed=args.seed, max_steps=CONFIG.env.episode_length)
    n_actions = 7
    state_dim = 6
    trajectory_buffer = TrajectoryBuffer()

    policy = get_policy(
        method=args.method,
        state_dim=state_dim,
        action_dim=n_actions,
        CONFIG=CONFIG,
        device=device,
    )

    logger.info(f"Selected Policy: {args.method}")
    logger.info("Environment: MiniGrid-FourRooms-v0")
    logger.info(f"Reachable cells: {env.n_reachable}")
    logger.info(f"Action space: {n_actions} actions")

    if args.mode == "evaluation":
        eval_logger, last_stats = run_evaluation(
            method=args.method,
            policy=policy,
            env=env,
            seed=args.seed,
            num_episodes=CONFIG.env.num_episodes,
            evaluate_multiple_seeds=not args.use_one_seed,
            from_checkpoint=True,
            checkpoint_path=args.checkpoint,
            save=True,
            results_directory=RESULTS_DIR,
            eval_directory=EVAL_DIR,
        )

    else:

        logger.info(
            f"Running {CONFIG.env.num_episodes} episodes with {args.method} policy\n"
        )

        train_logger, last_stats = run_training(
            method=args.method,
            policy=policy,
            env=env,
            seed=args.seed,
            num_episodes=CONFIG.env.num_episodes,
            save=True,
            results_directory=RESULTS_DIR,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            use_multiple_seeds=not args.use_one_seed,
            plot_rewards_interval=args.plot_rewards_interval,
        )


if __name__ == "__main__":
    main()
