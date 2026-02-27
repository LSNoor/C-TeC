import argparse
import logging
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Tuple

import minigrid  # noqa
import numpy as np
import torch

from c_tec.buffer import TrajectoryBuffer
from c_tec.config import get_config, Config
from c_tec.envs import make_env
from c_tec.evaluate import evaluate
from c_tec.models import CTeCPolicy, RandomPolicy, RNDPolicy
from c_tec.train import train
from c_tec.utils.MetricsLogger import MetricsLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def run_evaluation(
    method: Literal["random", "c-tec", "rnd"],
    policy,
    env,
    seed: int,
    num_episodes: int,
    evaluate_multiple_seeds: bool = True,
    from_checkpoint: bool = True,
    checkpoint_path: Optional[str] = None,
    save: bool = True,
    results_directory: Path | str | None = None,
    eval_directory: Path = None,
) -> Tuple[MetricsLogger, dict]:

    env.reset_reached_count()

    if save and (results_directory is None or eval_directory is None):
        raise RuntimeError(
            "To save an evaluation run, the result and evaluation directory must be specified."
        )

    if from_checkpoint == True and method != "random":
        if checkpoint_path is None:
            raise RuntimeError(
                "--checkpoint is required when running in --evaluate mode."
            )

        checkpoint_path = results_directory / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        policy.load(checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    logger.info(
        f"Running {num_episodes} evaluation episodes " f"with {method} policy\n"
    )

    eval_logger, last_stats = evaluate(
        env=env,
        policy=policy,
        n_episodes=num_episodes,
        seed=seed,
        evaluate_multiple_seeds=evaluate_multiple_seeds,
    )

    # --- Save evaluation results ---
    if save:
        eval_logger.save(eval_directory / "eval_metrics.json")

        eval_trajectory_buffer = last_stats["trajectory_buffer"]
        eval_trajectory_buffer.save(eval_directory / "trajectory_buffer.pkl")

        logger.info(f"Evaluation results saved to {eval_directory}")
    return eval_logger, last_stats


def run_training(
    method: Literal["random", "c-tec", "rnd"],
    policy,
    env,
    seed: int,
    num_episodes: int,
    save: bool = True,
    results_directory: Optional[Path] = None,
    log_interval: int = 1,
    checkpoint_interval: int = 0,
):

    env.reset_reached_count()

    logger.info(f"Running {num_episodes} episodes with {method} policy\n")

    train_logger, last_stats = train(
        env=env,
        policy=policy,
        n_episodes=num_episodes,
        seed=seed,
        method=method,
        log_interval=log_interval,
        save_path=results_directory if save else None,
        checkpoint_interval=checkpoint_interval,
    )

    trajectory_buffer = last_stats["trajectory_buffer"]

    # --- Save results ---
    if save:

        train_logger.save(results_directory / "metrics.json")
        trajectory_buffer.save(results_directory / "trajectory_buffer.pkl")

    return trajectory_buffer, train_logger


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
        trajectory_buffer = run_evaluation(
            method=args.method,
            policy=policy,
            env=env,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            CONFIG=CONFIG,
            save=True,
            eval_directory=EVAL_DIR,
            results_directory=RESULTS_DIR,
        )

    else:

        logger.info(
            f"Running {CONFIG.env.num_episodes} episodes with {args.method} policy\n"
        )
        train(
            env=env,
            policy=policy,
            n_episodes=CONFIG.env.num_episodes,
            seed=args.seed,
            method=args.method,
            log_interval=args.log_interval,
            save_path=RESULTS_DIR,
            checkpoint_interval=args.checkpoint_interval,
            use_multiple_seeds=False,
        )

        # # --- Save results ---
        # train_logger.save(RESULTS_DIR / "metrics.json")
        # trajectory_buffer.save(RESULTS_DIR / "trajectory_buffer.pkl")
        #
        # plot_coverage_over_time(
        #     trajectory_buffer=trajectory_buffer,
        #     n_reachable=env.n_reachable,
        #     save_path=RESULTS_DIR / "coverage_over_time.png",
        # )
        #
        # plot_heatmap_of_position(
        #     reached_count=last_stats["reached_count"],
        #     n_episodes=CONFIG.env.num_episodes,
        #     reachable_cells=env.compute_reachable(),
        #     starting_cell=last_stats["starting_pos"],
        #     save_path=RESULTS_DIR / "heatmap_of_positions.png",
        # )
        #
        # plot_heatmap_of_position_filtered(
        #     reached_count=last_stats["reached_count"],
        #     n_episodes=CONFIG.env.num_episodes,
        #     reachable_cells=env.compute_reachable(),
        #     starting_cell=last_stats["starting_pos"],
        #     save_path=RESULTS_DIR / "heatmap_filtered.png",
        #     min_probability=0.1,
        # )
        #
        # plot_reached_states(
        #     train_logger=train_logger,
        #     save_path=RESULTS_DIR / "reached_states.png",
        # )
        #
        # plot_cumulative_coverage(
        #     trajectory_buffer=trajectory_buffer,
        #     save_path=RESULTS_DIR / "cumulative_visited_states.png",
        #     label=args.method,
        #     title=f"Cumulative Visited States ({args.method})",
        # )
        #
        # logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
