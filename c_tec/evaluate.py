import logging

import numpy as np
from tqdm import trange

from c_tec.buffer import TrajectoryBuffer
from c_tec.train import collect_episode
from c_tec.utils.MetricsLogger import MetricsLogger

logger = logging.getLogger(__name__)


def evaluate(
    env,
    policy,
    n_episodes: int,
    seed: int,
    evaluate_multiple_seeds: bool = True,
) -> tuple[MetricsLogger, dict]:

    eval_logger = MetricsLogger()
    trajectory_buffer = TrajectoryBuffer()
    total_steps = 0
    last_stats: dict = {}

    coverages: list[float] = []

    if evaluate_multiple_seeds:
        seed_list = [(s % 10) * 25 for s in [(seed + i) for i in range(n_episodes)]]
    else:
        seed_list = [seed for _ in range(n_episodes)]

    for episode in trange(1, n_episodes + 1, desc="Evaluating"):
        stats = collect_episode(
            env,
            policy,
            trajectory_buffer,
            is_training=False,
            seed=seed_list[episode - 1],
        )
        total_steps += stats["episode_length"]
        last_stats = stats
        coverages.append(stats["episode_coverage_pct"])

        # Log per-episode stats (excluding non-serialisable keys)
        eval_logger.log(
            episode=episode,
            total_steps=total_steps,
            episode_length=stats["episode_length"],
            episode_coverage=stats["episode_coverage"],
            episode_coverage_pct=stats["episode_coverage_pct"],
        )

        logger.info(
            f"[Eval ep {episode:3d}] "
            f"Coverage: {stats['episode_coverage']} cells "
            f"({stats['episode_coverage_pct']:.1%}) | "
            f"Length: {stats['episode_length']}"
        )

    # Summary statistics
    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    logger.info(
        f"\nEvaluation complete: {n_episodes} episodes | "
        f"Coverage: {mean_cov:.1%} ± {std_cov:.1%} | "
        f"Total steps: {total_steps}"
    )

    # Attach the trajectory buffer to last_stats so the caller can use it
    # for visualizations (coverage_over_time needs it).
    last_stats["trajectory_buffer"] = trajectory_buffer

    return eval_logger, last_stats
