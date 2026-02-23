from __future__ import annotations

import logging

import numpy as np

from c_tec.buffer import TrajectoryBuffer
from c_tec.train import collect_episode

logger = logging.getLogger(__name__)


def evaluate(
    env,
    policy,
    trajectory_buffer: TrajectoryBuffer,
    n_eval_episodes: int,
    seed: int,
    episode_offset: int = 0,
) -> dict:
    """
    Run a fixed number of evaluation episodes with the given policy.

    No rollout buffer is passed to ``collect_episode``, so no gradient
    information is stored and no policy update is performed.  Works with
    both ``RandomPolicy`` and ``PPOPolicy``.

    Parameters
    ----------
    env               : Gymnasium environment (wrapped).
    policy            : RandomPolicy or PPOPolicy.
    trajectory_buffer : Buffer that receives the evaluation trajectories.
    n_eval_episodes   : Number of episodes to run.
    seed              : Base random seed; each episode uses seed + i.
    episode_offset    : Added to the per-episode seed index to avoid
                        collisions with training seeds
                        (e.g. pass the current training episode number).

    Returns
    -------
    dict with keys:
        mean_coverage_pct : float  – mean fractional coverage across episodes.
        std_coverage_pct  : float  – std of fractional coverage.
        mean_coverage     : float  – mean number of cells visited.
        std_coverage      : float  – std of cells visited.
        mean_length       : float  – mean episode length.
        coverages_pct     : list[float]  – per-episode fractional coverages.
    """
    coverages_pct: list[float] = []
    coverages: list[int] = []
    lengths: list[int] = []

    for i in range(n_eval_episodes):
        stats = collect_episode(
            env,
            policy,
            trajectory_buffer,
            rollout_buffer=None,  # evaluation mode: no gradient data stored
            seed=seed + episode_offset + i,
        )
        coverages_pct.append(stats["episode_coverage_pct"])
        coverages.append(stats["episode_coverage"])
        lengths.append(stats["episode_length"])

    results = {
        "mean_coverage_pct": float(np.mean(coverages_pct)),
        "std_coverage_pct": float(np.std(coverages_pct)),
        "mean_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "mean_length": float(np.mean(lengths)),
        "coverages_pct": coverages_pct,
    }

    logger.info(
        f"Evaluation over {n_eval_episodes} episodes: "
        f"coverage {results['mean_coverage_pct']:.1%} "
        f"± {results['std_coverage_pct']:.1%}"
    )

    return results