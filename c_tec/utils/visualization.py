"""Coverage heatmap and training curve plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_coverage_over_time(
    trajectory_buffer,
    n_reachable: int,
    save_path: str | Path,
):

    records = []
    for traj_idx, trajectory in enumerate(trajectory_buffer.trajectories):
        for i in range(len(trajectory.states)):
            if i % 10 == 0:
                records.append(
                    {
                        "step": i,
                        "value": trajectory.cell_covered_pct[i],
                        "trajectory": traj_idx,
                    }
                )

    df = pd.DataFrame(records)

    sns.lineplot(data=df, x="step", y="value", errorbar="ci")
    plt.title("Trajectory States")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_heatmap_of_position(
    reached_count: dict[tuple[int, int], int],
    n_episodes: int,
    reachable_cells: set[tuple[int, int]],
    starting_cell: tuple[int, int],
    save_path: str | Path,
):
    max_r = max(r for r, c in reachable_cells) + 2
    max_c = max(c for r, c in reachable_cells) + 2

    prob = np.full((max_r, max_c), np.nan)
    for r, c in reachable_cells:
        prob[r, c] = reached_count.get((r, c), 0) / n_episodes

    unreachable_mask = np.isnan(prob)

    fig, ax = plt.subplots(figsize=(max_c, max_r))

    ax.set_title("Position Reached Probability", fontsize=16)
    ax.set_xlabel("Column", fontsize=14)
    ax.set_ylabel("Row", fontsize=14)
    ax.tick_params(labelsize=12)

    cbar_kws = ({"label": "Probability", "shrink": 0.8},)

    # Gray background for unreachable cells
    sns.heatmap(
        np.zeros_like(prob),
        annot=False,
        cmap=ListedColormap(["lightgray"]),
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Heatmap for reachable cells (0 = reachable but not reached, >0 = reached)
    sns.heatmap(
        prob,
        annot=False,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        mask=unreachable_mask,
        cbar_kws={"label": "Probability"},
        ax=ax,
    )

    cbar = ax.collections[-1].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Probability", fontsize=14)

    sr, sc = starting_cell
    ax.add_patch(plt.Rectangle((sc, sr), 1, 1, fill=True, edgecolor="blue", lw=2))

    ax.set_title("Position Reach Probability")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_heatmap_of_position_filtered(
    reached_count: dict[tuple[int, int], int],
    n_episodes: int,
    reachable_cells: set[tuple[int, int]],
    starting_cell: tuple[int, int],
    save_path: str | Path,
    min_probability: float = 0.1,
):
    """Heatmap showing only cells with reach probability above a threshold.

    Cells whose probability is at or below *min_probability* are masked
    (shown as light-gray, same as unreachable cells).  This makes it easy
    to see which parts of the grid the agent has learned to reach
    reliably.

    Args:
        reached_count:   Mapping from ``(row, col)`` to the number of
                         episodes in which that cell was visited.
        n_episodes:      Total number of episodes (denominator for
                         probability).
        reachable_cells: Set of all ``(row, col)`` positions that are
                         physically reachable in the grid.
        starting_cell:   The agent's starting position, highlighted with
                         a blue border.
        save_path:       Destination file for the figure.
        min_probability: Cells with probability ≤ this value are masked.
    """
    max_r = max(r for r, c in reachable_cells) + 2
    max_c = max(c for r, c in reachable_cells) + 2

    prob = np.full((max_r, max_c), np.nan)
    for r, c in reachable_cells:
        p = reached_count.get((r, c), 0) / n_episodes
        prob[r, c] = p if p > min_probability else np.nan

    # Anything that is NaN (unreachable OR below threshold) is masked
    hidden_mask = np.isnan(prob)

    fig, ax = plt.subplots(figsize=(max_c, max_r))

    # Gray background for all cells (unreachable + below threshold)
    sns.heatmap(
        np.zeros_like(prob),
        annot=False,
        cmap=ListedColormap(["lightgray"]),
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Heatmap for cells above the threshold
    sns.heatmap(
        prob,
        annot=False,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        mask=hidden_mask,
        cbar_kws={"label": "Probability"},
        ax=ax,
    )

    cbar = ax.collections[-1].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Probability", fontsize=14)

    sr, sc = starting_cell
    ax.add_patch(plt.Rectangle((sc, sr), 1, 1, fill=True, edgecolor="blue", lw=2))

    ax.set_title(
        f"Position Reach Probability (>{min_probability:.0%})",
        fontsize=16,
    )
    ax.set_xlabel("Column", fontsize=14)
    ax.set_ylabel("Row", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_coverage_comparison(
    policies: list[str],
    results_dir: str | Path = "c_tec/results",
    title: str = "",
    save_path: str | Path | None = None,
    window: int = 10,
) -> None:
    """Plot episode coverage over total environment steps for multiple policies.

    For each policy the function discovers all seed files under
    ``results_dir/{policy}/``:

    * ``metrics.json``         – single flat file (one seed)
    * ``metrics_<seed>.json``  – multiple files at the same level
    * ``<seed>/metrics.json``  – one sub-directory per seed

    Each seed's coverage series is smoothed independently with a rolling
    window.  ``sns.lineplot`` with ``estimator="mean"`` and ``errorbar="sd"``
    then aggregates across seeds at each environment step, producing a mean
    line with a ±1 std shaded band.  With a single seed there is no CI band.

    Special colour assignments:
    - ``"c-tec"``  → green  (``#2eb88a``)
    - ``"random"`` → gray   (``#808080``)
    - all others   → seaborn default colour cycle

    Args:
        policies:    Ordered list of policy names, e.g. ``["random", "c-tec"]``.
        results_dir: Root directory containing one sub-folder per policy.
        title:       Optional plot title.
        save_path:   Destination file for the figure.  When *None* the figure is
                     displayed interactively via ``plt.show()``.
        window:      Rolling-window width used for smoothing (episodes).
    """
    _POLICY_COLORS: dict[str, str] = {
        "c-tec": "#2eb88a",
        "random": "#808080",
        "rnd": "#4c72b0",
    }
    _fallback_colors = sns.color_palette()

    results_dir = Path(results_dir)

    all_records: list[dict] = []
    palette: dict[str, str] = {}
    fallback_idx = 0

    for policy in policies:
        policy_dir = results_dir / policy
        if not policy_dir.is_dir():
            continue

        # Discover seed files: flat metrics*.json first, then sub-directory seeds.
        seed_files: list[Path] = sorted(policy_dir.glob("eval_metrics*.json"))
        if not seed_files:
            seed_files = sorted(policy_dir.glob("*/eval_metrics.json"))
        if not seed_files:
            continue

        for seed_idx, path in enumerate(seed_files):
            with open(path) as fh:
                data = json.load(fh)

            df = pd.DataFrame(data)
            # Smooth each seed independently so the mean/std seaborn computes
            # is across seeds (not across episodes within one seed).
            smoothed = (
                df["episode_coverage"]
                .astype(float)
                .rolling(window, min_periods=1)
                .mean()
            )
            for step, val in zip(df["total_steps"], smoothed):
                all_records.append(
                    {
                        "total_steps": step,
                        "episode_coverage": val,
                        "policy": policy,
                        "seed": seed_idx,
                    }
                )

        color = _POLICY_COLORS.get(policy)
        if color is None:
            color = _fallback_colors[fallback_idx % len(_fallback_colors)]
            fallback_idx += 1
        palette[policy] = color

    if not all_records:
        return

    df_plot = pd.DataFrame(all_records)

    fig, ax = plt.subplots(figsize=(8, 5))

    # estimator="mean" aggregates the smoothed values across seeds at each
    # x value.  errorbar="sd" draws the ±1 std band across seeds.
    # The "seed" column ensures seaborn treats each seed as a separate unit
    # rather than concatenating them into one continuous line.
    sns.lineplot(
        data=df_plot,
        x="total_steps",
        y="episode_coverage",
        hue="policy",
        units="seed",
        estimator="mean",
        errorbar="sd",
        palette=palette,
        linewidth=2,
        ax=ax,
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment Step", fontsize=12)
    ax.set_ylabel("# Visited States", fontsize=12)
    ax.legend(title=None, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_reached_states(
    train_logger,
    save_path: str | Path,
    window: int = 10,
    title: str = "Reached States Over Training",
) -> None:
    """Plot the number of reached (visited) states per episode against total env timesteps.

    Uses ``sns.lineplot`` over a smoothed coverage series so the trend is
    easy to read.  Raw per-episode values are shown as a faded scatter.

    Args:
        train_logger: A :class:`MetricsLogger` whose history contains
                      ``total_steps`` and ``episode_coverage`` per episode.
        save_path:    Destination file for the figure.
        window:       Rolling-window width used for smoothing (episodes).
        title:        Plot title.
    """
    total_steps = train_logger.get_series("total_steps")
    episode_coverage = train_logger.get_series("episode_coverage")

    if not total_steps or not episode_coverage:
        return

    df = pd.DataFrame(
        {"total_steps": total_steps, "episode_coverage": episode_coverage}
    )

    # Smooth the coverage curve
    df["smoothed"] = df["episode_coverage"].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw values as faded scatter
    sns.scatterplot(
        data=df,
        x="total_steps",
        y="episode_coverage",
        alpha=0.25,
        color="#2eb88a",
        s=15,
        legend=False,
        ax=ax,
    )

    # Smoothed line
    sns.lineplot(
        data=df,
        x="total_steps",
        y="smoothed",
        color="#2eb88a",
        linewidth=2,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Environment Timesteps", fontsize=12)
    ax.set_ylabel("# Reached States", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_coverage(
    trajectory_buffer,
    save_path: str | Path,
    label: str = "",
    title: str = "Visited States per Episode",
    color: str | None = None,
    subsample: int = 10,
) -> None:
    """Plot per-episode visited-state curves with mean ± std bands.

    Each trajectory's ``cell_covered`` list already records the running
    count of unique cells at each step.  ``sns.lineplot`` aggregates
    across all episodes in the buffer, producing a mean line with a ±1
    std shaded band.

    Args:
        trajectory_buffer: A :class:`TrajectoryBuffer` containing the
            collected trajectories (in chronological order).
        save_path:  Destination file for the figure.
        label:      Legend label for the curve (e.g. ``"C-TeC"``).
        title:      Plot title.
        color:      Matplotlib colour for the line.  When *None* a
                    default is chosen.
        subsample:  Record one data point every *subsample* steps.
    """
    records: list[dict] = []
    for traj_idx, trajectory in enumerate(trajectory_buffer.trajectories):
        for i in range(len(trajectory.cell_covered)):
            if i % subsample == 0 or i == len(trajectory.cell_covered) - 1:
                records.append(
                    {
                        "step": i,
                        "visited_states": trajectory.cell_covered[i],
                        "episode": traj_idx,
                    }
                )

    if not records:
        return

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="step",
        y="visited_states",
        estimator="mean",
        errorbar="sd",
        color=color or "#2eb88a",
        linewidth=2,
        label=label or None,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment Step", fontsize=12)
    ax.set_ylabel("# Visited States", fontsize=12)
    if label:
        ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_coverage_comparison(
    buffers: dict[str, "TrajectoryBuffer"],
    title: str = "Visited States per Episode",
    save_path: str | Path | None = None,
    subsample: int = 10,
) -> None:
    """Plot per-episode visited-state curves for multiple policies.

    For each policy the function iterates over every trajectory in the
    buffer and reads its ``cell_covered`` list (cumulative unique cells
    at each within-episode step).  ``sns.lineplot`` then aggregates
    across episodes, producing a **mean** line with a **±1 std** shaded
    band — mirroring the paper's "# Visited States vs Environment Step"
    figure.

    Args:
        buffers:    Mapping from policy name (e.g. ``"c-tec"``) to its
                    :class:`TrajectoryBuffer`.
        title:      Plot title.
        save_path:  Destination file.  When *None* the figure is shown
                    interactively.
        subsample:  Record one data point every *subsample* steps to
                    keep the DataFrame manageable for long episodes.
    """
    _POLICY_COLORS: dict[str, str] = {
        "c-tec": "#2eb88a",
        "random": "#808080",
        "rnd": "#4c72b0",
    }
    _fallback_colors = sns.color_palette()

    all_records: list[dict] = []
    palette: dict[str, str] = {}
    fallback_idx = 0

    for policy, buf in buffers.items():
        for traj_idx, trajectory in enumerate(buf.trajectories):
            for i in range(len(trajectory.cell_covered)):
                if i % subsample == 0 or i == len(trajectory.cell_covered) - 1:
                    all_records.append(
                        {
                            "step": i,
                            "visited_states": trajectory.cell_covered[i],
                            "policy": policy,
                            "episode": traj_idx,
                        }
                    )

        color = _POLICY_COLORS.get(policy)
        if color is None:
            color = _fallback_colors[fallback_idx % len(_fallback_colors)]
            fallback_idx += 1
        palette[policy] = color

    if not all_records:
        return

    df = pd.DataFrame(all_records)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x="step",
        y="visited_states",
        hue="policy",
        estimator="mean",
        errorbar="sd",
        palette=palette,
        linewidth=2,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment Step", fontsize=12)
    ax.set_ylabel("# Visited States", fontsize=12)
    ax.legend(title=None, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    from c_tec.buffer import TrajectoryBuffer

    path = Path(__file__).parent.parent.parent / "results"
    # Load saved buffers and compare
    buffers = {}
    for policy in ["random", "c-tec", "rnd"]:
        for candidate in [
            path / policy / "trajectory_buffer.pkl",
            path / policy / "eval" / "trajectory_buffer.pkl",
        ]:
            if candidate.exists():
                buffers[policy] = TrajectoryBuffer.load(candidate)
                break
    if buffers:
        plot_cumulative_coverage_comparison(
            buffers=buffers, title="C-TeC vs Random", subsample=1
        )
