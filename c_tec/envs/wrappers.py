import minigrid  # noqa

import gymnasium as gym
import numpy as np
from collections import defaultdict


class PositionObsWrapper(gym.ObservationWrapper):
    """
    Replaces MiniGrid's Dict observation with a flat vector:
        [x, y, direction_onehot(4)] -> R^6

    This matches the paper's "prior knowledge" setup (Section 6.2)
    where the future state is restricted to position components.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 2 (x, y) + 4 (direction one-hot)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=max(env.unwrapped.width, env.unwrapped.height),
            shape=(6,),
            dtype=np.float32,
        )

    def observation(self, obs: dict) -> np.ndarray:
        pos = self.unwrapped.agent_pos
        direction = self.unwrapped.agent_dir
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0
        return np.array([pos[0], pos[1], *dir_onehot], dtype=np.float32)


class OneHotActionWrapper(gym.ActionWrapper):
    """
    Provides a method to get the one-hot encoding of a discrete action
    for use in the contrastive model's φ(s, a) encoder.

    Does not change the action space — actions are still integers
    passed to MiniGrid. The one-hot is only used by C-TeC's encoders.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_actions = env.action_space.n

    def action(self, act: int) -> int:
        return act  # pass through to MiniGrid unchanged

    def action_to_onehot(self, act: int) -> np.ndarray:
        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[act] = 1.0
        return onehot


class StateCoverageTracker(gym.Wrapper):
    """
    Tracks unique (x, y) positions visited across all episodes.

    Used as the primary evaluation metric, matching the paper's
    "number of unique discretized states" (Section 6.2).
    """

    def __init__(self, env: gym.Env, fixed_seed: int | None = None):
        super().__init__(env)
        self.episode_visited: set[tuple[int, int]] = set()
        self.reached_count: dict[tuple[int, int], int] = defaultdict(lambda: 0)
        self._reachable: set[tuple[int, int]] | None = None
        self._fixed_seed = fixed_seed

    def reset(self, **kwargs):
        # Always reset with the same seed -> identical grid layout
        if self._fixed_seed is not None:
            kwargs["seed"] = self._fixed_seed
        obs, info = self.env.reset(**kwargs)
        # Clear per-episode tracking on each reset
        self.episode_visited = set()
        # Compute reachable once
        if self._reachable is None:
            self._reachable = self.compute_reachable()
            for pos in self._reachable:
                self.reached_count[pos] = self.reached_count.get(pos, 0)
        self._record_position()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_position()
        info["episode_coverage"] = len(self.episode_visited)
        info["episode_coverage_pct"] = len(self.episode_visited) / self.n_reachable
        if terminated or truncated:
            info["reached_count"] = self.reached_count
        return obs, reward, terminated, truncated, info

    def _record_position(self):
        pos = tuple(self.unwrapped.agent_pos)

        if pos not in self.episode_visited:
            self.reached_count[pos] = self.reached_count.get(pos, 0) + 1

        self.episode_visited.add(pos)

    @property
    def n_reachable(self) -> int:
        """Count reachable (non-wall) cells by inspecting the grid."""
        self._reachable = self.compute_reachable()
        return len(self._reachable)

    def compute_reachable(self) -> set[tuple[int, int]]:
        """Scan the MiniGrid grid for non-wall, non-None cells."""
        reachable = set()
        grid = self.unwrapped.grid
        for x in range(self.unwrapped.width):
            for y in range(self.unwrapped.height):
                cell = grid.get(x, y)
                if cell is None:
                    # Empty cell — agent can stand here
                    reachable.add((x, y))
                elif cell.type == "goal":
                    reachable.add((x, y))
                elif cell.type == "door":
                    reachable.add((x, y))
                # walls, etc. are excluded
        return reachable


class NoGoalTermination(gym.Wrapper):
    """
    Prevents the episode from terminating when the agent reaches the goal.

    MiniGrid sets ``terminated=True`` upon goal entry, which ends the
    episode. This wrapper overrides that signal so exploration can continue
    past the goal cell without an episode reset.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Suppress goal-triggered termination; only truncation (time limit)
        # is allowed to end the episode.
        terminated = False
        return obs, reward, terminated, truncated, info


def make_env(
    env_id: str = "MiniGrid-FourRooms-v0",
    seed: int | None = None,
    max_steps: int | None = None,
    *args,
    **kwargs,
) -> gym.Env:
    """
    Factory for a fully wrapped MiniGrid environment.

    Args:
        env_id:    Gymnasium environment ID.
        seed:      Optional seed for deterministic resets.
        max_steps: Maximum steps per episode (truncation limit).
                   If None, the environment's built-in default is used.

    Wrapper order matters:
        MiniGrid -> NoGoalTermination -> PositionObs -> OneHotAction -> CoverageTracker
    """
    if max_steps is not None:
        # Override both Gymnasium's TimeLimit wrapper AND MiniGrid's internal
        # step counter (FourRoomsEnv defaults to max_steps=100 internally).
        kwargs["max_episode_steps"] = max_steps
        kwargs["max_steps"] = max_steps
    env = gym.make(env_id, *args, **kwargs)
    env = NoGoalTermination(env)
    env = PositionObsWrapper(env)
    env = OneHotActionWrapper(env)
    env = StateCoverageTracker(env)
    if seed is not None:
        env.reset(seed=seed)
    return env
