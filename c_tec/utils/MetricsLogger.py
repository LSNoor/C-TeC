"""Lightweight per-episode metrics logger."""

from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    """Accumulates per-episode stats as a list of dicts."""

    def __init__(self):
        self.history: list[dict] = []

    def log(self, **kwargs):
        self.history.append(kwargs)

    def recent(self, n: int) -> dict[str, float]:
        """Mean of numeric fields over the last n entries."""
        recent = self.history[-n:]
        keys = [k for k in recent[0] if isinstance(recent[0][k], (int, float))]
        return {k: sum(d[k] for d in recent) / len(recent) for k in keys}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_series(self, key: str) -> list:
        return [d[key] for d in self.history if key in d]