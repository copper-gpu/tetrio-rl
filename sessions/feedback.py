from __future__ import annotations

import csv
from pathlib import Path

from ai.trainer import DQNTrainer


def evaluate(trainer: DQNTrainer, games: int = 5) -> float:
    """Return the average reward of ``trainer`` over ``games`` games."""
    return trainer._evaluate(games)


def log_score(save_dir: str, step: int, value: float) -> None:
    """Append ``step`` and evaluation ``value`` to ``feedback.csv`` in ``save_dir``."""
    path = Path(save_dir) / "feedback.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "value"])
        writer.writerow([step, value])
