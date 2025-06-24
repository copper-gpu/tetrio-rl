"""Command‑line interface for the TETR.IO AI Trainer.

Usage
-----
Train a brand‑new model for 50 k steps (with deterministic 7‑bag order):

    python main.py train --steps 50000 --seed 42

The script currently exposes a **single** sub‑command (`train`). Additional
commands (e.g. `eval`, `swap`, `resume`) can be added later without touching
the outer structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai.trainer import DQNTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Sub‑command callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_train(args: argparse.Namespace) -> None:  # noqa: D401
    """Train a DQN model from scratch (or continue training)."""

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = DQNTrainer(save_dir=str(save_dir), seed=args.seed)
    trainer.train(total_steps=args.steps)


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:  # noqa: D401
    """Construct and return the program's argument parser."""

    parser = argparse.ArgumentParser(description="TETR.IO AI Trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # train -----------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a new DQN model")
    p_train.add_argument("--steps", type=int, default=100_000, help="Total environment steps")
    p_train.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p_train.add_argument("--save-dir", default="models", help="Directory for checkpoints")
    p_train.set_defaults(func=_cmd_train)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry‑point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: D401
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
