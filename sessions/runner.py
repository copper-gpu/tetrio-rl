from __future__ import annotations

from ai.trainer import DQNTrainer


class SessionRunner:
    """Thin wrapper around :class:`DQNTrainer`."""

    def __init__(self, save_dir: str = "models", seed: int | None = None) -> None:
        self.trainer = DQNTrainer(save_dir=save_dir, seed=seed)
        self._running = False

    def start(self, steps: int = 100_000) -> None:
        """Start a training session for ``steps`` environment steps."""
        if self._running:
            raise RuntimeError("Session already running")
        self._running = True
        try:
            self.trainer.train(total_steps=steps)
        finally:
            self._running = False

    def stop(self) -> None:
        """Mark the session as stopped. (No mid‑episode interrupt.)"""
        self._running = False
