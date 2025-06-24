# engine/bag.py
"""Seven‑bag randomiser for authentic Tetris piece spawning.

Each bag contains exactly one of every tetromino type (I, O, T, S, Z, J, L).
When the bag is empty, it is reshuffled to create a new bag, matching the
Tetris Guideline randomisation used by TETR.IO.

Usage
-----
```python
from engine.bag import SevenBag
bag = SevenBag(seed=42)
next_piece = bag.next_piece()  # → engine.piece.Piece instance
```
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterator

from engine.piece import Piece, PIECES  # type: ignore

__all__ = ["SevenBag"]


class SevenBag(Iterator[Piece]):
    """Iterator that yields tetromino pieces in 7‑bag order."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._bag: Deque[str] = deque()
        self._refill()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _refill(self) -> None:
        tetrominoes = list(PIECES.keys())  # ["I", "O", "T", "S", "Z", "J", "L"]
        self._rng.shuffle(tetrominoes)
        self._bag.extend(tetrominoes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def next_piece(self) -> Piece:
        """Return the next :class:`engine.piece.Piece` from the bag."""
        if not self._bag:
            self._refill()
        return Piece(self._bag.popleft())

    # Iterator protocol --------------------------------------------------
    def __iter__(self) -> "SevenBag":  # noqa: D401
        return self

    def __next__(self) -> Piece:  # noqa: D401
        return self.next_piece()
