import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.bag import SevenBag
from engine.piece import PIECES


def test_cycle_contains_all_pieces():
    bag = SevenBag(seed=0)
    pieces = [bag.next_piece().type for _ in range(7)]
    assert sorted(pieces) == sorted(PIECES.keys())


def test_second_cycle_contains_all_pieces():
    bag = SevenBag(seed=1)
    _ = [bag.next_piece() for _ in range(7)]
    pieces = [bag.next_piece().type for _ in range(7)]
    assert sorted(pieces) == sorted(PIECES.keys())


def test_deterministic_order_with_seed():
    bag1 = SevenBag(seed=42)
    bag2 = SevenBag(seed=42)
    seq1 = [bag1.next_piece().type for _ in range(14)]
    seq2 = [bag2.next_piece().type for _ in range(14)]
    assert seq1 == seq2
