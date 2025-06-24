import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from engine.core import TetrisBoard, BOARD_HEIGHT, BOARD_WIDTH
from engine.piece import Piece


def test_clear_single_line():
    board = TetrisBoard()
    board.board[-1] = 1
    board.clear_lines()
    assert board.board.shape == (BOARD_HEIGHT, BOARD_WIDTH)
    assert not board.board[-1].any()


def test_clear_multiple_lines():
    board = TetrisBoard()
    board.board[-1] = 1
    board.board[-2] = 1
    board.clear_lines()
    assert not board.board[-1].any()
    assert not board.board[-2].any()


def test_collision_with_walls():
    board = TetrisBoard()
    piece = Piece('O')
    assert board.check_collision(piece.shape(0), -1, 0)
    assert board.check_collision(piece.shape(0), BOARD_WIDTH, 0)


def test_collision_with_stack():
    board = TetrisBoard()
    board.board[5, 5] = 1
    piece = Piece('O')
    assert board.check_collision(piece.shape(0), 5, 5)


def test_no_collision_valid_position():
    board = TetrisBoard()
    piece = Piece('O')
    assert not board.check_collision(piece.shape(0), 4, 0)
