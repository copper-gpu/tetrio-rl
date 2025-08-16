import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.agent import HeuristicAgent
from ai.trainer import DQNAgent, EPS_DECAY
from engine.core import TetrisBoard
from engine.piece import Piece


class DummyNet(torch.nn.Module):
    """Network returning zero scores for all actions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.size(0), 40), device=x.device)


def _valid_action(board_state, piece, x, rot) -> bool:
    """Check that placing ``piece`` at ``(x, rot)`` is collision free."""
    sim = TetrisBoard()
    sim.board = board_state.copy()
    sim.spawn(piece)
    sim.piece_rotation = rot
    sim.piece_x = x
    shape = piece.shape(rot)
    return not sim.check_collision(shape, x, 0)


def test_heuristic_agent_choose_action_valid():
    agent = HeuristicAgent()
    env = TetrisBoard()
    piece = Piece('I')
    env.spawn(piece)
    board, p, px, py, pr = env.get_state()
    x, rot = agent.choose_action(board, p, px, py, pr)

    assert 0 <= rot < 4
    assert _valid_action(board, p, x, rot)


def test_dqn_agent_choose_action_valid():
    random.seed(0)
    agent = DQNAgent(DummyNet())
    agent.steps = EPS_DECAY  # ensure greedy action

    env = TetrisBoard()
    piece = Piece('I')
    env.spawn(piece)
    board, p, px, py, pr = env.get_state()
    x, rot = agent.choose_action(board, p, px, py, pr)

    assert 0 <= rot < 4
    assert _valid_action(board, p, x, rot)


def test_evaluate_board_penalizes_holes():
    agent = HeuristicAgent()
    board_flat = np.zeros((20, 10), dtype=int)
    board_flat[-1, :] = 1

    board_hole = board_flat.copy()
    board_hole[-2, 5] = 1
    board_hole[-1, 5] = 0

    assert agent.evaluate_board(board_flat) > agent.evaluate_board(board_hole)


def test_evaluate_board_prefers_lower_height():
    agent = HeuristicAgent()
    empty = np.zeros((20, 10), dtype=int)
    stack = np.zeros((20, 10), dtype=int)
    stack[10:, 0] = 1

    assert agent.evaluate_board(empty) > agent.evaluate_board(stack)
