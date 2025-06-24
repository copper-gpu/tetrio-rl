# ai/agent.py
# Defines the AI agent interface and a simple heuristic-based baseline

import numpy as np

class BaseAgent:
    def __init__(self):
        pass

    def choose_action(self, board_state, piece, piece_x, piece_y, piece_rotation):
        """
        Should return a tuple (target_x, target_rotation) for the active piece.
        """
        raise NotImplementedError


class HeuristicAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def evaluate_board(self, board):
        """Basic heuristics: lower total height and more cleared lines are better"""
        heights = board.shape[0] - np.argmax(board[::-1], axis=0)
        heights[np.all(board == 0, axis=0)] = 0
        aggregate_height = np.sum(heights)
        holes = np.sum((board == 0) & (np.cumsum(board != 0, axis=0) > 0))
        bumpiness = np.sum(np.abs(np.diff(heights)))
        return -0.5 * aggregate_height - 0.7 * holes - 0.3 * bumpiness

    def choose_action(self, board_state, piece, piece_x, piece_y, piece_rotation):
        """
        Brute-force simulation of all placements. Chooses the best-scoring one.
        """
        from engine.core import TetrisBoard

        best_score = float('-inf')
        best_action = (piece_x, piece_rotation)

        for rot in range(4):
            shape = piece.shape(rot)
            width = shape.shape[1]

            for x in range(-2, 10 - width + 2):
                sim = TetrisBoard()
                sim.board = board_state.copy()
                sim.spawn(piece)
                sim.piece_rotation = rot
                sim.piece_x = x
                sim.piece_y = 0

                if sim.check_collision(shape, x, 0):
                    continue

                while not sim.check_collision(shape, sim.piece_x, sim.piece_y + 1):
                    sim.piece_y += 1

                sim.lock_piece()
                score = self.evaluate_board(sim.board)

                if score > best_score:
                    best_score = score
                    best_action = (x, rot)

        return best_action
