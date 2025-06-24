# Main structure of TETR.IO AI Trainer

# --- File Structure ---
# project/
# ├── engine/              # Tetris game simulation
# │   ├── core.py          # Board logic
# │   └── piece.py         # Piece logic (Tetris SRS)
# ├── ai/
# │   ├── agent.py         # AI model interface
# │   └── trainer.py       # RL training loop
# ├── sessions/
# │   ├── runner.py        # Launch and manage sessions
# │   └── feedback.py      # Evaluation and logging
# ├── models/             # Saved models & restore points
# └── main.py              # CLI entry point

# --- engine/core.py ---
# Implements Tetris board state and operations (place, clear lines, game over)

import numpy as np
from engine.piece import Piece, SRS_KICKS

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

class TetrisBoard:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.active_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.piece_rotation = 0
        self.game_over = False

    def spawn(self, piece: Piece):
        self.active_piece = piece
        self.piece_x = 3
        self.piece_y = 0
        self.piece_rotation = 0
        if self.check_collision(piece.shape(self.piece_rotation), self.piece_x, self.piece_y):
            self.game_over = True

    def move(self, dx, dy):
        if not self.active_piece:
            return False
        if not self.check_collision(self.active_piece.shape(self.piece_rotation), self.piece_x + dx, self.piece_y + dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        return False

    def rotate(self, direction):
        if not self.active_piece:
            return
        new_rotation = (self.piece_rotation + direction) % 4
        kicks = SRS_KICKS[self.active_piece.type][(self.piece_rotation, new_rotation)]
        for dx, dy in kicks:
            if not self.check_collision(self.active_piece.shape(new_rotation), self.piece_x + dx, self.piece_y + dy):
                self.piece_rotation = new_rotation
                self.piece_x += dx
                self.piece_y += dy
                return True
        return False

    def hard_drop(self):
        while self.move(0, 1):
            pass
        self.lock_piece()

    def lock_piece(self):
        shape = self.active_piece.shape(self.piece_rotation)
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + y, self.piece_x + x] = 1
        self.clear_lines()
        self.active_piece = None

    def clear_lines(self):
        new_board = self.board[~np.all(self.board == 1, axis=1)]
        lines_cleared = BOARD_HEIGHT - len(new_board)
        if lines_cleared > 0:
            self.board = np.vstack((np.zeros((lines_cleared, BOARD_WIDTH), dtype=int), new_board))

    def check_collision(self, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    px, py = x + j, y + i
                    if px < 0 or px >= BOARD_WIDTH or py >= BOARD_HEIGHT:
                        return True
                    if py >= 0 and self.board[py, px]:
                        return True
        return False

    def get_state(self):
        return self.board.copy(), self.active_piece, self.piece_x, self.piece_y, self.piece_rotation

