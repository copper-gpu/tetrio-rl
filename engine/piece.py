# Implements Tetris pieces and Super Rotation System (SRS)

import numpy as np

class Piece:
    def __init__(self, type_):
        self.type = type_

    def shape(self, rotation):
        return PIECES[self.type][rotation % 4]

# Tetrimino definitions (0–3: 0°, R, 180°, L)
PIECES = {
    'I': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0]]),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]])
    ],
    'O': [
        np.array([[1, 1],
                  [1, 1]])
    ] * 4,
    'T': [
        np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 0, 0]]),
        np.array([[0, 1, 0],
                  [0, 1, 1],
                  [0, 1, 0]]),
        np.array([[0, 0, 0],
                  [1, 1, 1],
                  [0, 1, 0]]),
        np.array([[0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0]])
    ],
    'S': [
        np.array([[0, 1, 1],
                  [1, 1, 0],
                  [0, 0, 0]]),
        np.array([[0, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1]]),
        np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 1, 0]]),
        np.array([[1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0]])
    ],
    'Z': [
        np.array([[1, 1, 0],
                  [0, 1, 1],
                  [0, 0, 0]]),
        np.array([[0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 0]]),
        np.array([[0, 0, 0],
                  [1, 1, 0],
                  [0, 1, 1]]),
        np.array([[0, 1, 0],
                  [1, 1, 0],
                  [1, 0, 0]])
    ],
    'J': [
        np.array([[1, 0, 0],
                  [1, 1, 1],
                  [0, 0, 0]]),
        np.array([[0, 1, 1],
                  [0, 1, 0],
                  [0, 1, 0]]),
        np.array([[0, 0, 0],
                  [1, 1, 1],
                  [0, 0, 1]]),
        np.array([[0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0]])
    ],
    'L': [
        np.array([[0, 0, 1],
                  [1, 1, 1],
                  [0, 0, 0]]),
        np.array([[0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 1]]),
        np.array([[0, 0, 0],
                  [1, 1, 1],
                  [1, 0, 0]]),
        np.array([[1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0]])
    ]
}

# Super Rotation System (SRS) Wall Kick Data
SRS_KICKS = {
    'I': {
        (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    },
    'J': {}, 'L': {}, 'S': {}, 'T': {}, 'Z': {}, 'O': {}
}

# Apply same kick data for JLSTZ pieces (simplified standard kicks)
JLSTZ_KICKS = [
    (0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)
]
for piece in ['J', 'L', 'S', 'T', 'Z']:
    for frm in range(4):
        to = (frm + 1) % 4
        SRS_KICKS[piece][(frm, to)] = JLSTZ_KICKS
        SRS_KICKS[piece][(to, frm)] = [(-dx, -dy) for dx, dy in JLSTZ_KICKS]

# O piece does not rotate around a center, so all kicks are (0, 0)
for frm in range(4):
    to = (frm + 1) % 4
    SRS_KICKS['O'][(frm, to)] = [(0, 0)]
    SRS_KICKS['O'][(to, frm)] = [(0, 0)]
