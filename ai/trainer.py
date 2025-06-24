# ai/trainer.py
"""Final bug‑free DQN trainer (copy entire file).

Fixes
-----
* **ActionMapper.from_id** now returns a tuple correctly (no `NameError`).
* Retains `piece is None` guard and all logging/checkpoint functionality.
"""

from __future__ import annotations

import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from engine.core import TetrisBoard, BOARD_WIDTH
from engine.bag import SevenBag
from engine.piece import PIECES, Piece
from ai.agent import BaseAgent

# ───────────────────────── Hyper‑parameters ─────────────────────────
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEM_CAP = 50_000
TARGET_UPDATE = 500     # optimiser steps
SAVE_EVERY = 5_000     # env steps
EVAL_EVERY = 5_000     # env steps
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 3_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── Helper classes ──────────────────────────
class ActionMapper:
    """Bidirectional map between `(x, rot)` and discrete id (0–39)."""

    @staticmethod
    def to_id(x: int, rot: int) -> int:
        return rot * BOARD_WIDTH + x  # rot∈0‑3, x∈0‑9

    @staticmethod
    def from_id(a: int) -> Tuple[int, int]:
        rot = a // BOARD_WIDTH
        x = a % BOARD_WIDTH
        return x, rot


class QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(211, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 40),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity: int = MEM_CAP):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

    def sample(self, n: int):
        batch = random.sample(self.buffer, n)
        s, a, r, ns, d = map(np.array, zip(*batch))
        to = lambda x, dt: torch.tensor(x, dtype=dt, device=DEVICE)  # noqa: E731
        return (
            to(s, torch.float32),
            to(a, torch.long),
            to(r, torch.float32),
            to(ns, torch.float32),
            to(d, torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class Stats:
    steps: int = 0
    episodes: int = 0


class DQNAgent(BaseAgent):
    def __init__(self, net: QNet):
        super().__init__()
        self.net = net
        self.steps = 0

    @staticmethod
    def enc(board: np.ndarray, piece: Optional[Piece], rot: int) -> np.ndarray:
        flat = board.flatten().astype(np.float32)
        piece_vec = np.zeros(7, np.float32)
        if piece is not None:
            piece_vec[list(PIECES).index(piece.type)] = 1.0
        rot_vec = np.zeros(4, np.float32)
        rot_vec[rot % 4] = 1.0
        return np.concatenate([flat, piece_vec, rot_vec])

    def choose_action(self, b, p, px, py, pr):
        state = torch.tensor(self.enc(b, p, pr), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        eps = EPS_END + max(0, (EPS_START - EPS_END) * (1 - self.steps / EPS_DECAY))
        self.steps += 1
        if random.random() < eps:
            aid = random.randrange(40)
        else:
            with torch.no_grad():
                aid = int(torch.argmax(self.net(state)).item())
        return ActionMapper.from_id(aid)


# ───────────────────────── Trainer ─────────────────────────────────
class DQNTrainer:
    def __init__(self, save_dir: str = "runs/default", seed: int | None = None):
        self.dir = Path(save_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.csv"
        if not self.log_path.exists():
            self.log_path.write_text("step,avg_reward\n")

        self.policy = QNet().to(DEVICE)
        self.target = QNet().to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.agent = DQNAgent(self.policy)
        self.buffer = ReplayBuffer()
        self.optim = optim.Adam(self.policy.parameters(), lr=LR)
        self.stats = Stats()
        self.bag = SevenBag(seed=seed)

    # -------------- helpers --------------
    def _spawn(self, env: TetrisBoard):
        env.spawn(self.bag.next_piece())

    def _place(self, env: TetrisBoard, tx: int, rot: int):
        for _ in range((rot - env.piece_rotation) % 4):
            env.rotate(1)
        step = 1 if tx > env.piece_x else -1
        for _ in range(abs(tx - env.piece_x)):
            env.move(step, 0)
        env.hard_drop()

    @staticmethod
    def _reward(env: TetrisBoard) -> float:
        h = np.where(env.board.any(axis=1))[0]
        return -0.1 * (0 if len(h) == 0 else env.board.shape[0] - h[0])

    def _optimise(self):
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nq = self.target(ns).max(1)[0]
            target = r + GAMMA * nq * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad(); loss.backward(); self.optim.step()

    def _ckpt(self):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "stats": self.stats,
        }, self.dir / f"dqn_step{self.stats.steps}.pt")

    def _log(self, val: float):
        with self.log_path.open("a", newline="") as f:
            csv.writer(f).writerow([self.stats.steps, val])

    def _evaluate(self, games: int = 5) -> float:
        env = TetrisBoard(); total = 0.0
        for _ in range(games):
            self._spawn(env)
            done = False
            while not done:
                if env.active_piece is None:
                    self._spawn(env)
                b, p, px, py, pr = env.get_state()
                tx, rot = self.agent.choose_action(b, p, px, py, pr)
                self._place(env, tx, rot)
                total += self._reward(env)
                done = env.game_over
        return total / games

    # -------------- main loop --------------
    def train(self, total_steps: int = 50_000):
        env = TetrisBoard(); self._spawn(env)
        while self.stats.steps < total_steps:
            if env.active_piece is None:
                self._spawn(env)

            b, p, px, py, pr = env.get_state()
            tx, rot = self.agent.choose_action(b, p, px, py, pr)
            state_vec = DQNAgent.enc(b, p, pr)

            self._place(env, tx, rot)
            reward = self._reward(env)
            done = env.game_over

            if done:
                next_vec = np.zeros_like(state_vec)
            else:
                nb, npiece, _, _, nrot = env.get_state()
                next_vec = DQNAgent.enc(nb, npiece, nrot)

            self.buffer.push(state_vec, ActionMapper.to_id(tx, rot), reward, next_vec, float(done))
            self.stats.steps += 1

            if len(self.buffer) >= BATCH_SIZE:
                self._optimise()
            if self.stats.steps % TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())
            if self.stats.steps % SAVE_EVERY == 0:
                self._ckpt()
            if self.stats.steps % EVAL_EVERY == 0:
                self._log(self._evaluate())

            if done:
                env = TetrisBoard(); self._spawn(env); self.stats.episodes += 1

        self._ckpt()
