# Tetrio RL

This repository provides a small reinforcement learning trainer for the TETR.IO puzzle game. It includes a minimal game engine and a DQN based agent implementation.

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Training

To train a new agent, run:

```bash
python main.py train
```

This will start training with the default parameters.

## Code structure

```
engine/   - Tetris board and piece implementation
ai/       - agent interface and DQN trainer
sessions/ - session management utilities
main.py   - command line entry point
```
