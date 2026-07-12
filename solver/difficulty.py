"""
solver/difficulty.py

Computes a weighted difficulty score based on:
  - Number of pre-filled cells (fewer clues = harder)
  - Number of recursive backtracking steps taken to solve
  - Solving time

Classifies the puzzle into Easy / Medium / Hard / Expert.
"""

from solver.validator import count_filled_cells

# Tunable weights
W_CLUES = 0.4
W_STEPS = 0.4
W_TIME = 0.2

# Normalization anchors (empirically reasonable for a 9x9 grid)
MAX_EXPECTED_STEPS = 5000
MAX_EXPECTED_TIME = 1.0  # seconds


def compute_difficulty(grid, steps, time_seconds):
    filled = count_filled_cells(grid)

    # Clue score: fewer clues -> higher difficulty contribution (0-1)
    clue_score = max(0.0, min(1.0, (36 - filled) / 28))  # 36 clues -> 0, 8 clues -> 1

    step_score = max(0.0, min(1.0, steps / MAX_EXPECTED_STEPS))
    time_score = max(0.0, min(1.0, time_seconds / MAX_EXPECTED_TIME))

    weighted_score = (
        W_CLUES * clue_score +
        W_STEPS * step_score +
        W_TIME * time_score
    )

    # Scale to a friendly 0-100 display score
    score_100 = round(weighted_score * 100, 1)

    if score_100 < 25:
        label = "Easy"
    elif score_100 < 50:
        label = "Medium"
    elif score_100 < 75:
        label = "Hard"
    else:
        label = "Expert"

    return {
        "label": label,
        "score": score_100,
        "filled_cells": filled,
        "backtracking_steps": steps,
        "time_seconds": round(time_seconds, 4),
    }
