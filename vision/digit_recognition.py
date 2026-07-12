"""
vision/digit_recognition.py

Loads the trained CNN (models/cnn_model.h5) and predicts the digit (1-9)
contained in a single Sudoku cell image. Empty-cell detection happens
upstream in segmentation.py, so this module only ever sees non-empty cells.
"""

import os
import numpy as np

from vision.segmentation import prepare_cell_for_cnn

_model = None
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "cnn_model.h5")


def _load_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"CNN model not found at {_MODEL_PATH}. "
                "Run `python train_model.py` first to train and save it."
            )
        _model = load_model(_MODEL_PATH)
    return _model


def predict_digit(cell_image):
    """
    cell_image: grayscale numpy array (already trimmed of the grid border).
    Returns (digit:int[1-9], confidence:float[0-1]).
    """
    model = _load_model()
    tensor = prepare_cell_for_cnn(cell_image)
    predictions = model.predict(tensor, verbose=0)[0]

    # Model outputs 9 classes representing digits 1-9 (see train_model.py)
    digit = int(np.argmax(predictions)) + 1
    confidence = float(np.max(predictions))
    return digit, confidence


def recognize_grid(cells, empty_flags):
    """
    cells: list of 81 grayscale cell images
    empty_flags: list of 81 booleans (True = empty cell)

    Returns (grid: 9x9 list of ints [0 = blank], confidences: 9x9 list of floats,
             recognized_count: int)
    """
    grid = [[0] * 9 for _ in range(9)]
    confidences = [[0.0] * 9 for _ in range(9)]
    recognized_count = 0

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        if empty_flags[idx]:
            continue
        digit, conf = predict_digit(cell)
        grid[row][col] = digit
        confidences[row][col] = conf
        recognized_count += 1

    return grid, confidences, recognized_count
