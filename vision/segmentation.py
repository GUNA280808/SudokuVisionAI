"""
vision/segmentation.py

Splits the warped 9x9 Sudoku square into 81 individual cell images,
trims cell borders (to remove grid lines), and flags empty cells.
"""

import cv2
import numpy as np

GRID_SIZE = 9


def split_into_cells(warped_gray, grid_size=GRID_SIZE):
    """Returns a list of 81 cell images (row-major order)."""
    h, w = warped_gray.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = warped_gray[y1:y2, x1:x2]
            cells.append(trim_cell_border(cell))
    return cells


def trim_cell_border(cell, margin_ratio=0.16):
    """Crop a margin off each side to remove grid line artifacts."""
    h, w = cell.shape[:2]
    my, mx = int(h * margin_ratio), int(w * margin_ratio)
    trimmed = cell[my:h - my, mx:w - mx]
    if trimmed.size == 0:
        return cell
    return trimmed


def is_cell_empty(cell, white_pixel_threshold=0.035):
    """
    Determines whether a cell contains a digit or is blank, based on the
    proportion of 'foreground' (thresholded white) pixels in the very
    center of the cell (well clear of any residual grid-line artifacts
    that can survive perspective warping near the border).
    """
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground is white regardless of Otsu's polarity choice
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Clean up thin line artifacts with a light erosion before measuring
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    h, w = thresh.shape
    center = thresh[int(h * 0.28):int(h * 0.72), int(w * 0.28):int(w * 0.72)]
    if center.size == 0:
        return True

    fg_ratio = np.count_nonzero(center) / center.size
    return fg_ratio < white_pixel_threshold


def normalize_digit_glyph(thresholded_28, canvas_size=28, target_size=20):
    """
    Takes an already-thresholded 28x28 image (white digit on black bg) and
    re-centers/re-scales the digit's own bounding box to a canonical size,
    the same way MNIST itself was normalized. This is critical: without it,
    a model trained on digits that happen to fill (say) 45% of the frame
    will generalize poorly to real-world crops where the digit fills 90%
    of the frame (or vice-versa), regardless of how carefully upstream
    cropping ratios are tuned. Centering on the glyph's own bounding box
    makes the model's job scale-invariant.
    """
    ys, xs = np.where(thresholded_28 > 127)
    if len(ys) == 0:
        return thresholded_28

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    glyph = thresholded_28[y0:y1 + 1, x0:x1 + 1]

    gh, gw = glyph.shape
    scale = target_size / max(gh, gw)
    new_h, new_w = max(1, int(round(gh * scale))), max(1, int(round(gw * scale)))
    resized_glyph = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size), dtype="uint8")
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_glyph
    return canvas


def prepare_cell_for_cnn(cell, size=28):
    """Resize + threshold + canonically normalize a cell image for CNN inference."""
    resized = cv2.resize(cell, (size, size), interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    normalized_glyph = normalize_digit_glyph(thresh, canvas_size=size, target_size=20)
    normalized = normalized_glyph.astype("float32") / 255.0
    return normalized.reshape(1, size, size, 1)
