"""
vision/contour.py

Finds the largest contour in the preprocessed binary image (assumed to be
the outer border of the Sudoku grid) and reduces it to its four corner points.
"""

import cv2
import numpy as np


class GridNotFoundError(Exception):
    """Raised when no plausible Sudoku grid contour can be located."""
    pass


def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise GridNotFoundError("No contours found in the image.")

    largest = max(contours, key=cv2.contourArea)

    # Reject if the found contour is implausibly small relative to the image
    image_area = binary_image.shape[0] * binary_image.shape[1]
    if cv2.contourArea(largest) < 0.05 * image_area:
        raise GridNotFoundError("Largest contour is too small to be a Sudoku grid.")

    return largest


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def get_four_corners(contour):
    """
    Approximates the contour to a polygon and extracts exactly four corners.
    Falls back to the contour's minAreaRect if approxPolyDP doesn't yield 4 points.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return order_points(approx)

    # Fallback: use bounding rotated rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_points(np.array(box, dtype="float32"))


def detect_grid_corners(binary_image):
    contour = find_largest_contour(binary_image)
    corners = get_four_corners(contour)
    return corners, contour
