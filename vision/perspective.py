"""
vision/perspective.py

Warps the detected grid corners into a top-down, perfectly square image.
"""

import cv2
import numpy as np


def warp_perspective(image, corners, output_size=450):
    """
    image: the (grayscale or color) resized source image
    corners: 4x2 array ordered [top-left, top-right, bottom-right, bottom-left]
    output_size: side length (px) of the resulting square warped image
    """
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))
    return warped, matrix


def inverse_warp_overlay(solved_cell_image, original_image, corners, output_size=450):
    """
    Takes the square 'solved' overlay (same size as the warped grid) and
    projects it back onto the original perspective, so the final result
    image looks like the original photo with missing digits filled in.
    """
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    matrix_inv = cv2.getPerspectiveTransform(dst, corners)
    h, w = original_image.shape[:2]
    unwarped = cv2.warpPerspective(solved_cell_image, matrix_inv, (w, h))

    mask = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY) if len(unwarped.shape) == 3 else unwarped
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    if len(original_image.shape) == 2:
        background = cv2.bitwise_and(original_image, mask_inv)
        combined = cv2.add(background, cv2.bitwise_and(unwarped, mask))
    else:
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv_3c = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(original_image, mask_inv_3c)
        combined = cv2.add(background, cv2.bitwise_and(unwarped, mask_3c))

    return combined
