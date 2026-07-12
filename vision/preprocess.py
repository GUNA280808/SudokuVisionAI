"""
vision/preprocess.py

Handles the initial OpenCV preprocessing stage:
Resize -> Grayscale -> Gaussian Blur -> Adaptive Threshold -> Edge Detection
"""

import cv2
import numpy as np

TARGET_WIDTH = 900


def resize_image(image, width=TARGET_WIDTH):
    """Resize image preserving aspect ratio, capped at `width`."""
    h, w = image.shape[:2]
    if w <= width:
        return image
    ratio = width / float(w)
    new_dim = (width, int(h * ratio))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(gray_image, ksize=(9, 9)):
    return cv2.GaussianBlur(gray_image, ksize, 0)


def adaptive_threshold(blurred_image):
    thresh = cv2.adaptiveThreshold(
        blurred_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )
    return thresh


def detect_edges(thresh_image):
    """Dilate slightly to close gaps in the grid lines before contour search."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh_image, kernel, iterations=1)
    return dilated


def preprocess_pipeline(image):
    """
    Runs the full preprocessing pipeline and returns a dict of every
    intermediate stage (useful for debugging / displaying on the frontend)
    plus the final binary image used for contour detection.
    """
    resized = resize_image(image)
    gray = to_grayscale(resized)
    blurred = blur(gray)
    thresh = adaptive_threshold(blurred)
    edges = detect_edges(thresh)

    return {
        "resized": resized,
        "gray": gray,
        "blurred": blurred,
        "thresh": thresh,
        "edges": edges,
    }
