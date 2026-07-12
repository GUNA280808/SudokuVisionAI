"""
utils/helpers.py

Shared helper functions: file validation, unique filename generation,
and drawing the solved digits onto the warped grid image.
"""

import os
import uuid
import cv2
import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MIN_RESOLUTION = (150, 150)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_storage):
    """
    Validates an uploaded Flask FileStorage object.
    Returns (is_valid: bool, message: str)
    """
    filename = file_storage.filename
    if not filename:
        return False, "No file selected."

    if not allowed_file(filename):
        return False, "Unsupported file type. Please upload a PNG or JPG/JPEG image."

    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > 8 * 1024 * 1024:
        return False, "File too large. Maximum size is 8 MB."
    if size == 0:
        return False, "The uploaded file is empty."

    try:
        img = Image.open(file_storage)
        img.verify()
        file_storage.seek(0)
        img = Image.open(file_storage)
        w, h = img.size
        if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
            return False, f"Image resolution too low. Minimum is {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}px."
    except Exception:
        return False, "The uploaded file is not a valid image."
    finally:
        file_storage.seek(0)

    return True, "Valid image."


def generate_unique_filename(original_filename):
    ext = original_filename.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


def draw_solved_digits(warped_color_image, original_grid, solved_grid, grid_size=9):
    """
    Draws only the *newly solved* digits (cells that were originally blank)
    onto a copy of the warped color image, in a distinct color so the
    user can see what the AI filled in.
    """
    output = warped_color_image.copy()
    h, w = output.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(grid_size):
        for c in range(grid_size):
            if original_grid[r][c] == 0 and solved_grid[r][c] != 0:
                text = str(solved_grid[r][c])
                font_scale = cell_h / 45.0
                thickness = max(1, int(cell_h / 25))
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = c * cell_w + (cell_w - tw) // 2
                y = r * cell_h + (cell_h + th) // 2
                cv2.putText(output, text, (x, y), font, font_scale,
                            (40, 130, 240), thickness, cv2.LINE_AA)
    return output


def save_cv2_image(image, path):
    cv2.imwrite(path, image)
    return path
