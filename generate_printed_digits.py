"""
generate_printed_digits.py

Sudoku puzzles (photographed from newspapers, apps, books, or screenshots)
almost always use PRINTED/TYPED digits, not handwriting. A CNN trained only
on MNIST (handwritten digits) generalizes poorly to printed fonts and causes
frequent misreads -> "duplicate digit" errors on every puzzle.

This script renders digits 1-9 in dozens of real fonts, sizes, weights,
slight rotations, and noise/blur conditions to build a synthetic dataset
that matches what the OpenCV pipeline actually hands the CNN: a tightly
cropped, thresholded, black-and-white 28x28 cell image.

Output: printed_digits_dataset.npz with x_train/y_train/x_test/y_test,
ready to be consumed by train_model.py.
"""

import os
import glob
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "printed_digits_dataset.npz")
IMG_SIZE = 28
SAMPLES_PER_DIGIT_PER_FONT = 25  # varied by augmentation, not literal duplicates

# A broad, license-friendly set of font families that cover the range of
# typefaces you'll actually see in printed/digital Sudoku puzzles.
FONT_SEARCH_TERMS = [
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
    "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",
    "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
    "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
    "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf",
    "FreeSans.ttf", "FreeSansBold.ttf", "FreeSerif.ttf", "FreeSerifBold.ttf",
    "FreeMono.ttf", "FreeMonoBold.ttf",
]


def find_fonts():
    found = []
    for term in FONT_SEARCH_TERMS:
        matches = glob.glob(f"/usr/share/fonts/**/{term}", recursive=True)
        found.extend(matches)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def render_digit(digit, font_path, canvas_size=150):
    """
    Renders a single digit inside a CELL-sized canvas (not a tightly-cropped
    digit canvas), then the caller crops it the same way
    vision/segmentation.py::trim_cell_border does. This matters: a real
    Sudoku cell photographed and cropped by the CV pipeline shows a
    relatively small, sometimes off-center digit surrounded by empty cell
    space -- not a digit that fills almost the whole crop. Training on
    tightly-cropped bold digits causes a train/inference distribution
    mismatch that tanks real-world accuracy.
    """
    # Font size relative to the *cell*, matching typical printed Sudoku
    # proportions (digit occupies roughly 45-65% of the cell height).
    font_size = random.randint(int(canvas_size * 0.45), int(canvas_size * 0.62))
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Small random jitter around center, like a slightly off-center photo crop
    jitter_x = random.uniform(-0.06, 0.06) * canvas_size
    jitter_y = random.uniform(-0.06, 0.06) * canvas_size
    x = (canvas_size - tw) / 2 - bbox[0] + jitter_x
    y = (canvas_size - th) / 2 - bbox[1] + jitter_y
    draw.text((x, y), text, fill=0, font=font)
    return img


def crop_like_real_pipeline(img, margin_ratio=0.16):
    """
    Mirrors vision/segmentation.py::trim_cell_border so the synthetic
    training distribution matches what the real CV pipeline hands the CNN.
    """
    arr = np.array(img)
    h, w = arr.shape
    my, mx = int(h * margin_ratio), int(w * margin_ratio)
    return arr[my:h - my, mx:w - mx]


def augment(img):
    """Applies small random rotation, noise, and blur to simulate photo conditions."""
    angle = random.uniform(-6, 6)
    img = img.rotate(angle, fillcolor=255, resample=Image.BICUBIC)

    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.1)))

    arr = np.array(img).astype("float32")
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    # Random contrast jitter, mimicking varied lighting/exposure
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        arr = np.clip((arr - 127.5) * factor + 127.5, 0, 255)

    return arr.astype("uint8")


def to_cnn_ready(arr, size=IMG_SIZE):
    """
    Mimics the exact preprocessing the real pipeline applies in
    vision/segmentation.py::prepare_cell_for_cnn -- Otsu threshold,
    polarity normalization, and bounding-box-based glyph centering/scaling
    -- so the training distribution matches the inference distribution
    regardless of how large the digit happens to render within its cell.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from vision.segmentation import normalize_digit_glyph

    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    return normalize_digit_glyph(thresh, canvas_size=size, target_size=20)


def build_dataset():
    fonts = find_fonts()
    if not fonts:
        raise RuntimeError(
            "No fonts found. Install fonts-dejavu-core / fonts-liberation / fonts-freefont-ttf "
            "or point FONT_SEARCH_TERMS at fonts available on your system."
        )
    print(f"Using {len(fonts)} font files.")

    X, y = [], []
    for digit in range(1, 10):
        for font_path in fonts:
            for _ in range(SAMPLES_PER_DIGIT_PER_FONT):
                base = render_digit(digit, font_path)
                if base is None:
                    continue
                augmented = augment(base)
                cnn_ready = to_cnn_ready(augmented)
                X.append(cnn_ready)
                y.append(digit - 1)  # classes 0-8 represent digits 1-9

    X = np.array(X, dtype="uint8")
    y = np.array(y, dtype="int64")
    print(f"Generated {len(X)} synthetic printed-digit samples.")

    # Shuffle and split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.9)
    x_train, x_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    np.savez_compressed(
        OUTPUT_PATH,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Saved dataset to {OUTPUT_PATH}")
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    build_dataset()
