"""
train_model.py

Trains a small CNN to recognize PRINTED digits 1-9 as they appear in
photographed/screenshotted Sudoku puzzles.

IMPORTANT: Sudoku puzzles use printed/typed digits, not handwriting, so
this trains on a synthetic printed-digit dataset (see
generate_printed_digits.py) rather than MNIST. MNIST is handwritten digits
and generalizes poorly to printed fonts -- using it directly causes
frequent misreads that show up as "duplicate digit" validation errors on
nearly every real puzzle.

Digit '0' / blank cells are handled separately by the empty-cell-detection
step in vision/segmentation.py, so this model only ever needs to classify
digits 1-9 (remapped to class indices 0-8).

Usage:
    python generate_printed_digits.py   # one-time: build the dataset
    python train_model.py               # train and save models/cnn_model.h5

No internet access is required (unlike training on MNIST) since the
dataset is generated locally from system fonts.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
DATASET_PATH = os.path.join(BASE_DIR, "printed_digits_dataset.npz")


def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python generate_printed_digits.py` first "
            "to build the synthetic printed-digit training set."
        )

    data = np.load(DATASET_PATH)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_cat = to_categorical(y_train, num_classes=9)
    y_test_cat = to_categorical(y_test, num_classes=9)

    return (x_train, y_train_cat), (x_test, y_test_cat)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(9, activation="softmax"),  # classes represent digits 1-9
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading printed-digit dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=15,
        batch_size=64,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
