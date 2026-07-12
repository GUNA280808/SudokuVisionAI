import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # Folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
    PDF_FOLDER = os.path.join(BASE_DIR, "static", "pdfs")

    # Upload validation
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
    MIN_RESOLUTION = (150, 150)

    # Database
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "database.db")

    # CNN model
    MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")

    # Sudoku grid
    GRID_SIZE = 9
    CELL_IMAGE_SIZE = 28
