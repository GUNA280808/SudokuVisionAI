"""
app.py

Main Flask application tying together the full pipeline:
Upload -> Validate -> Preprocess -> Detect Grid -> Warp -> Segment ->
Recognize Digits -> Validate Sudoku -> Solve -> Analyze Difficulty ->
Overlay -> Save -> Generate PDF -> Return Result
"""

import os
import time
import base64
import traceback

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, jsonify, abort
)
from werkzeug.utils import secure_filename

from config import Config
from database import db
from utils.helpers import (
    validate_image_file, generate_unique_filename,
    draw_solved_digits, save_cv2_image
)
from vision.preprocess import preprocess_pipeline
from vision.contour import detect_grid_corners, GridNotFoundError
from vision.perspective import warp_perspective
from vision.segmentation import split_into_cells, is_cell_empty
from solver.validator import is_valid_grid
from solver.sudoku_solver import solve_sudoku
from solver.difficulty import compute_difficulty
from pdf.pdf_generator import generate_pdf

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

for folder in (Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.PDF_FOLDER):
    os.makedirs(folder, exist_ok=True)
os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)

db.init_db()


# --------------------------------------------------------------------------
# Core pipeline
# --------------------------------------------------------------------------

class PipelineError(Exception):
    pass


def run_pipeline(image_path):
    """
    Runs the full CV + ML + solving pipeline on a saved image.
    Returns a dict with everything the result page / PDF / DB need.
    """
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        raise PipelineError("Could not read the uploaded image. It may be corrupted.")

    stages = preprocess_pipeline(image)

    try:
        corners, _ = detect_grid_corners(stages["edges"])
    except GridNotFoundError as e:
        raise PipelineError(
            "Could not detect a Sudoku grid in this image. "
            "Try a clearer, well-lit photo taken from directly above the puzzle."
        ) from e

    warped_gray, matrix = warp_perspective(stages["gray"], corners, output_size=450)
    warped_color, _ = warp_perspective(stages["resized"], corners, output_size=450)

    cells = split_into_cells(warped_gray)
    empty_flags = [is_cell_empty(cell) for cell in cells]

    # Digit recognition (lazy import so the app can still boot / show clear
    # errors even if the CNN model file hasn't been trained yet)
    try:
        from vision.digit_recognition import recognize_grid
        grid, confidences, recognized_count = recognize_grid(cells, empty_flags)
    except FileNotFoundError as e:
        raise PipelineError(str(e)) from e

    valid, message = is_valid_grid(grid)
    if not valid:
        raise PipelineError(
            f"The recognized Sudoku puzzle is invalid ({message}). "
            "This is usually caused by a misread digit -- try a clearer photo."
        )

    solve_result = solve_sudoku(grid)
    if not solve_result["solved"]:
        raise PipelineError("This puzzle could not be solved. It may have no valid solution.")

    difficulty_info = compute_difficulty(
        grid, solve_result["steps"], solve_result["time_seconds"]
    )

    overlay = draw_solved_digits(warped_color, grid, solve_result["grid"])

    all_confidences = [c for row in confidences for c in row if c > 0]
    avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

    total_time = time.time() - start_time

    return {
        "original_grid": grid,
        "solved_grid": solve_result["grid"],
        "recognized_count": recognized_count,
        "avg_confidence": avg_confidence,
        "difficulty": difficulty_info,
        "backtracking_steps": solve_result["steps"],
        "solve_time": solve_result["time_seconds"],
        "total_time": round(total_time, 4),
        "warped_original": warped_color,
        "overlay_image": overlay,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    recent = db.get_all_records(limit=6)
    return render_template("index.html", recent=recent)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("home"))

    file = request.files["image"]
    valid, message = validate_image_file(file)
    if not valid:
        flash(message, "danger")
        return redirect(url_for("home"))

    unique_name = generate_unique_filename(secure_filename(file.filename))
    upload_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    return process_and_store(upload_path, unique_name)


@app.route("/capture", methods=["POST"])
def capture():
    """Handles images captured via the browser camera (sent as base64 dataURL)."""
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "No image data received."}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        binary_data = base64.b64decode(encoded)
    except Exception:
        return jsonify({"success": False, "message": "Malformed image data."}), 400

    if len(binary_data) > Config.MAX_CONTENT_LENGTH:
        return jsonify({"success": False, "message": "Captured image is too large."}), 400

    unique_name = generate_unique_filename("capture.png")
    upload_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    with open(upload_path, "wb") as f:
        f.write(binary_data)

    result = process_and_store(upload_path, unique_name, redirect_response=False)
    return result


def process_and_store(upload_path, unique_name, redirect_response=True):
    try:
        result = run_pipeline(upload_path)
    except PipelineError as e:
        if redirect_response:
            flash(str(e), "danger")
            return redirect(url_for("home"))
        return jsonify({"success": False, "message": str(e)}), 422
    except Exception:
        traceback.print_exc()
        msg = "An unexpected error occurred while processing the image."
        if redirect_response:
            flash(msg, "danger")
            return redirect(url_for("home"))
        return jsonify({"success": False, "message": msg}), 500

    base_name = os.path.splitext(unique_name)[0]
    original_out = f"{base_name}_original.png"
    solved_out = f"{base_name}_solved.png"

    save_cv2_image(result["warped_original"], os.path.join(Config.OUTPUT_FOLDER, original_out))
    save_cv2_image(result["overlay_image"], os.path.join(Config.OUTPUT_FOLDER, solved_out))

    record_id = db.add_record(
        image_name=unique_name,
        original_image=original_out,
        solved_image=solved_out,
        difficulty=result["difficulty"]["label"],
        difficulty_score=result["difficulty"]["score"],
        processing_time=result["total_time"],
        recognized_digits=result["recognized_count"],
        confidence=round(result["avg_confidence"], 4),
        solved_status="Solved",
    )

    # Generate the PDF report right away so it's ready for download
    pdf_name = f"report_{record_id}.pdf"
    pdf_path = os.path.join(Config.PDF_FOLDER, pdf_name)
    generate_pdf(
        output_path=pdf_path,
        original_image_path=os.path.join(Config.OUTPUT_FOLDER, original_out),
        solved_image_path=os.path.join(Config.OUTPUT_FOLDER, solved_out),
        difficulty_info=result["difficulty"],
        processing_time=result["total_time"],
        recognized_digits=result["recognized_count"],
        confidence=result["avg_confidence"],
        solved_grid=result["solved_grid"],
    )
    db.update_pdf_name(record_id, pdf_name)

    if redirect_response:
        return redirect(url_for("result", record_id=record_id))
    return jsonify({"success": True, "redirect": url_for("result", record_id=record_id)})


@app.route("/result/<int:record_id>")
def result(record_id):
    record = db.get_record(record_id)
    if not record:
        abort(404)
    return render_template("result.html", record=record)


@app.route("/history")
def history():
    records = db.get_all_records()
    return render_template("history.html", records=records)


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete(record_id):
    record = db.get_record(record_id)
    if record:
        for filename, folder in (
            (record["original_image"], Config.OUTPUT_FOLDER),
            (record["solved_image"], Config.OUTPUT_FOLDER),
            (record["pdf_name"], Config.PDF_FOLDER),
        ):
            if filename:
                path = os.path.join(folder, filename)
                if os.path.exists(path):
                    os.remove(path)
        db.delete_record(record_id)
        flash("History entry deleted.", "success")
    return redirect(url_for("history"))


@app.route("/download_pdf/<int:record_id>")
def download_pdf(record_id):
    record = db.get_record(record_id)
    if not record or not record["pdf_name"]:
        abort(404)
    return send_from_directory(Config.PDF_FOLDER, record["pdf_name"], as_attachment=True)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum upload size is 8 MB.", "danger")
    return redirect(url_for("home"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
