"""
database/db.py

Thin SQLite wrapper for the `history` table.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    image_name TEXT NOT NULL,
    original_image TEXT NOT NULL,
    solved_image TEXT,
    difficulty TEXT,
    difficulty_score REAL,
    processing_time REAL,
    recognized_digits INTEGER,
    confidence REAL,
    solved_status TEXT NOT NULL,
    pdf_name TEXT
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def add_record(image_name, original_image, solved_image, difficulty,
               difficulty_score, processing_time, recognized_digits,
               confidence, solved_status, pdf_name=None):
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO history
        (date, image_name, original_image, solved_image, difficulty,
         difficulty_score, processing_time, recognized_digits, confidence,
         solved_status, pdf_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name, original_image, solved_image, difficulty,
            difficulty_score, processing_time, recognized_digits, confidence,
            solved_status, pdf_name,
        ),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id


def update_pdf_name(record_id, pdf_name):
    conn = get_connection()
    conn.execute("UPDATE history SET pdf_name = ? WHERE id = ?", (pdf_name, record_id))
    conn.commit()
    conn.close()


def get_record(record_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_records(limit=None):
    conn = get_connection()
    query = "SELECT * FROM history ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_record(record_id):
    conn = get_connection()
    conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
