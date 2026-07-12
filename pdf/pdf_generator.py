"""
pdf/pdf_generator.py

Generates the "AI Sudoku Solver Report" PDF using ReportLab:
date, original image, solved image, difficulty, processing time,
recognized digit count, and the final solved grid.
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _grid_table(grid):
    style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#444444")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (2, 2), 2, colors.HexColor("#111111")),
    ])
    # Bold borders around each 3x3 box
    thick = TableStyle([
        ("LINEBELOW", (0, 2), (8, 2), 2, colors.black),
        ("LINEBELOW", (0, 5), (8, 5), 2, colors.black),
        ("LINEAFTER", (2, 0), (2, 8), 2, colors.black),
        ("LINEAFTER", (5, 0), (5, 8), 2, colors.black),
    ])
    data = [[str(v) if v != 0 else "" for v in row] for row in grid]
    t = Table(data, colWidths=1.0 * cm, rowHeights=1.0 * cm)
    t.setStyle(style)
    t.setStyle(thick)
    return t


def generate_pdf(output_path, original_image_path, solved_image_path,
                  difficulty_info, processing_time, recognized_digits,
                  confidence, solved_grid):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=22, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6
    )
    normal_style = styles["Normal"]

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm
    )

    story = []
    story.append(Paragraph("AI Sudoku Solver Report", title_style))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Summary table
    summary_data = [
        ["Difficulty", difficulty_info.get("label", "N/A")],
        ["Difficulty Score", f"{difficulty_info.get('score', 0)} / 100"],
        ["Processing Time", f"{processing_time:.3f} sec"],
        ["Digits Recognized", str(recognized_digits)],
        ["Average Confidence", f"{confidence * 100:.1f}%"],
        ["Backtracking Steps", str(difficulty_info.get("backtracking_steps", "N/A"))],
    ]
    summary_table = Table(summary_data, colWidths=[6 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.7 * cm))

    # Images side by side
    story.append(Paragraph("Original vs Solved", heading_style))
    img_row = []
    if original_image_path and os.path.exists(original_image_path):
        img_row.append(RLImage(original_image_path, width=7 * cm, height=7 * cm))
    if solved_image_path and os.path.exists(solved_image_path):
        img_row.append(RLImage(solved_image_path, width=7 * cm, height=7 * cm))
    if img_row:
        img_table = Table([img_row])
        story.append(img_table)
    story.append(Spacer(1, 0.7 * cm))

    # Final grid
    story.append(Paragraph("Final Solved Grid", heading_style))
    story.append(_grid_table(solved_grid))

    doc.build(story)
    return output_path
