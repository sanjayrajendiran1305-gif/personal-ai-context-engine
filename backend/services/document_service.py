import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)


def _markdown_lines(text):
    for line in (text or "").replace("\r", "").split("\n"):
        yield line.rstrip()


def create_document(text, fmt, title="AI Response"):
    fmt = fmt.lower().lstrip(".")
    if fmt not in {"pdf", "docx", "pptx", "txt", "md"}:
        raise ValueError("Supported export formats are PDF, DOCX, PPTX, TXT and MD.")
    import uuid
    output = OUTPUT_DIR / f"{uuid.uuid4().hex}.{fmt}"
    title = (title or "AI Response")[:100]

    if fmt in {"txt", "md"}:
        output.write_text((f"# {title}\n\n" if fmt == "md" else title + "\n\n") + text, encoding="utf-8")
    elif fmt == "docx":
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        doc = Document()
        section = doc.sections[0]
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        doc.add_heading(title, 0)
        lines = list(_markdown_lines(text))
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines = []
                while index < len(lines) and lines[index].strip().startswith("|") and lines[index].strip().endswith("|"):
                    table_lines.append(lines[index].strip().strip("|").split("|"))
                    index += 1
                table_rows = [[cell.strip() for cell in row] for row in table_lines
                              if not all(re.fullmatch(r"\s*:?-{2,}:?\s*", cell or "") for cell in row)]
                if table_rows:
                    cols = max(len(row) for row in table_rows)
                    table = doc.add_table(rows=1, cols=cols)
                    try:
                        table.style = "Light Shading Accent 1"
                    except KeyError:
                        pass
                    for col, value in enumerate(table_rows[0]):
                        table.rows[0].cells[col].text = value
                    for row in table_rows[1:]:
                        cells = table.add_row().cells
                        for col in range(cols):
                            cells[col].text = row[col] if col < len(row) else ""
                continue
            if stripped.startswith("### "):
                doc.add_heading(stripped[4:], 3)
            elif stripped.startswith("## "):
                doc.add_heading(stripped[3:], 2)
            elif stripped.startswith("# "):
                doc.add_heading(stripped[2:], 1)
            elif re.match(r"^\s*[-*+]\s+", line):
                doc.add_paragraph(re.sub(r"^\s*[-*+]\s+", "", line), style="List Bullet")
            elif re.match(r"^\s*\d+[.)]\s+", line):
                doc.add_paragraph(re.sub(r"^\s*\d+[.)]\s+", "", line), style="List Number")
            else:
                doc.add_paragraph(stripped.replace("**", "").replace("`", ""))
            index += 1
        doc.save(output)
    elif fmt == "pdf":
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#176b5b"), spaceAfter=18))
        styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], textColor=colors.HexColor("#176b5b"), spaceBefore=12, spaceAfter=6))
        story = [Paragraph(_escape(title), styles["DocTitle"]), Spacer(1, 5)]
        lines = list(_markdown_lines(text))
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 5))
                index += 1
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines = []
                while index < len(lines) and lines[index].strip().startswith("|") and lines[index].strip().endswith("|"):
                    table_lines.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                    index += 1
                table_rows = [row for row in table_lines if not all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in row)]
                if table_rows:
                    cols = max(len(row) for row in table_rows)
                    table_data = [[Paragraph(_inline_markup(cell), styles["BodyText"]) for cell in row] + [""] * (cols-len(row)) for row in table_rows]
                    table = Table(table_data, repeatRows=1, hAlign="LEFT")
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f5ee")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#245544")),
                        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbded4")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]))
                    story.extend([table, Spacer(1, 8)])
                continue
            if stripped.startswith("### ") or stripped.startswith("## ") or stripped.startswith("# "):
                heading = re.sub(r"^#+\s*", "", stripped)
                story.append(Paragraph(_escape(heading), styles["H2Custom"]))
            elif re.match(r"^\s*[-*+]\s+", line):
                bullet = re.sub(r"^\s*[-*+]\s+", "", line)
                story.append(Paragraph("&#8226; " + _inline_markup(bullet), styles["BodyText"]))
            elif re.match(r"^\s*\d+[.)]\s+", line):
                story.append(Paragraph(_inline_markup(stripped), styles["BodyText"]))
            else:
                story.append(Paragraph(_inline_markup(stripped), styles["BodyText"]))
            story.append(Spacer(1, 3))
            index += 1
        SimpleDocTemplate(str(output), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm).build(story)
    elif fmt == "pptx":
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        deck = Presentation()
        deck.slide_width = Inches(13.333)
        deck.slide_height = Inches(7.5)
        blank = deck.slide_layouts[6]
        title_slide = deck.slides.add_slide(blank)
        bg = title_slide.background.fill
        bg.solid(); bg.fore_color.rgb = RGBColor(244, 251, 248)
        box = title_slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.7))
        p = box.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = RGBColor(23, 107, 91)
        subtitle = title_slide.shapes.add_textbox(Inches(0.85), Inches(4.0), Inches(11.5), Inches(0.7))
        p = subtitle.text_frame.paragraphs[0]; p.text = "Prepared with Personal AI Context Engine"; p.font.size = Pt(16); p.font.color.rgb = RGBColor(90, 110, 105)
        chunks, current = [], []
        for line in _markdown_lines(text):
            if line.startswith("#") and current:
                chunks.append(current); current = [line]
            else:
                current.append(line)
            if len(current) >= 7:
                chunks.append(current); current = []
        if current:
            chunks.append(current)
        for chunk in chunks[:30]:
            slide = deck.slides.add_slide(blank)
            slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
            heading = next((re.sub(r"^#+\s*", "", x).strip() for x in chunk if x.startswith("#")), "Key points")
            heading_box = slide.shapes.add_textbox(Inches(0.65), Inches(0.35), Inches(12), Inches(0.75))
            hp = heading_box.text_frame.paragraphs[0]; hp.text = heading[:90]; hp.font.size = Pt(25); hp.font.bold = True; hp.font.color.rgb = RGBColor(23, 107, 91)
            body = slide.shapes.add_textbox(Inches(0.85), Inches(1.35), Inches(11.6), Inches(5.5))
            tf = body.text_frame; tf.word_wrap = True
            first = True
            for line in chunk:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                is_bullet = value.startswith(("- ", "* ", "+ "))
                value = re.sub(r"^[-*+]\s+", "", value).replace("**", "").replace("`", "")
                p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
                p.text = ("• " if is_bullet else "") + value
                p.font.size = Pt(17 if not is_bullet else 16)
                p.space_after = Pt(9)
        deck.save(output)
    return output


def _escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_markup(value):
    value = _escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", value)
    return value
