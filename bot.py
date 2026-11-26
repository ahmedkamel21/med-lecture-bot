import asyncio
import gc
import io
import json
import logging
import os
import re
import tempfile
from fpdf import FPDF
from google import genai
from google.genai import types
from pypdf import PdfReader
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.8-flash"

# المحاضرات المخزنة مؤقتًا في الجلسة وحدود النص لحماية الرامات
MAX_LECTURES_PER_SESSION = 8
MAX_CHARS_PER_LECTURE = 7500
MAX_COMBINED_CHARS = 48000

FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
)
FONT_LOCAL_PATH = "/tmp/Amiri-Regular.ttf"

MEDICAL_HIGH_YIELD_SYSTEM_PROMPT = """
You are a high-yield medical exam tutor helping a medical student study efficiently for English medical exams.
Golden Rule: Focus heavily on high-yield exam topics ("what gets tested"):
- Core mechanisms, pharmacology/pathophysiology principles, and key clinical concepts.
- Classic clinical vignettes, distinguishing presentations, and high-yield indications/contraindications/adverse effects.
- Crucial comparisons, differential points, and critical exceptions.
- When multiple lectures are provided, integrate concepts across lectures, highlight key contrasts/comparisons, and test high-yield connections.
Strictly avoid trivial facts or low-yield filler details.
Write ALL medical content, questions, choices, answers, summaries, explanations, and Anki cards strictly in clear, readable English.
CRITICAL FORMATTING RULE:
Do NOT use asterisks (* or **), markdown headers (#, ##, ###), horizontal rules (---), or markdown table pipes (| :--- |).
Use plain text with clear line breaks, numbered items, or simple dashes (- ) only where helpful so the text is clean and easy to read.
"""


def clean_plain_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("*", "")
    lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.fullmatch(r"[-_=~]{3,}", s):
            continue
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", s):
            continue
        s = re.sub(r"^#{1,6}\s*", "", s)
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            s = "  •  ".join(cells)
        lines.append(s)

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def format_anki_tab_deck(raw_text: str) -> str:
    """ينسق بطاقات Anki بحيث يكون الفاصل المثبت دائمًا هو Tab (\t) بين Front و Back."""
    cleaned_block = clean_plain_text(raw_text)
    cards = []

    for line in cleaned_block.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        # إزالة ترقيم البداية إن وُجد مثل "1. " أو "1) " أو "- "
        s = re.sub(r"^(?:\d+[\).\:-]\s*|[-•]\s+)", "", s).strip()
        if not s:
            continue

        front = ""
        back = ""

        if "\t" in s:
            parts = s.split("\t", 1)
            front, back = parts[0].strip(), parts[1].strip()
        elif "|" in s:
            parts = s.split("|", 1)
            front, back = parts[0].strip(), parts[1].strip()
        elif "::" in s:
            parts = s.split("::", 1)
            front, back = parts[0].strip(), parts[1].strip()
        elif "->" in s:
            parts = s.split("->", 1)
            front, back = parts[0].strip(), parts[1].strip()

        if front and back:
            front_clean = re.sub(r"[\t\r\n]+", " ", front).replace("*", "").strip()
            back_clean = re.sub(r"[\t\r\n]+", " ", back).replace("*", "").strip()
            if front_clean and back_clean:
                cards.append(f"{front_clean}\t{back_clean}")

    return "\n".join(cards)


def sanitize_pdf_text(text: str) -> str:
    if not text:
        return ""
    cleaned = clean_plain_text(text)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    words = []
    for word in cleaned.split(" "):
        if len(word) > 40:
            chunks = [word[i : i + 35] for i in range(0, len(word), 35)]
            words.extend(chunks)
        else:
            words.append(word)
    return " ".join(words)


def get_font_path() -> str | None:
    if os.path.exists(FONT_LOCAL_PATH) and os.path.getsize(FONT_LOCAL_PATH) > 10000:
        return FONT_LOCAL_PATH
    try:
        resp = requests.get(FONT_URL, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 10000:
            with open(FONT_LOCAL_PATH, "wb") as f:
                f.write(resp.content)
            return FONT_LOCAL_PATH
    except Exception as e:
        logger.warning("Could not download font: %s", e)
    return None


def _create_base_pdf() -> tuple[FPDF, str, float]:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=14, top=14, right=14)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    font_path = get_font_path()
    font_family = "Helvetica"

    if font_path:
        try:
            pdf.add_font("UniFont", "", font_path)
            font_family = "UniFont"
        except Exception as e:
            logger.warning("Failed adding custom font, fallback to Helvetica: %s", e)
            font_family = "Helvetica"

    return pdf, font_family, usable_width


def _write_pdf_block(
    pdf: FPDF,
    font_family: str,
    usable_width: float,
    txt: str,
    font_size: int,
    line_h: float,
    space_after: float = 2.0,
) -> None:
    safe_txt = sanitize_pdf_text(txt)
    if not safe_txt:
        return
    if font_family == "Helvetica":
        safe_txt = safe_txt.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_x(pdf.l_margin)
    pdf.set_font(font_family, size=font_size)
    pdf.multi_cell(usable_width, line_h, text=safe_txt)
    if space_after > 0:
        pdf.ln(space_after)


def _output_pdf_bytes(pdf: FPDF) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_mcq_pdf_bytes(session_label: str, mcq_items: list[dict]) -> bytes:
    pdf, font_family, usable_width = _create_base_pdf()

    safe_label = sanitize_pdf_text(session_label) or "Lectures"
    _write_pdf_block(
        pdf,
        font_family,
        usable_width,
        f"High-Yield MCQ ({len(mcq_items)} Questions) - {safe_label}",
        font_size=16,
        line_h=8.5,
        space_after=5.0,
    )

    for idx, item in enumerate(mcq_items, start=1):
        try:
            question_text = str(item.get("question", "")).strip()
            options = item.get("options") or []
            correct_idx = item.get("correct_option_id", 0)
            explanation = str(item.get("explanation", "")).strip()

            _write_pdf_block(
                pdf,
                font_family,
                usable_width,
                f"Q{idx}: {question_text}",
                font_size=13,
                line_h=7.2,
                space_after=2.0,
            )

            for opt_i, opt_val in enumerate(options):
                letter = chr(65 + opt_i) if opt_i < 26 else str(opt_i + 1)
                _write_pdf_block(
                    pdf,
                    font_family,
                    usable_width,
                    f"   {letter}) {str(opt_val).strip()}",
                    font_size=12,
                    line_h=6.8,
                    space_after=1.2,
                )

            if isinstance(correct_idx, int) and 0 <= correct_idx < len(options):
                correct_letter = chr(65 + correct_idx) if correct_idx < 26 else str(correct_idx + 1)
                correct_label = str(options[correct_idx]).strip()
            else:
                correct_letter = "A"
                correct_label = ""

            _write_pdf_block(
                pdf,
                font_family,
                usable_width,
                f"Correct Answer: ({correct_letter}) {correct_label}",
                font_size=12,
                line_h=6.8,
                space_after=1.5,
            )

            if explanation:
                _write_pdf_block(
                    pdf,
                    font_family,
                    usable_width,
                    f"Explanation: {explanation}",
                    font_size=12,
                    line_h=6.8,
                    space_after=5.0,
                )
            else:
                pdf.ln(3.5)
        except Exception as q_err:
            logger.warning("Skipping malformed question %s in PDF: %s", idx, q_err)
            continue

    return _output_pdf_bytes(pdf)


def generate_summary_pdf_bytes(session_label: str, summary_text: str) -> bytes:
    pdf, font_family, usable_width = _create_base_pdf()

    safe_label = sanitize_pdf_text(session_label) or "Lectures"
    _write_pdf_block(
        pdf,
        font_family,
        usable_width,
        f"High-Yield Summary - {safe_label}",
        font_size=16,
        line_h=8.5,
        space_after=5.0,
    )

    cleaned_summary = clean_plain_text(summary_text)
    paragraphs = [p.strip() for p in cleaned_summary.split("\n") if p.strip()]

    for para in paragraphs:
        is_heading = (
            len(para) <= 85
            and not para.endswith((".", "?", "!", ":", ";"))
            and (
                para.istitle()
                or para.isupper()
                or re.match(r"^(\d+[\).\s]|Part\s+\d+|Section\s+\d+|High-Yield)", para, re.I)
            )
        )
        if is_heading:
            pdf.ln(2.0)
            _write_pdf_block(
                pdf,
                font_family,
                usable_width,
                para,
                font_size=14,
                line_h=7.8,
                space_after=2.0,
            )
        else:
            _write_pdf_block(
                pdf,
                font_family,
                usable_width,
                para,
                font_size=12.5,
                line_h=7.0,
                space_after=2.5,
            )

    return _output_pdf_bytes(pdf)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_parts = []
    current_len = 0
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            part = f"--- Page {idx} ---\n{text}"
            if current_len + len(part) > MAX_CHARS_PER_LECTURE:
                remaining = MAX_CHARS_PER_LECTURE - current_len
                if remaining > 200:
                    extracted_parts.append(part[:remaining])
                extracted_parts.append("\n[Truncated per-lecture limit to keep memory safe]")
                break
            extracted_parts.append(part)
            current_len += len(part)
    return "\n\n".join(extracted_parts)


def get_session_lectures(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    lectures = context.user_data.get("lectures")
    if not isinstance(lectures, list):
        lectures = []
        context.user_data["lectures"] = lectures
    return lectures


def get_active_lecture_index(context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]) -> int | None:
    idx = context.user_data.get("active_lecture_idx")
    if isinstance(idx, int) and 0 <= idx < len(lectures):
        return idx
    return None


def get_target_lectures_and_label(
    context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]
) -> tuple[list[dict], str]:
    if not lectures:
        return [], "Lectures"

    active_idx = get_active_lecture_index(context, lectures)
    if active_idx is not None:
        lec = lectures[active_idx]
        title = lec.get("title", f"Lecture_{active_idx + 1}")[:40]
        return [lec], title

    if len(lectures) == 1:
        return [lectures[0]], lectures[0].get("title", "Lecture")[:40]

    return lectures, f"{len(lectures)}_Combined_Lectures"


def build_combined_lecture_text(target_lectures: list[dict]) -> str:
    combined_parts = []
    total_len = 0
    for idx, lec in enumerate(target_lectures, start=1):
        title = lec.get("title", f"Lecture_{idx}")
        txt = lec.get("text", "")
        header = f"================ LECTURE {idx}: {title} ================\n"
        block = header + txt
        if total_len + len(block) > MAX_COMBINED_CHARS:
            remaining = MAX_COMBINED_CHARS - total_len
            if remaining > 300:
                combined_parts.append(block[:remaining])
            combined_parts.append("\n[Combined text capped to protect server memory]")
            break
        combined_parts.append(block)
        total_len += len(block)
    return "\n\n".join(combined_parts)


def build_mcq_json_prompt(combined_text: str, num_lectures: int, traps_mode: bool = False) -> str:
    multi_rule = (
        f"- You are given {num_lectures} lectures in this study session: distribute questions across ALL lectures AND include cross-lecture integration/comparison questions."
        if num_lectures > 1
        else "- Focus on the high-yield exam points in this single selected lecture."
    )
    if traps_mode:
        focus_rules = f"""
{multi_rule}
- Special Focus (Traps, Look-Alikes & Integrations): Build questions specifically loaded with exam traps, easily confused look-alike entities/drugs/conditions/mechanisms, and high-yield cross-topic/cross-lecture integrations.
- Include subtle distractors that test directionality (increase vs. decrease, agonist vs. antagonist, exception vs. rule) and close look-alikes.
- In `explanation`, clearly point out the exact exam trap and how to distinguish the correct answer from the main look-alike distractor.
"""
    else:
        focus_rules = f"""
{multi_rule}
- Prioritize clinical vignettes, mechanisms, key distinctions, contraindications, adverse effects, and classic exam traps; avoid trivial details.
- In `explanation`, provide a concise, high-yield explanation in English.
"""

    return f"""
Generate high-yield English medical Multiple Choice Questions (MCQs) from the provided lecture material.
Strict count rules:
- Minimum is 30 questions; NEVER output fewer than 30 questions.
- Maximum is open-ended depending on how rich and long the provided lecture material is (generate more than 30 when covering multiple lectures or substantial high-yield material).
- Write ALL questions, options, and explanations strictly in English.
{focus_rules}
- Each question must have exactly 4 options (list of 4 concise strings, each <= 85 characters for Telegram polls).
- Question stem must be <= 260 characters.
- Set `correct_option_id` as an integer from 0 to 3.
- Do NOT use asterisks (*) in any field.

Return valid JSON only in this exact structure:
{{
  "questions": [
    {{
      "question": "Concise English clinical vignette or high-yield question",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_option_id": 0,
      "explanation": "Brief high-yield explanation in English"
    }}
  ]
}}

Provided lectures:
\"\"\"
{combined_text}
\"\"\"
"""


def build_mode_prompt(mode: str, combined_text: str, num_lectures: int) -> str:
    multi_note = (
        f"Note: There are {num_lectures} lectures combined below. Cover all of them and explicitly link/compare high-yield concepts across lectures where relevant.\n"
        if num_lectures > 1
        else "Note: Focus strictly on this single selected lecture.\n"
    )
    if mode == "saq":
        task_instruction = f"""
{multi_note}Generate high-yield Short Answer Questions (SAQs) in English based on the most testable points in the provided material.
- Focus on mechanisms, comparisons, clinical reasoning, and key distinctions.
- Provide a concise model answer in English right after each question.
- Do NOT use asterisks (*), markdown headers (#), or markdown tables.
"""
    elif mode == "mix":
        task_instruction = f"""
{multi_note}Generate a focused high-yield Mix of English MCQs (vignettes/exam traps) and concise English SAQs from the most testable parts of the provided material.
- Include clear answers and brief high-yield explanations in English.
- Do NOT use asterisks (*), markdown headers (#), or markdown tables. Present comparisons as clean structured lines instead of tables.
"""
    elif mode == "summary":
        task_instruction = f"""
{multi_note}Create a well-structured, high-yield English medical exam review summary of the provided material.
- Organize with clear section titles and short, readable paragraphs or dash points (- ).
- Highlight key mechanisms, comparisons, exceptions, and clinical/exam correlations. Remove fluff.
- Do NOT use asterisks (*), markdown headers (#), or markdown tables.
"""
    elif mode == "anki":
        task_instruction = f"""
{multi_note}Create high-yield English Anki flashcards for medical exam review, one card per line, with NO numbering, NO headings, NO intro/outro text, and NO asterisks (*).
Each line must have ONLY two parts separated by a single TAB character:
Front (question, clinical vignette, or key exam concept in English)<TAB>Back (concise accurate answer + key exam differentiator in English)
Do not put extra tabs inside the front or back text. Focus strictly on high-yield testable points.
"""
    else:
        task_instruction = f"{multi_note}Summarize the high-yield exam points in English without asterisks."

    return f"""{task_instruction}

Provided lectures:
\"\"\"
{combined_text}
\"\"\"
"""


def get_choices_keyboard(
    has_mcq_pdf: bool = False,
    has_summary_pdf: bool = False,
    has_lectures: bool = False,
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("MCQ Poll (High-Yield)", callback_data="mode_mcq"),
            InlineKeyboardButton("MCQ Traps & Mix", callback_data="mode_traps"),
        ],
        [
            InlineKeyboardButton("SAQ", callback_data="mode_saq"),
            InlineKeyboardButton("Mix (MCQ + SAQ)", callback_data="mode_mix"),
        ],
        [
            InlineKeyboardButton("Summary (PDF)", callback_data="mode_summary"),
            InlineKeyboardButton("An
