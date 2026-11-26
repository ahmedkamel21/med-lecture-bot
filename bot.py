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

# حدود النص: حد أعلى للمحاضرة المفردة وحد متوازن للمحاضرات المجمعة لحماية الرامات وتقليل ضغط التوليد
MAX_LECTURES_PER_SESSION = 8
MAX_CHARS_SINGLE_LECTURE = 28000
MAX_CHARS_PER_LECTURE = 6000
MAX_COMBINED_CHARS = 34000
MAX_PREVIOUS_QUESTIONS_CONTEXT = 120

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


async def generate_with_retry(
    contents: str,
    config: types.GenerateContentConfig,
    max_attempts: int = 3,
):
    """ينفذ طلب التوليد مع إعادة المحاولة تلقائيًا عند ضغط السيرفر المؤقت (503 / ServerError)."""
    delays = [2.0, 4.5, 7.5]
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_transient = (
                "503" in err_str
                or "unavailable" in err_str
                or "servererror" in err_str
                or "high demand" in err_str
                or "429" in err_str
                or "resource_exhausted" in err_str
                or "timeout" in err_str
            )
            if not is_transient or attempt == max_attempts:
                raise
            wait_sec = delays[min(attempt - 1, len(delays) - 1)]
            logger.warning(
                "Transient Gemini error on attempt %s/%s (%s). Retrying in %.1fs...",
                attempt,
                max_attempts,
                type(e).__name__,
                wait_sec,
            )
            await asyncio.sleep(wait_sec)

    if last_err:
        raise last_err


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


def generate_mcq_pdf_bytes(session_label: str, mcq_items: list[dict], heading_prefix: str = "High-Yield MCQ") -> bytes:
    pdf, font_family, usable_width = _create_base_pdf()

    safe_label = sanitize_pdf_text(session_label) or "Lectures"
    safe_prefix = sanitize_pdf_text(heading_prefix) or "High-Yield MCQ"
    _write_pdf_block(
        pdf,
        font_family,
        usable_width,
        f"{safe_prefix} ({len(mcq_items)} Questions) - {safe_label}",
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


def generate_text_document_pdf_bytes(doc_title: str, session_label: str, body_text: str) -> bytes:
    """ينشئ ملف PDF منسق بخط كبير إلى متوسط وخالٍ من النجوم للملخصات أو SAQ أو Mix."""
    pdf, font_family, usable_width = _create_base_pdf()

    safe_title = sanitize_pdf_text(doc_title) or "High-Yield Document"
    safe_label = sanitize_pdf_text(session_label) or "Lectures"
    _write_pdf_block(
        pdf,
        font_family,
        usable_width,
        f"{safe_title} - {safe_label}",
        font_size=16,
        line_h=8.5,
        space_after=5.0,
    )

    cleaned_body = clean_plain_text(body_text)
    paragraphs = [p.strip() for p in cleaned_body.split("\n") if p.strip()]

    for para in paragraphs:
        is_heading = (
            len(para) <= 90
            and not para.endswith((".", "?", "!", ":", ";"))
            and (
                para.istitle()
                or para.isupper()
                or re.match(
                    r"^(\d+[\).\s]|Q\d+[:.)\s]|Question\s*\d+|Part\s+\d+|Section\s+\d+|High-Yield)",
                    para,
                    re.I,
                )
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


def extract_text_from_pdf(pdf_bytes: bytes, max_chars: int = MAX_CHARS_SINGLE_LECTURE) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_parts = []
    current_len = 0
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            part = f"--- Page {idx} ---\n{text}"
            if current_len + len(part) > max_chars:
                remaining = max_chars - current_len
                if remaining > 200:
                    extracted_parts.append(part[:remaining])
                extracted_parts.append("\n[Truncated at character cap to keep memory safe]")
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


def can_use_cross_lecture_links(
    context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]
) -> bool:
    return len(lectures) > 1 and get_active_lecture_index(context, lectures) is None


def build_combined_lecture_text(target_lectures: list[dict]) -> str:
    is_single = len(target_lectures) == 1
    cap = MAX_CHARS_SINGLE_LECTURE if is_single else MAX_COMBINED_CHARS
    combined_parts = []
    total_len = 0
    for idx, lec in enumerate(target_lectures, start=1):
        title = lec.get("title", f"Lecture_{idx}")
        txt = lec.get("text", "")
        if is_single:
            txt = txt[:MAX_CHARS_SINGLE_LECTURE]
        else:
            txt = txt[:MAX_CHARS_PER_LECTURE]
        header = f"================ LECTURE {idx}: {title} ================\n"
        block = header + txt
        if total_len + len(block) > cap:
            remaining = cap - total_len
            if remaining > 300:
                combined_parts.append(block[:remaining])
            combined_parts.append("\n[Text capped to protect server memory]")
            break
        combined_parts.append(block)
        total_len += len(block)
    return "\n\n".join(combined_parts)


def get_seen_questions_list(context: ContextTypes.DEFAULT_TYPE, mode_key: str) -> list[str]:
    seen_map = context.user_data.get("seen_questions_by_mode")
    if not isinstance(seen_map, dict):
        seen_map = {}
        context.user_data["seen_questions_by_mode"] = seen_map
    items = seen_map.get(mode_key)
    if not isinstance(items, list):
        items = []
        seen_map[mode_key] = items
    return items


def record_seen_questions(
    context: ContextTypes.DEFAULT_TYPE, mode_key: str, new_questions: list[str]
) -> None:
    seen = get_seen_questions_list(context, mode_key)
    existing_lower = {q.strip().lower() for q in seen}
    for q in new_questions:
        q_clean = re.sub(r"\s+", " ", str(q)).strip()
        if q_clean and q_clean.lower() not in existing_lower:
            seen.append(q_clean[:220])
            existing_lower.add(q_clean.lower())
    if len(seen) > MAX_PREVIOUS_QUESTIONS_CONTEXT:
        del seen[: len(seen) - MAX_PREVIOUS_QUESTIONS_CONTEXT]


def extract_question_lines_from_text(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "?" in s or re.match(r"^(?:Q\d+|Question\s*\d+|\d+[\).\s-])", s, re.I):
            found.append(s[:220])
    return found


def build_avoid_duplicates_block(seen_questions: list[str]) -> str:
    if not seen_questions:
        return ""
    sample = seen_questions[-80:]
    joined = "\n".join(f"- {q}" for q in sample)
    return f"""
AVOID DUPLICATES (CRITICAL):
The student has already received the following questions in this session. Do NOT repeat these questions or generate near-identical duplicates of them; cover fresh high-yield angles, different vignettes, different traps, or remaining testable points from the material:
{joined}
"""


def build_mcq_json_prompt(
    combined_text: str,
    num_lectures: int,
    traps_mode: bool = False,
    cross_links_mode: bool = False,
    is_more: bool = False,
    seen_questions: list[str] | None = None,
) -> str:
    min_q = 15 if is_more else 30
    count_rule = f"- Minimum is {min_q} questions; NEVER output fewer than {min_q} questions."
    multi_rule = (
        f"- You are given {num_lectures} lectures in this study session: distribute questions across ALL lectures AND include cross-lecture integration/comparison questions."
        if num_lectures > 1
        else "- Focus on the high-yield exam points in this single selected lecture."
    )

    if cross_links_mode and num_lectures > 1:
        focus_rules = f"""
- Special Focus (Cross-Lecture Links & Comparisons ONLY): Every single question must explicitly bridge or compare TWO OR MORE of the provided lectures ({num_lectures} lectures total).
- Focus on differential diagnoses across lectures, drug/mechanism/pathway comparisons between different lectures, overlapping clinical presentations, or cause-effect/complication/treatment links spanning multiple lectures.
- Do NOT make isolated single-lecture trivia questions when a cross-lecture link or comparison can be tested.
- In `explanation`, explicitly state which lectures are connected and the exact comparative or mechanistic link between them.
"""
    elif traps_mode:
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

    avoid_block = build_avoid_duplicates_block(seen_questions or [])

    return f"""
Generate high-yield English medical Multiple Choice Questions (MCQs) from the provided lecture material.
Strict count rules:
{count_rule}
- Maximum is open-ended depending on how rich and long the provided lecture material is (generate more than {min_q} when covering multiple lectures or substantial high-yield material).
- Write ALL questions, options, and explanations strictly in English.
{focus_rules}
{avoid_block}
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


def build_mode_prompt(
    mode: str,
    combined_text: str,
    num_lectures: int,
    is_more: bool = False,
    seen_questions: list[str] | None = None,
) -> str:
    multi_note = (
        f"Note: There are {num_lectures} lectures combined below. Cover all of them and explicitly link/compare high-yield concepts across lectures where relevant.\n"
        if num_lectures > 1
        else "Note: Focus strictly on this single selected lecture.\n"
    )
    more_count_note = (
        "Strict count rule: Generate AT LEAST 15 new questions (minimum 15; open maximum depending on lecture richness). Do NOT output fewer than 15 questions.\n"
        if is_more
        else ""
    )
    avoid_block = build_avoid_duplicates_block(seen_questions or [])

    if mode == "saq":
        task_instruction = f"""
{multi_note}{more_count_note}Generate high-yield Short Answer Questions (SAQs) in English based on the most testable points in the provided material.
- Focus on mechanisms, comparisons, clinical reasoning, and key distinctions.
- Provide a concise model answer in English right after each question.
- Do NOT use asterisks (*), markdown headers (#), or markdown tables.
{avoid_block}
"""
    elif mode == "mix":
        task_instruction = f"""
{multi_note}{more_count_note}Generate a focused high-yield Mix of English MCQs (vignettes/exam traps) and concise English SAQs from the most testable parts of the provided material.
- Include clear answers and brief high-yield explanations in English.
- Do NOT use asterisks (*), markdown headers (#), or markdown tables. Present comparisons as clean structured lines instead of tables.
{avoid_block}
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
    has_more_mcq_pdf: bool = False,
    has_summary_pdf: bool = False,
    has_saq_pdf: bool = False,
    has_mix_pdf: bool = False,
    has_lectures: bool = False,
    show_cross_links: bool = False,
    last_question_mode: str | None = None,
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
            InlineKeyboardButton("Anki Ready (.txt)", callback_data="mode_anki"),
        ],
    ]

    if show_cross_links:
        keyboard.append(
            [
                InlineKeyboardButton(
                    "ربط ومقارنات بين المحاضرات (Cross-Links)",
                    callback_data="mode_links",
                )
            ]
        )

    if last_question_mode in ("mcq", "traps", "links", "saq", "mix"):
        mode_labels = {
            "mcq": "MCQ Poll",
            "traps": "MCQ Traps",
            "links": "Cross-Links",
            "saq": "SAQ",
            "mix": "Mix",
        }
        btn_label = f"المزيد من الأسئلة ({mode_labels[last_question_mode]} - حد أدنى 15)"
        keyboard.append(
            [InlineKeyboardButton(btn_label, callback_data=f"more_{last_question_mode}")]
        )

    mcq_pdf_row = []
    if has_mcq_pdf:
        mcq_pdf_row.append(
            InlineKeyboardButton("PDF أسئلة MCQ (الكل)", callback_data="export_mcq_pdf")
        )
    if has_more_mcq_pdf:
        mcq_pdf_row.append(
            InlineKeyboardButton("PDF أسئلة المزيد (+15)", callback_data="export_more_mcq_pdf")
        )
    if mcq_pdf_row:
        keyboard.append(mcq_pdf_row)

    export_buttons = []
    if has_saq_pdf:
        export_buttons.append(
            InlineKeyboardButton("PDF أسئلة SAQ", callback_data="export_saq_pdf")
        )
    if has_mix_pdf:
        export_buttons.append(
            InlineKeyboardButton("PDF أسئلة Mix", callback_data="export_mix_pdf")
        )
    if export_buttons:
        keyboard.append(export_buttons)

    if has_summary_pdf:
        keyboard.append(
            [InlineKeyboardButton("إعادة تحميل الملخص كملف PDF", callback_data="export_summary_pdf")]
        )

    session_row = []
    if has_lectures:
        session_row.append(
            InlineKeyboardButton("اختيار محاضرة محددة", callback_data="select_lecture")
        )
    session_row.append(
        InlineKeyboardButton("حالة الجلسة / الذاكرة", callback_data="session_status")
    )
    session_row.append(
        InlineKeyboardButton("مسح (Clear)", callback_data="clear_lectures")
    )
    keyboard.append(session_row)

    return InlineKeyboardMarkup(keyboard)


def build_lecture_selection_keyboard(
    lectures: list[dict], active_idx: int | None
) -> InlineKeyboardMarkup:
    rows = []
    for i, lec in enumerate(lectures):
        title = str(lec.get("title", f"Lecture {i + 1}"))[:28]
        prefix = "✅ " if active_idx == i else f"{i + 1}. "
        rows.append(
            [InlineKeyboardButton(f"{prefix}{title}", callback_data=f"pick_lec_{i}")]
        )

    all_prefix = "✅ " if active_idx is None else ""
    rows.append(
        [InlineKeyboardButton(f"{all_prefix}الكل (مجمّع - {len(lectures)} محاضرات)", callback_data="pick_lec_all")]
    )
    rows.append(
        [
            InlineKeyboardButton("رجوع للأزرار", callback_data="back_to_menu"),
            InlineKeyboardButton("مسح المحاضرات (Clear)", callback_data="clear_lectures"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def get_active_status_text(context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]) -> str:
    if not lectures:
        return "الوضع الحالي: لا توجد محاضرات مخزنة."
    active_idx = get_active_lecture_index(context, lectures)
    if active_idx is not None:
        title = lectures[active_idx].get("title", f"Lecture {active_idx + 1}")
        return f"الوضع الحالي: محاضرة محددة ({active_idx + 1}: {title})"
    if len(lectures) == 1:
        return f"الوضع الحالي: محاضرة واحدة ({lectures[0].get('title', 'Lecture')})"
    return f"الوضع الحالي: مجمّع ({len(lectures)} محاضرات)"


def build_detailed_session_status(
    context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]
) -> str:
    seen_map = context.user_data.get("seen_questions_by_mode")
    seen_counts = {}
    total_seen = 0
    if isinstance(seen_map, dict):
        for k, v in seen_map.items():
            if isinstance(v, list):
                seen_counts[k] = len(v)
                total_seen += len(v)

    total_chars = sum(len(str(lec.get("text", ""))) for lec in lectures)
    approx_kb = round(total_chars / 1024, 1)

    mcq_saved = len(context.user_data.get("last_mcq_items") or [])
    more_mcq_saved = len(context.user_data.get("last_more_mcq_items") or [])
    has_saq = bool(context.user_data.get("last_saq_text"))
    has_mix = bool(context.user_data.get("last_mix_text"))
    has_sum = bool(context.user_data.get("last_summary_pdf_bytes"))

    mcq_seen = (
        seen_counts.get("mcq", 0)
        + seen_counts.get("traps", 0)
        + seen_counts.get("links", 0)
    )

    lines = [
        "حالة الجلسة والذاكرة (Session Status & GC)",
        f"- المحاضرات المخزنة: {len(lectures)} / {MAX_LECTURES_PER_SESSION}",
        f"- {get_active_status_text(context, lectures)}",
        f"- حجم النصوص المخزنة في الذاكرة: ~{approx_kb} KB",
        f"- إجمالي الأسئلة المسجلة لتفادي التكرار: {total_seen}",
        f"  (MCQ/Traps/Links: {mcq_seen} | SAQ: {seen_counts.get('saq', 0)} | Mix: {seen_counts.get('mix', 0)})",
        f"- العناصر الجاهزة للتصدير: MCQ ({mcq_saved} سؤال | المزيد: {more_mcq_saved}) | SAQ PDF: {'نعم' if has_saq else 'لا'} | Mix PDF: {'نعم' if has_mix else 'لا'} | Summary PDF: {'نعم' if has_sum else 'لا'}",
        "- تم تنفيذ تنظيف فوري للذاكرة المؤقتة (gc.collect).",
    ]
    if lectures:
        lines.append("\nقائمة المحاضرات:")
        for idx, lec in enumerate(lectures, start=1):
            lines.append(f"{idx}. {lec.get('title', f'Lecture {idx}')}")
    return "\n".join(lines)


def reset_generated_session_artifacts(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("last_mcq_items", None)
    context.user_data.pop("last_more_mcq_items", None)
    context.user_data.pop("last_saq_text", None)
    context.user_data.pop("last_mix_text", None)
    context.user_data.pop("last_summary_pdf_bytes", None)
    context.user_data.pop("last_question_mode", None)
    context.user_data.pop("seen_questions_by_mode", None)


def _keyboard_kwargs_from_state(
    context: ContextTypes.DEFAULT_TYPE, lectures: list[dict]
) -> dict:
    return {
        "has_mcq_pdf": bool(context.user_data.get("last_mcq_items")),
        "has_more_mcq_pdf": bool(context.user_data.get("last_more_mcq_items")),
        "has_summary_pdf": bool(context.user_data.get("last_summary_pdf_bytes")),
        "has_saq_pdf": bool(context.user_data.get("last_saq_text")),
        "has_mix_pdf": bool(context.user_data.get("last_mix_text")),
        "has_lectures": bool(lectures),
        "show_cross_links": can_use_cross_lecture_links(context, lectures),
        "last_question_mode": context.user_data.get("last_question_mode"),
    }


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lectures = get_session_lectures(context)
    welcome_text = (
        "أهلًا بك في بوت مذاكرة الطب (High-Yield - English Exams)\n\n"
        f"1. أرسل ملف محاضرة بصيغة PDF (ويمكنك تجميع حتى {MAX_LECTURES_PER_SESSION} محاضرات في الجلسة).\n"
        "2. اختر من الأزرار لتوليد محتوى لمحاضرة محددة أو مجمّع على الكل:\n"
        "   - MCQ Poll (أسئلة تفاعلية كـ Poll بحد أدنى 30 سؤال + تصدير PDF بخط واضح)\n"
        "   - MCQ Traps & Mix (أسئلة MCQ مليانة فخاخ امتحانات ومتشابهات وتجميعات)\n"
        "   - Cross-Links (يظهر عند وجود أكتر من محاضرة لربط ومقارنة المحاضرات)\n"
        "   - SAQ (أسئلة قصيرة بالإنجليزية + تصدير PDF منسق)\n"
        "   - Mix (مزيج MCQ + SAQ بدون نجوم + تصدير PDF منسق)\n"
        "   - Summary (PDF) (ملخص High-Yield منسق بصيغة PDF بخط واضح)\n"
        "   - Anki Ready (.txt) (ملف جاهز للاستيراد المباشر بفاصل Tab)\n\n"
        f"{get_active_status_text(context, lectures)}"
    )
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not document:
        return

    file_name = document.file_name or "lecture.pdf"
    if not file_name.lower().endswith(".pdf"):
        await update.message.reply_text("من فضلك أرسل ملف المحاضرة بصيغة PDF.")
        return

    status_msg = await update.message.reply_text("جاري استلام المحاضرة وتحليل محتواها...")

    try:
        tg_file = await document.get_file()
        pdf_bytes = bytes(await tg_file.download_as_bytearray())
        lecture_text = await asyncio.to_thread(
            extract_text_from_pdf, pdf_bytes, MAX_CHARS_SINGLE_LECTURE
        )
        del pdf_bytes
        gc.collect()

        if not lecture_text.strip():
            await status_msg.edit_text(
                "لم أتمكن من استخراج نص واضح من هذا الـ PDF (قد يكون صورًا فقط). جرّب ملف PDF يحتوي على نص."
            )
            return

        lectures = get_session_lectures(context)
        if len(lectures) >= MAX_LECTURES_PER_SESSION:
            removed = lectures.pop(0)
            logger.info("Removed oldest lecture %s to stay within limit", removed.get("title"))

        clean_title = os.path.splitext(file_name)[0][:40]
        lectures.append({"title": clean_title, "text": lecture_text})
        context.user_data.pop("active_lecture_idx", None)
        reset_generated_session_artifacts(context)

        await status_msg.edit_text(
            f"تمت إضافة المحاضرة: {clean_title}\n"
            f"إجمالي المحاضرات في الجلسة: {len(lectures)}/{MAX_LECTURES_PER_SESSION}\n"
            f"{get_active_status_text(context, lectures)}\n"
            "تقدر تبعت محاضرة كمان، أو تختار من الأزرار دلوقتي:",
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
    except Exception as e:
        logger.error("Error handling document: %s", e)
        await status_msg.edit_text(f"حدث خطأ أثناء قراءة الملف: {type(e).__name__}")


async def send_long_text(chat, text: str) -> None:
    cleaned = clean_plain_text(text)
    chunk_size = 3800
    for i in range(0, len(cleaned), chunk_size):
        await chat.send_message(cleaned[i : i + chunk_size])


def parse_mcq_json(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    data = json.loads(cleaned)
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    valid_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = clean_plain_text(str(item.get("question", "")))[:280]
        opts = [
            clean_plain_text(str(o))[:90]
            for o in (item.get("options") or [])
            if clean_plain_text(str(o))
        ][:10]
        correct_id = item.get("correct_option_id", 0)
        exp = clean_plain_text(str(item.get("explanation", "")))

        if q and len(opts) >= 2 and isinstance(correct_id, int) and 0 <= correct_id < len(opts):
            valid_items.append(
                {
                    "question": q,
                    "options": opts,
                    "correct_option_id": correct_id,
                    "explanation": exp,
                }
            )
    return valid_items


async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    action = query.data
    lectures = get_session_lectures(context)
    active_idx = get_active_lecture_index(context, lectures)

    if action == "session_status":
        gc.collect()
        status_report = build_detailed_session_status(context, lectures)
        await query.message.reply_text(
            status_report,
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
        return

    if action == "select_lecture":
        if not lectures:
            await query.message.reply_text(
                "لا توجد محاضرات مخزنة حاليًا لإختيار محاضرة محددة. أرسل ملف PDF أولًا.",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
            return

        await query.message.reply_text(
            "اختر المحاضرة اللي عايز تولّد عليها المحتوى، أو اختر (الكل - مجمّع):",
            reply_markup=build_lecture_selection_keyboard(lectures, active_idx),
        )
        return

    if action.startswith("pick_lec_"):
        choice = action.replace("pick_lec_", "")
        reset_generated_session_artifacts(context)

        if choice == "all":
            context.user_data.pop("active_lecture_idx", None)
            status_msg = f"تم تفعيل الوضع المجمّع على كل المحاضرات ({len(lectures)} محاضرة)."
        else:
            try:
                idx = int(choice)
                if 0 <= idx < len(lectures):
                    context.user_data["active_lecture_idx"] = idx
                    lec_title = lectures[idx].get("title", f"Lecture {idx + 1}")
                    status_msg = f"تم اختيار المحاضرة ({idx + 1}): {lec_title}\nسيتم التوليد على هذه المحاضرة فقط."
                else:
                    status_msg = "اختيار غير صالح."
            except ValueError:
                status_msg = "اختيار غير صالح."

        await query.message.edit_text(
            f"{status_msg}\nاختر نوع المحتوى المطلوب:",
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
        return

    if action == "back_to_menu":
        await query.message.edit_text(
            f"{get_active_status_text(context, lectures)}\nاختر من الأزرار التالية:",
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
        return

    if action == "clear_lectures":
        context.user_data.clear()
        gc.collect()
        await query.message.reply_text(
            "تم مسح جميع المحاضرات والملفات المؤقتة وتنظيف الذاكرة. أرسل ملفات PDF جديدة للبدء.",
            reply_markup=get_choices_keyboard(has_lectures=False),
        )
        return

    target_lectures, session_label = get_target_lectures_and_label(context, lectures)

    if action == "export_mcq_pdf":
        mcq_items = context.user_data.get("last_mcq_items")
        if not mcq_items:
            await query.message.reply_text("لا توجد أسئلة MCQ محفوظة حاليًا. اضغط على MCQ Poll أو MCQ Traps & Mix أولًا.")
            return

        pdf_wait = await query.message.reply_text("جاري تجهيز ملف الـ PDF للأسئلة الكاملة بخط واضح...")
        try:
            pdf_bytes = await asyncio.to_thread(
                generate_mcq_pdf_bytes, session_label, mcq_items, "High-Yield MCQ (All)"
            )
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{session_label}_MCQ_All.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_MCQ_All.pdf",
                caption=f"High-Yield MCQ PDF - All ({len(mcq_items)} questions) - {session_label}.",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
            del pdf_bytes
            gc.collect()
        except Exception as e:
            logger.error("Error generating MCQ PDF: %s", e)
            await pdf_wait.edit_text(f"حدث خطأ أثناء إنشاء ملف الـ PDF: {type(e).__name__}")
        return

    if action == "export_more_mcq_pdf":
        more_items = context.user_data.get("last_more_mcq_items")
        if not more_items:
            await query.message.reply_text("لا توجد أسئلة إضافية محفوظة حاليًا. اضغط على زر المزيد من الأسئلة أولًا.")
            return

        pdf_wait = await query.message.reply_text("جاري تجهيز ملف الـ PDF لأسئلة المزيد فقط بخط واضح...")
        try:
            pdf_bytes = await asyncio.to_thread(
                generate_mcq_pdf_bytes, session_label, more_items, "High-Yield MCQ (Additional Batch)"
            )
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{session_label}_MCQ_More.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_MCQ_More.pdf",
                caption=f"High-Yield MCQ PDF - Additional Only ({len(more_items)} questions) - {session_label}.",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
            del pdf_bytes
            gc.collect()
        except Exception as e:
            logger.error("Error generating More MCQ PDF: %s", e)
            await pdf_wait.edit_text(f"حدث خطأ أثناء إنشاء ملف الـ PDF: {type(e).__name__}")
        return

    if action in ("export_saq_pdf", "export_mix_pdf"):
        is_saq = action == "export_saq_pdf"
        stored_text = context.user_data.get("last_saq_text" if is_saq else "last_mix_text")
        doc_label = "High-Yield SAQ" if is_saq else "High-Yield Mix (MCQ + SAQ)"
        suffix = "SAQ" if is_saq else "Mix"

        if not stored_text:
            await query.message.reply_text(
                f"لا توجد أسئلة {suffix} محفوظة حاليًا. اضغط على زر {suffix} أولًا."
            )
            return

        pdf_wait = await query.message.reply_text(f"جاري تجهيز ملف الـ PDF لأسئلة {suffix} بخط واضح...")
        try:
            pdf_bytes = await asyncio.to_thread(
                generate_text_document_pdf_bytes, doc_label, session_label, stored_text
            )
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{session_label}_{suffix}.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_{suffix}.pdf",
                caption=f"{doc_label} PDF - {session_label}.",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
            del pdf_bytes
            gc.collect()
        except Exception as e:
            logger.error("Error generating %s PDF: %s", suffix, e)
            await pdf_wait.edit_text(f"حدث خطأ أثناء إنشاء ملف الـ PDF: {type(e).__name__}")
        return

    if action == "export_summary_pdf":
        summary_pdf_bytes = context.user_data.get("last_summary_pdf_bytes")
        if not summary_pdf_bytes:
            await query.message.reply_text("لا يوجد ملخص محفوظ حاليًا. اضغط على Summary (PDF) أولًا.")
            return

        pdf_io = io.BytesIO(summary_pdf_bytes)
        pdf_io.name = f"{session_label}_Summary.pdf"
        await query.message.reply_document(
            document=pdf_io,
            filename=f"{session_label}_Summary.pdf",
            caption=f"High-Yield Summary PDF - {session_label}.",
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
        return

    if not target_lectures:
        await query.message.reply_text(
            "من فضلك أرسل ملف PDF لمحاضرة واحدة على الأقل أولًا.",
            reply_markup=get_choices_keyboard(has_lectures=False),
        )
        return

    is_more = action.startswith("more_")
    mode = action.replace("more_", "") if is_more else action.replace("mode_", "")

    if mode == "links" and not can_use_cross_lecture_links(context, lectures):
        await query.message.reply_text(
            "خيار ربط ومقارنات بين المحاضرات متاح عند وجود محاضرتين أو أكثر في الوضع المجمّع.",
            reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
        )
        return

    combined_text = build_combined_lecture_text(target_lectures)
    num_lectures = len(target_lectures)
    scope_desc = session_label if num_lectures == 1 else f"{num_lectures} محاضرات مجمعة"

    if is_more:
        wait_msg = await query.message.reply_text(
            f"جاري توليد أسئلة إضافية جديدة (حد أدنى 15 سؤال بدون تكرار - {scope_desc})..."
        )
    else:
        wait_msg = await query.message.reply_text(
            f"جاري إعداد المحتوى الطبي المركز ({scope_desc})..."
        )

    try:
        if mode in ("mcq", "traps", "links"):
            is_traps = mode == "traps"
            is_links = mode == "links"
            seen_list = get_seen_questions_list(context, mode) if is_more else []
            prompt = build_mcq_json_prompt(
                combined_text,
                num_lectures=num_lectures,
                traps_mode=is_traps,
                cross_links_mode=is_links,
                is_more=is_more,
                seen_questions=seen_list,
            )
            response = await generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                    temperature=0.38 if (is_traps or is_links or is_more) else 0.3,
                    response_mime_type="application/json",
                ),
            )
            raw_out = response.text or ""
            new_mcq_items = parse_mcq_json(raw_out)

            if not new_mcq_items:
                await wait_msg.edit_text("لم أتمكن من تنسيق أسئلة الـ Poll بشكل صحيح. حاول مرة أخرى.")
                return

            existing_mcq_items = context.user_data.get("last_mcq_items")
            if not isinstance(existing_mcq_items, list):
                existing_mcq_items = []

            existing_q_lower = {str(x.get("question", "")).strip().lower() for x in existing_mcq_items}
            truly_new_items = [
                it for it in new_mcq_items if str(it.get("question", "")).strip().lower() not in existing_q_lower
            ]
            items_to_send = truly_new_items if truly_new_items else new_mcq_items

            start_number = len(existing_mcq_items) + 1 if is_more else 1
            if is_more:
                existing_mcq_items.extend(items_to_send)
                context.user_data["last_mcq_items"] = existing_mcq_items
                prev_more = context.user_data.get("last_more_mcq_items")
                if not isinstance(prev_more, list):
                    prev_more = []
                prev_more.extend(items_to_send)
                context.user_data["last_more_mcq_items"] = prev_more
            else:
                context.user_data["last_mcq_items"] = items_to_send
                context.user_data.pop("last_more_mcq_items", None)

            record_seen_questions(
                context,
                mode,
                [str(it.get("question", "")) for it in items_to_send],
            )
            context.user_data["last_question_mode"] = mode

            if is_links:
                label = "Cross-Lecture Links"
            elif is_traps:
                label = "MCQ Traps & Mix"
            else:
                label = "MCQ"

            await wait_msg.edit_text(
                f"تم توليد {len(items_to_send)} سؤال {label} جديد ({scope_desc}). جاري إرسال الاستفتاءات..."
            )

            for offset, item in enumerate(items_to_send):
                q_num = start_number + offset
                await query.message.reply_poll(
                    question=f"{q_num}. {item['question']}"[:295],
                    options=item["options"],
                    type="quiz",
                    correct_option_id=item["correct_option_id"],
                    is_anonymous=False,
                )
                await asyncio.sleep(0.15)

            total_saved = len(context.user_data["last_mcq_items"])
            more_saved = len(context.user_data.get("last_more_mcq_items") or [])
            more_info = f" | الأسئلة الإضافية: {more_saved} سؤال" if more_saved else ""
            await query.message.reply_text(
                f"تم إرسال {len(items_to_send)} سؤال ({label}) كـ Poll بالإنجليزية (إجمالي المحفوظ: {total_saved} سؤال{more_info}).\n"
                "تقدر تطلب المزيد (حد أدنى 15 سؤال بدون تكرار) أو تحمّل ملف الـ PDF الكامل أو ملف المزيد لوحده.",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
            return

        seen_list = get_seen_questions_list(context, mode) if (is_more and mode in ("saq", "mix")) else []
        prompt = build_mode_prompt(
            mode,
            combined_text,
            num_lectures=num_lectures,
            is_more=is_more,
            seen_questions=seen_list,
        )
        response = await generate_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                temperature=0.35 if is_more else 0.3,
            ),
        )

        raw_output = response.text or ""
        if not raw_output.strip():
            await wait_msg.edit_text("لم يتمكن النموذج من توليد رد مناسب. حاول مرة أخرى.")
            return

        await wait_msg.delete()

        if mode in ("saq", "mix"):
            context.user_data["last_question_mode"] = mode
            extracted_q = extract_question_lines_from_text(raw_output)
            record_seen_questions(context, mode, extracted_q)

        if mode == "summary":
            output_text = clean_plain_text(raw_output)
            summary_pdf_bytes = await asyncio.to_thread(
                generate_text_document_pdf_bytes, "High-Yield Summary", session_label, output_text
            )
            context.user_data["last_summary_pdf_bytes"] = summary_pdf_bytes
            pdf_io = io.BytesIO(summary_pdf_bytes)
            pdf_io.name = f"{session_label}_Summary.pdf"

            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_Summary.pdf",
                caption=f"ملخص High-Yield ({scope_desc}) بصيغة PDF بخط واضح. تحب تعمل حاجة تانية؟",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
        elif mode == "anki":
            anki_content = format_anki_tab_deck(raw_output)
            if not anki_content:
                anki_content = clean_plain_text(raw_output)

            anki_bytes = io.BytesIO(anki_content.encode("utf-8"))
            anki_bytes.name = f"{session_label}_anki.txt"
            cards_count = len(anki_content.splitlines()) if anki_content else 0

            await query.message.reply_document(
                document=anki_bytes,
                filename=f"{session_label}_anki.txt",
                caption=f"Anki deck ready ({cards_count} cards, Separator: Tab). تحب تعمل حاجة تانية؟",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )
        else:
            output_text = clean_plain_text(raw_output)
            if mode == "saq":
                prev_saq = context.user_data.get("last_saq_text")
                context.user_data["last_saq_text"] = (
                    f"{prev_saq}\n\n--- Additional Questions ---\n\n{output_text}"
                    if (is_more and prev_saq)
                    else output_text
                )
            elif mode == "mix":
                prev_mix = context.user_data.get("last_mix_text")
                context.user_data["last_mix_text"] = (
                    f"{prev_mix}\n\n--- Additional Questions ---\n\n{output_text}"
                    if (is_more and prev_mix)
                    else output_text
                )

            await send_long_text(query.message.chat, output_text)
            await query.message.reply_text(
                f"تحب تولّد المزيد من الأسئلة الجديدة ({scope_desc}) أو تحمّلهم PDF؟",
                reply_markup=get_choices_keyboard(**_keyboard_kwargs_from_state(context, lectures)),
            )

    except Exception as e:
        logger.error("Error generating content: %s", e)
        await wait_msg.edit_text(f"حدث خطأ أثناء التوليد ({type(e).__name__}). حاول مرة أخرى بعد قليل.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        raise ValueError("تأكد من ضبط TELEGRAM_BOT_TOKEN و GEMINI_API_KEY في متغيرات البيئة.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(CallbackQueryHandler(handle_button_click))

    logger.info("Bot started polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
