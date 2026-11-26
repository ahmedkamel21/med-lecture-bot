import asyncio
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
MAX_LECTURES_PER_SESSION = 5
MAX_CHARS_PER_LECTURE = 11000
MAX_COMBINED_CHARS = 42000

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


def build_combined_lecture_text(lectures: list[dict]) -> str:
    combined_parts = []
    total_len = 0
    for idx, lec in enumerate(lectures, start=1):
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


def get_session_summary_label(lectures: list[dict]) -> str:
    if not lectures:
        return "Lectures"
    if len(lectures) == 1:
        return lectures[0].get("title", "Lecture")[:40]
    return f"{len_words_clean(len(lectures))}" if False else f"{len(lectures)}_Combined_Lectures"


def len_words_clean(n: int) -> str:
    return f"{n}_Lectures"


def build_mcq_json_prompt(combined_text: str, num_lectures: int, traps_mode: bool = False) -> str:
    multi_rule = (
        f"- You are given {num_lectures} lectures in this study session: distribute questions across ALL lectures AND include cross-lecture integration/comparison questions."
        if num_lectures > 1
        else "- Focus on the high-yield exam points in this lecture."
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
        else ""
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
{multi_note}Create Anki flashcards in English ready for direct import using pipe ( | ) separation ONLY, one card per line, with NO numbering, intro, or extra commentary.
Format per line:
Question or clinical vignette in English | Concise high-yield answer + key differentiator in English
Focus strictly on high-yield exam points and avoid asterisks (*).
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
            InlineKeyboardButton("Anki Ready (.txt)", callback_data="mode_anki"),
        ],
    ]
    if has_mcq_pdf:
        keyboard.append(
            [InlineKeyboardButton("تحميل أسئلة الـ MCQ كملف PDF", callback_data="export_mcq_pdf")]
        )
    if has_summary_pdf:
        keyboard.append(
            [InlineKeyboardButton("إعادة تحميل الملخص كملف PDF", callback_data="export_summary_pdf")]
        )
    if has_lectures:
        keyboard.append(
            [
                InlineKeyboardButton("قائمة المحاضرات المجمعة", callback_data="list_lectures"),
                InlineKeyboardButton("مسح المحاضرات (Clear)", callback_data="clear_lectures"),
            ]
        )
    return InlineKeyboardMarkup(keyboard)


def format_lectures_list_text(lectures: list[dict]) -> str:
    if not lectures:
        return "لا توجد محاضرات مخزنة حاليًا في الجلسة. أرسل ملف PDF للبدء."
    lines = [f"المحاضرات المخزنة مؤقتًا في الجلسة ({len(lectures)}/{MAX_LECTURES_PER_SESSION}):"]
    for i, lec in enumerate(lectures, start=1):
        lines.append(f"{i}. {lec.get('title', 'Lecture')}")
    lines.append("\nيمكنك إرسال محاضرة إضافية لضمها للمجموعة، أو اختيار أي زر لتوليد محتوى مجمع على الكل.")
    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "أهلًا بك في بوت مذاكرة الطب (High-Yield - English Exams)\n\n"
        f"1. أرسل ملف محاضرة بصيغة PDF (أو أرسل عدة محاضرات ورا بعض حتى {MAX_LECTURES_PER_SESSION} محاضرات لتجميعهم مؤقتًا في نفس الجلسة).\n"
        "2. اختر من الأزرار لتوليد محتوى مجمّع أو لمحاضرة واحدة:\n"
        "   - MCQ Poll (أسئلة تفاعلية كـ Poll بحد أدنى 30 سؤال + إمكانية تحميلها PDF بخط واضح)\n"
        "   - MCQ Traps & Mix (أسئلة MCQ مليانة فخاخ امتحانات ومتشابهات وتجميعات)\n"
        "   - SAQ (أسئلة قصيرة بالإنجليزية مع إجابات نموذجية)\n"
        "   - Mix (مزيج MCQ + SAQ بتنسيق نظيف بدون نجوم)\n"
        "   - Summary (ملخص High-Yield بتنسيق جميل وملف PDF بخط واضح)\n"
        "   - Anki Ready (.txt) (ملف جاهز للاستيراد المباشر في Anki بالإنجليزية)\n\n"
        "التركيز الأساسي دائمًا على الأجزاء المهمة وموضع السؤال وتجميع الروابط بين المحاضرات."
    )
    lectures = get_session_lectures(context)
    await update.message.reply_text(
        welcome_text,
        reply_markup=get_choices_keyboard(
            has_mcq_pdf=bool(context.user_data.get("last_mcq_items")),
            has_summary_pdf=bool(context.user_data.get("last_summary_pdf_bytes")),
            has_lectures=bool(lectures),
        ),
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
        lecture_text = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)
        del pdf_bytes

        if not lecture_text.strip():
            await status_msg.edit_text(
                "لم أتمكن من استخراج نص واضح من هذا الـ PDF (قد يكون صورًا فقط). جرّب ملف PDF يحتوي على نص."
            )
            return

        lectures = get_session_lectures(context)
        if len(lectures) >= MAX_LECTURES_PER_SESSION:
            # استبدال أقدم محاضرة للحفاظ على الرامات ضمن الحدود الآمنة
            removed = lectures.pop(0)
            logger.info("Removed oldest lecture %s to stay within limit", removed.get("title"))

        clean_title = os.path.splitext(file_name)[0][:40]
        lectures.append({"title": clean_title, "text": lecture_text})
        context.user_data.pop("last_mcq_items", None)
        context.user_data.pop("last_summary_pdf_bytes", None)

        await status_msg.edit_text(
            f"تمت إضافة المحاضرة: {clean_title}\n"
            f"إجمالي المحاضرات المجمعة حاليًا: {len(lectures)}/{MAX_LECTURES_PER_SESSION}\n"
            "تقدر تبعت محاضرة كمان لضمها، أو تختار من الأزرار دلوقتي:",
            reply_markup=get_choices_keyboard(
                has_mcq_pdf=False,
                has_summary_pdf=False,
                has_lectures=True,
            ),
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
    has_mcq_pdf = bool(context.user_data.get("last_mcq_items"))
    has_summary_pdf = bool(context.user_data.get("last_summary_pdf_bytes"))
    has_lectures = bool(lectures)
    session_label = get_session_summary_label(lectures)

    if action == "list_lectures":
        await query.message.reply_text(
            format_lectures_list_text(lectures),
            reply_markup=get_choices_keyboard(
                has_mcq_pdf=has_mcq_pdf,
                has_summary_pdf=has_summary_pdf,
                has_lectures=has_lectures,
            ),
        )
        return

    if action == "clear_lectures":
        context.user_data.clear()
        await query.message.reply_text(
            "تم مسح جميع المحاضرات والملفات المؤقتة من الجلسة. أرسل ملفات PDF جديدة للبدء."
        )
        return

    if action == "export_mcq_pdf":
        mcq_items = context.user_data.get("last_mcq_items")
        if not mcq_items:
            await query.message.reply_text("لا توجد أسئلة MCQ محفوظة حاليًا. اضغط على MCQ Poll أو MCQ Traps & Mix أولًا.")
            return

        pdf_wait = await query.message.reply_text("جاري تجهيز ملف الـ PDF للأسئلة بخط واضح...")
        try:
            pdf_bytes = await asyncio.to_thread(generate_mcq_pdf_bytes, session_label, mcq_items)
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{session_label}_MCQ.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_MCQ.pdf",
                caption=f"High-Yield MCQ PDF ({len(mcq_items)} questions with answers & explanations).",
                reply_markup=get_choices_keyboard(
                    has_mcq_pdf=True,
                    has_summary_pdf=has_summary_pdf,
                    has_lectures=has_lectures,
                ),
            )
        except Exception as e:
            logger.error("Error generating MCQ PDF: %s", e)
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
            caption="High-Yield Summary PDF.",
            reply_markup=get_choices_keyboard(
                has_mcq_pdf=has_mcq_pdf,
                has_summary_pdf=True,
                has_lectures=has_lectures,
            ),
        )
        return

    if not lectures:
        await query.message.reply_text("من فضلك أرسل ملف PDF لمحاضرة واحدة على الأقل أولًا.")
        return

    combined_text = build_combined_lecture_text(lectures)
    num_lectures = len(lectures)
    mode = action.replace("mode_", "")
    wait_msg = await query.message.reply_text(
        f"جاري إعداد المحتوى الطبي المركز ({num_lectures} محاضرة)..."
    )

    try:
        if mode in ("mcq", "traps"):
            is_traps = mode == "traps"
            prompt = build_mcq_json_prompt(combined_text, num_lectures=num_lectures, traps_mode=is_traps)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                    temperature=0.35 if is_traps else 0.3,
                    response_mime_type="application/json",
                ),
            )
            raw_out = response.text or ""
            mcq_items = parse_mcq_json(raw_out)

            if not mcq_items:
                await wait_msg.edit_text("لم أتمكن من تنسيق أسئلة الـ Poll بشكل صحيح. حاول مرة أخرى.")
                return

            context.user_data["last_mcq_items"] = mcq_items
            label = "MCQ Traps & Mix" if is_traps else "MCQ"
            await wait_msg.edit_text(
                f"تم توليد {len(mcq_items)} سؤال {label} من {num_lectures} محاضرة. جاري إرسال الاستفتاءات..."
            )

            for idx, item in enumerate(mcq_items, start=1):
                await query.message.reply_poll(
                    question=f"{idx}. {item['question']}"[:295],
                    options=item["options"],
                    type="quiz",
                    correct_option_id=item["correct_option_id"],
                    is_anonymous=False,
                )
                await asyncio.sleep(0.15)

            await query.message.reply_text(
                f"تم إرسال {len(mcq_items)} سؤال ({label}) كـ Poll بالإنجليزية. الشرح التفصيلي والفخاخ موجودة داخل ملف الـ PDF.\n"
                "تحب تحمّل الأسئلة دي كملف PDF أو تختار حاجة تانية؟",
                reply_markup=get_choices_keyboard(
                    has_mcq_pdf=True,
                    has_summary_pdf=has_summary_pdf,
                    has_lectures=True,
                ),
            )
            return

        prompt = build_mode_prompt(mode, combined_text, num_lectures=num_lectures)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )

        output_text = clean_plain_text(response.text or "")
        if not output_text:
            await wait_msg.edit_text("لم يتمكن النموذج من توليد رد مناسب. حاول مرة أخرى.")
            return

        await wait_msg.delete()

        if mode == "summary":
            summary_pdf_bytes = await asyncio.to_thread(
                generate_summary_pdf_bytes, session_label, output_text
            )
            context.user_data["last_summary_pdf_bytes"] = summary_pdf_bytes
            pdf_io = io.BytesIO(summary_pdf_bytes)
            pdf_io.name = f"{session_label}_Summary.pdf"

            await query.message.reply_document(
                document=pdf_io,
                filename=f"{session_label}_Summary.pdf",
                caption=f"ملخص High-Yield مجمع لـ {num_lectures} محاضرة بصيغة PDF بخط واضح. تحب تعمل حاجة تانية؟",
                reply_markup=get_choices_keyboard(
                    has_mcq_pdf=has_mcq_pdf,
                    has_summary_pdf=True,
                    has_lectures=True,
                ),
            )
        elif mode == "anki":
            cleaned_lines = []
            for line in output_text.splitlines():
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("```"):
                    continue
                if "|" in line_stripped:
                    cleaned_lines.append(line_stripped)

            anki_content = "\n".join(cleaned_lines) if cleaned_lines else output_text
            anki_bytes = io.BytesIO(anki_content.encode("utf-8"))
            anki_bytes.name = f"{session_label}_anki.txt"

            await query.message.reply_document(
                document=anki_bytes,
                filename=f"{session_label}_anki.txt",
                caption="Anki deck ready for direct import (Separator: |). تحب تعمل حاجة تانية؟",
                reply_markup=get_choices_keyboard(
                    has_mcq_pdf=has_mcq_pdf,
                    has_summary_pdf=has_summary_pdf,
                    has_lectures=True,
                ),
            )
        else:
            await send_long_text(query.message.chat, output_text)
            await query.message.reply_text(
                "تحب تولّد حاجة تانية من المحاضرات المجمعة؟",
                reply_markup=get_choices_keyboard(
                    has_mcq_pdf=has_mcq_pdf,
                    has_summary_pdf=has_summary_pdf,
                    has_lectures=True,
                ),
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
