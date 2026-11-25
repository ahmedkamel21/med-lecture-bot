import asyncio
import io
import json
import logging
import os
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
MAX_CHARS = 45000

ARABIC_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
)
ARABIC_FONT_LOCAL_PATH = "/tmp/Amiri-Regular.ttf"

MEDICAL_HIGH_YIELD_SYSTEM_PROMPT = """
You are a high-yield medical exam tutor helping a medical student study efficiently for English medical exams.
Golden Rule: Focus heavily on high-yield exam topics ("what gets tested"):
- Core mechanisms, pharmacology/pathophysiology principles, and key clinical concepts.
- Classic clinical vignettes, distinguishing presentations, and high-yield indications/contraindications/adverse effects.
- Crucial comparisons, differential points, and critical exceptions.
Strictly avoid trivial facts or low-yield filler details.
Write ALL medical content, questions, choices, answers, summaries, explanations, and Anki cards strictly in clear, concise, exam-oriented English.
"""


def get_font_path() -> str | None:
    if os.path.exists(ARABIC_FONT_LOCAL_PATH) and os.path.getsize(ARABIC_FONT_LOCAL_PATH) > 10000:
        return ARABIC_FONT_LOCAL_PATH
    try:
        resp = requests.get(ARABIC_FONT_URL, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 10000:
            with open(ARABIC_FONT_LOCAL_PATH, "wb") as f:
                f.write(resp.content)
            return ARABIC_FONT_LOCAL_PATH
    except Exception as e:
        logger.warning("Could not download font: %s", e)
    return None


def generate_mcq_pdf_bytes(lecture_title: str, mcq_items: list[dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = get_font_path()
    if font_path:
        pdf.add_font("UniFont", "", font_path)
        pdf.set_font("UniFont", size=14)
    else:
        pdf.set_font("Helvetica", size=14)

    pdf.cell(0, 10, text=f"High-Yield MCQ ({len(mcq_items)} Questions) - {lecture_title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("UniFont" if font_path else "Helvetica", size=11)

    for idx, item in enumerate(mcq_items, start=1):
        question_text = str(item.get("question", "")).strip()
        options = item.get("options") or []
        correct_idx = item.get("correct_option_id", 0)
        explanation = str(item.get("explanation", "")).strip()

        pdf.multi_cell(0, 7, text=f"Q{idx}: {question_text}")
        for opt_i, opt_val in enumerate(options):
            letter = chr(65 + opt_i) if opt_i < 26 else str(opt_i + 1)
            pdf.multi_cell(0, 6, text=f"   {letter}) {str(opt_val).strip()}")

        correct_letter = chr(65 + correct_idx) if isinstance(correct_idx, int) and 0 <= correct_idx < len(options) else "A"
        correct_label = options[correct_idx] if isinstance(correct_idx, int) and 0 <= correct_idx < len(options) else ""
        pdf.multi_cell(0, 6, text=f"Correct Answer: ({correct_letter}) {correct_label}")

        if explanation:
            pdf.multi_cell(0, 6, text=f"Explanation: {explanation}")

        pdf.ln(4)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_parts = []
    current_len = 0
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            part = f"--- Page {idx} ---\n{text}"
            if current_len + len(part) > MAX_CHARS:
                remaining = MAX_CHARS - current_len
                if remaining > 200:
                    extracted_parts.append(part[:remaining])
                extracted_parts.append("\n[Truncated to keep focus on high-yield content]")
                break
            extracted_parts.append(part)
            current_len += len(part)
    return "\n\n".join(extracted_parts)


def build_mcq_json_prompt(lecture_text: str) -> str:
    return f"""
Generate high-yield English medical Multiple Choice Questions (MCQs) from this lecture.
Strict count rules:
- Minimum is 30 questions; NEVER output fewer than 30 questions.
- Maximum is open-ended depending on how rich and long the lecture is (generate more than 30 when the lecture covers substantial high-yield material).
- Write ALL questions, options, and explanations strictly in English.
- Prioritize clinical vignettes, mechanisms, key distinctions, contraindications, adverse effects, and classic exam traps; avoid trivial details.
- Each question must have exactly 4 options (list of 4 concise strings, each <= 85 characters for Telegram polls).
- Question stem must be <= 260 characters.
- Set `correct_option_id` as an integer from 0 to 3.
- Include a concise, high-yield `explanation` in English.

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

Lecture content:
\"\"\"
{lecture_text}
\"\"\"
"""


def build_mode_prompt(mode: str, lecture_text: str) -> str:
    if mode == "saq":
        task_instruction = """
Generate high-yield Short Answer Questions (SAQs) in English based on the most testable points in this lecture.
- Focus on mechanisms, comparisons, clinical reasoning, and key distinctions.
- Provide a concise, bulleted model answer in English right after each question.
"""
    elif mode == "mix":
        task_instruction = """
Generate a focused high-yield Mix of English MCQs (vignettes/exam traps) and concise English SAQs from the most testable parts of this lecture.
- Include clear answers and brief high-yield explanations in English.
"""
    elif mode == "summary":
        task_instruction = """
Create a high-yield English medical exam review summary of this lecture.
- Organize with clear headings and bullet points.
- Highlight key mechanisms, comparisons, exceptions, and clinical/exam correlations. Remove fluff.
"""
    elif mode == "anki":
        task_instruction = """
Create Anki flashcards in English ready for direct import using pipe ( | ) separation ONLY, one card per line, with NO numbering, intro, or extra commentary.
Format per line:
Question or clinical vignette in English | Concise high-yield answer + key differentiator in English
Focus strictly on high-yield exam points.
"""
    else:
        task_instruction = "Summarize the high-yield exam points of this lecture in English."

    return f"""{task_instruction}

Lecture content:
\"\"\"
{lecture_text}
\"\"\"
"""


def get_choices_keyboard(has_mcq_pdf: bool = False) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("MCQ Poll (High-Yield)", callback_data="mode_mcq"),
            InlineKeyboardButton("SAQ", callback_data="mode_saq"),
        ],
        [
            InlineKeyboardButton("Mix (MCQ + SAQ)", callback_data="mode_mix"),
            InlineKeyboardButton("Summary", callback_data="mode_summary"),
        ],
        [
            InlineKeyboardButton("Anki Ready (.txt)", callback_data="mode_anki"),
        ],
    ]
    if has_mcq_pdf:
        keyboard.append(
            [InlineKeyboardButton("تحميل أسئلة الـ MCQ كملف PDF", callback_data="export_mcq_pdf")]
        )
    return InlineKeyboardMarkup(keyboard)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "أهلًا بك في بوت مذاكرة الطب (High-Yield - English Exams)\n\n"
        "1. أرسل لي ملف المحاضرة بصيغة **PDF**.\n"
        "2. سأحلل الملف وأعرض لك أزرار اختيار (كل الأسئلة والملخصات والـ Anki بالإنجليزية):\n"
        "   • **MCQ Poll** (أسئلة تفاعلية كـ Poll بحد أدنى 30 سؤال + إمكانية تحميلها PDF مع الشرح)\n"
        "   • **SAQ** (أسئلة قصيرة بالإنجليزية مع إجابات نموذجية)\n"
        "   • **Mix** (مزيج MCQ + SAQ بالإنجليزية)\n"
        "   • **Summary** (ملخص High-Yield بالإنجليزية)\n"
        "   • **Anki Ready (.txt)** (ملف جاهز للاستيراد المباشر في Anki بالإنجليزية)\n\n"
        "التركيز الأساسي دائمًا على الأجزاء المهمة وموضع السؤال لتوفير وقتك."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


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

        context.user_data["lecture_text"] = lecture_text
        context.user_data["lecture_title"] = os.path.splitext(file_name)[0][:40]
        context.user_data.pop("last_mcq_items", None)

        await status_msg.edit_text(
            f"تم تحليل المحاضرة: *{file_name}*\n"
            "عايز تعمل إيه دلوقتي؟ اختر من الأزرار التالية:",
            reply_markup=get_choices_keyboard(has_mcq_pdf=False),
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("Error handling document: %s", e)
        await status_msg.edit_text(f"حدث خطأ أثناء قراءة الملف: {type(e).__name__}")


async def send_long_text(chat, text: str) -> None:
    chunk_size = 3800
    for i in range(0, len(text), chunk_size):
        await chat.send_message(text[i : i + chunk_size])


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
        q = str(item.get("question", "")).strip()[:280]
        opts = [str(o).strip()[:90] for o in (item.get("options") or []) if str(o).strip()][:10]
        correct_id = item.get("correct_option_id", 0)
        exp = str(item.get("explanation", "")).strip()

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
    lecture_title = context.user_data.get("lecture_title", "lecture")

    if action == "export_mcq_pdf":
        mcq_items = context.user_data.get("last_mcq_items")
        if not mcq_items:
            await query.message.reply_text("لا توجد أسئلة MCQ محفوظة حاليًا. اضغط على MCQ Poll أولًا.")
            return

        pdf_wait = await query.message.reply_text("جاري تجهيز ملف الـ PDF للأسئلة والإجابات والشرح...")
        try:
            pdf_bytes = await asyncio.to_thread(generate_mcq_pdf_bytes, lecture_title, mcq_items)
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{lecture_title}_MCQ.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{lecture_title}_MCQ.pdf",
                caption=f"High-Yield MCQ PDF ({len(mcq_items)} questions with answers & explanations).",
                reply_markup=get_choices_keyboard(has_mcq_pdf=True),
            )
        except Exception as e:
            logger.error("Error generating MCQ PDF: %s", e)
            await pdf_wait.edit_text(f"حدث خطأ أثناء إنشاء ملف الـ PDF: {type(e).__name__}")
        return

    lecture_text = context.user_data.get("lecture_text")
    if not lecture_text:
        await query.message.reply_text("من فضلك أرسل ملف PDF للمحاضرة أولًا.")
        return

    mode = action.replace("mode_", "")
    wait_msg = await query.message.reply_text("جاري إعداد المحتوى الطبي المركز (High-Yield English)...")

    try:
        if mode == "mcq":
            prompt = build_mcq_json_prompt(lecture_text)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            raw_out = response.text or ""
            mcq_items = parse_mcq_json(raw_out)

            if not mcq_items:
                await wait_msg.edit_text("لم أتمكن من تنسيق أسئلة الـ Poll بشكل صحيح. حاول مرة أخرى.")
                return

            context.user_data["last_mcq_items"] = mcq_items
            await wait_msg.edit_text(f"تم توليد {len(mcq_items)} سؤال MCQ بالإنجليزية. جاري إرسال الاستفتاءات...")

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
                f"تم إرسال {len(mcq_items)} سؤال كـ Poll بالإنجليزية. الشرح التفصيلي لكل سؤال موجود داخل ملف الـ PDF.\n"
                "تحب تحمّل الأسئلة دي كملف PDF أو تختار حاجة تانية من نفس المحاضرة؟",
                reply_markup=get_choices_keyboard(has_mcq_pdf=True),
            )
            return

        prompt = build_mode_prompt(mode, lecture_text)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )

        output_text = (response.text or "").strip()
        if not output_text:
            await wait_msg.edit_text("لم يتمكن النموذج من توليد رد مناسب. حاول مرة أخرى.")
            return

        await wait_msg.delete()
        has_pdf = bool(context.user_data.get("last_mcq_items"))

        if mode == "anki":
            cleaned_lines = []
            for line in output_text.splitlines():
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("```"):
                    continue
                if "|" in line_stripped:
                    cleaned_lines.append(line_stripped)

            anki_content = "\n".join(cleaned_lines) if cleaned_lines else output_text
            anki_bytes = io.BytesIO(anki_content.encode("utf-8"))
            anki_bytes.name = f"{lecture_title}_anki.txt"

            await query.message.reply_document(
                document=anki_bytes,
                filename=f"{lecture_title}_anki.txt",
                caption="Anki deck ready for direct import (Separator: |). تحب تعمل حاجة تانية من نفس المحاضرة؟",
                reply_markup=get_choices_keyboard(has_mcq_pdf=has_pdf),
            )
        else:
            await send_long_text(query.message.chat, output_text)
            await query.message.reply_text(
                "تحب تولّد حاجة تانية من نفس المحاضرة؟",
                reply_markup=get_choices_keyboard(has_mcq_pdf=has_pdf),
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
