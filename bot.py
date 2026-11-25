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

MAX_CHARS = 35000
ARABIC_FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
)
ARABIC_FONT_LOCAL_PATH = "/tmp/Amiri-Regular.ttf"

MEDICAL_HIGH_YIELD_SYSTEM_PROMPT = """
أنت مساعد أكاديمي متخصص في مساعدة طلاب كلية الطب على المذاكرة بكفاءة عالية.
القاعدة الذهبية: ركّز بنسبة كبيرة جدًا على الأجزاء المهمة و"موضع السؤال" (High-Yield) في الامتحانات الطبية:
- المفاهيم الأساسية والميكانيزمات (Mechanisms / Pathophysiology).
- الحالات الكلينيكية المميزة (Classic clinical vignettes)، الأعراض والعلامات الفارقة.
- الفروق التشخيصية والمقارنات الحاسمة (Key comparisons / Differential points).
- الاستثناءات المهمة، المضاعفات الشائعة، أو النقاط المحورية في المحاضرة.
تجنب تمامًا الأسئلة أو النقاط التافهة أو التفاصيل الهامشية غير الجوهرية.
اكتب بأسلوب طبي واضح، منظم، ومباشر لتوفير وقت الطالب.
"""


def get_arabic_font_path() -> str | None:
    if os.path.exists(ARABIC_FONT_LOCAL_PATH) and os.path.getsize(ARABIC_FONT_LOCAL_PATH) > 10000:
        return ARABIC_FONT_LOCAL_PATH
    try:
        resp = requests.get(ARABIC_FONT_URL, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 10000:
            with open(ARABIC_FONT_LOCAL_PATH, "wb") as f:
                f.write(resp.content)
            return ARABIC_FONT_LOCAL_PATH
    except Exception as e:
        logger.warning("Could not download Arabic font: %s", e)
    return None


def generate_mcq_pdf_bytes(lecture_title: str, mcq_items: list[dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = get_arabic_font_path()
    if font_path:
        pdf.add_font("ArabicFont", "", font_path)
        pdf.set_font("ArabicFont", size=15)
    else:
        pdf.set_font("Helvetica", size=14)

    pdf.cell(0, 10, text=f"High-Yield MCQ - {lecture_title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("ArabicFont" if font_path else "Helvetica", size=11)

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
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            extracted_parts.append(f"--- Page {idx} ---\n{text}")
    full_text = "\n\n".join(extracted_parts)
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS] + "\n\n[تم الاقتطاع للحفاظ على الأهم والتركيز]"
    return full_text


def build_mcq_json_prompt(lecture_text: str) -> str:
    return f"""
أنشئ 6 إلى 10 أسئلة اختيار من متعدد (MCQ) عالية الأهمية (High-Yield) من هذه المحاضرة الطبية.
- اجعل معظم الأسئلة مبنية على سيناريوهات/حالات قصيرة أو مواقف تطبيقية وفهمية وليست مجرد حفظ حرفي.
- كل سؤال يحتوي على 4 اختيارات فقط (قائمة من 4 نصوص، كل اختيار لا يتجاوز 90 حرفًا ليتناسب مع استفتاءات تيليجرام).
- نص السؤال لا يتجاوز 280 حرفًا.
- حدد رقم الإجابة الصحيحة (من 0 إلى 3) في `correct_option_id`.
- أضف شرحًا مختصرًا ودقيقًا في `explanation`.

أرجع النتيجة بصيغة JSON صالحة فقط بدون أي نص إضافي أو علامات ```json بهذا الشكل:
{{
  "questions": [
    {{
      "question": "نص السؤال أو الحالة",
      "options": ["اختيار 1", "اختيار 2", "اختيار 3", "اختيار 4"],
      "correct_option_id": 0,
      "explanation": "شرح مختصر لسبب الصحة واستبعاد البدائل المهمة"
    }}
  ]
}}

محتوى المحاضرة:
\"\"\"
{lecture_text}
\"\"\"
"""


def build_mode_prompt(mode: str, lecture_text: str) -> str:
    if mode == "saq":
        task_instruction = """
أنشئ أسئلة مقالية قصيرة (SAQ) مركزة على أهم نقاط المحاضرة ومواضع الامتحان.
- اجعل الأسئلة واضحة ومحددة (آلية حدوث، مقارنة، تفسير حالة، أو نقاط رئيسية).
- بعد كل سؤال اكتب نموذج إجابة مختصر ومنظم في نقاط (Bullet points) يسهل استرجاعه في الامتحان.
"""
    elif mode == "mix":
        task_instruction = """
أنشئ مزيجًا مركّزًا (Mix) من أسئلة MCQ (حالات/مواضع سؤال) وأسئلة SAQ قصيرة من أهم أجزاء المحاضرة.
- ركّز فقط على النقاط التي تتكرر في الامتحانات وتفرّق بين الطلاب.
- ضع الإجابات والتفسيرات المختصرة بوضوح.
"""
    elif mode == "summary":
        task_instruction = """
لخّص المحاضرة بأسلوب High-Yield طبي مركز ومناسب للمراجعة السريعة قبل الامتحان.
- قسّم الملخص إلى عناوين واضحة.
- أبرز الفروق المهمة، الاستثناءات، والربط بين المعلومة وأهميتها الإكلينيكية أو الامتحانية.
- احذف الحشو والكلام المكرر.
"""
    elif mode == "anki":
        task_instruction = """
أنشئ بطاقات Anki جاهزة للاستيراد المباشر بصيغة نصية مفصولة بعلامة الأنابيب ( | ) فقط، سطر واحد لكل بطاقة بدون ترقيم أو مقدمات أو شرح خارج البطاقات.
الصيغة المطلوبة لكل سطر:
السؤال أو الحالة أو النقطة المهمة | الإجابة المختصرة الدقيقة + نقطة تمييز سريعة
مثال:
ما الميكانيزم الأساسي لـ ... وكيف يظهر سريريًا؟ | يحدث بسبب ... ويتميز بـ ... (موضع سؤال شائع)
ركّز فقط على أهم نقاط المحاضرة ومواضع الأسئلة.
"""
    else:
        task_instruction = "لخّص أهم نقاط المحاضرة بأسلوب طبي مركز."

    return f"""{task_instruction}

محتوى المحاضرة:
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
        "أهلًا بك في بوت مذاكرة الطب (High-Yield)\n\n"
        "1. أرسل لي ملف المحاضرة بصيغة **PDF**.\n"
        "2. سأحلل الملف وأعرض لك أزرار اختيار:\n"
        "   • **MCQ Poll** (أسئلة تفاعلية كـ Poll + شرح + إمكانية تحميلها PDF)\n"
        "   • **SAQ** (أسئلة قصيرة بإجابات نموذجية)\n"
        "   • **Mix** (مزيج MCQ + SAQ)\n"
        "   • **Summary** (ملخص High-Yield سريع)\n"
        "   • **Anki Ready (.txt)** (ملف جاهز للاستيراد المباشر في Anki)\n\n"
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
        lecture_text = extract_text_from_pdf(pdf_bytes)

        if not lecture_text.strip():
            await status_msg.edit_text(
                "لم أتمكن من استخراج نص واضح من هذا الـ PDF (قد يكون صورًا فقط). جرّب ملف PDF يحتوي على نص."
            )
            return

        context.user_data["lecture_text"] = lecture_text
        context.user_data["lecture_title"] = os.path.splitext(file_name)[0]
        context.user_data.pop("last_mcq_items", None)

        await status_msg.edit_text(
            f"تم تحليل المحاضرة: *{file_name}*\n"
            "عايز تعمل إيه دلوقتي؟ اختر من الأزرار التالية:",
            reply_markup=get_choices_keyboard(has_mcq_pdf=False),
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("Error handling document: %s", e)
        await status_msg.edit_text("حدث خطأ أثناء قراءة الملف. حاول مرة أخرى.")


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
        q = str(item.get("question", "")).strip()[:295]
        opts = [str(o).strip()[:95] for o in (item.get("options") or []) if str(o).strip()][:10]
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

    # تصدير آخر أسئلة MCQ تم توليدها إلى ملف PDF
    if action == "export_mcq_pdf":
        mcq_items = context.user_data.get("last_mcq_items")
        if not mcq_items:
            await query.message.reply_text("لا توجد أسئلة MCQ محفوظة حاليًا. اضغط على MCQ Poll أولًا.")
            return

        pdf_wait = await query.message.reply_text("جاري تجهيز ملف الـ PDF للأسئلة...")
        try:
            pdf_bytes = await asyncio.to_thread(generate_mcq_pdf_bytes, lecture_title, mcq_items)
            pdf_io = io.BytesIO(pdf_bytes)
            pdf_io.name = f"{lecture_title}_MCQ.pdf"
            await pdf_wait.delete()
            await query.message.reply_document(
                document=pdf_io,
                filename=f"{lecture_title}_MCQ.pdf",
                caption="ملف PDF بأسئلة الـ MCQ والإجابات والشرح.",
                reply_markup=get_choices_keyboard(has_mcq_pdf=True),
            )
        except Exception as e:
            logger.error("Error generating MCQ PDF: %s", e)
            await pdf_wait.edit_text("حدث خطأ أثناء إنشاء ملف الـ PDF.")
        return

    lecture_text = context.user_data.get("lecture_text")
    if not lecture_text:
        await query.message.reply_text("من فضلك أرسل ملف PDF للمحاضرة أولًا.")
        return

    mode = action.replace("mode_", "")
    wait_msg = await query.message.reply_text("جاري إعداد المحتوى الطبي المركز (High-Yield)...")

    try:
        if mode == "mcq":
            prompt = build_mcq_json_prompt(lecture_text)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MEDICAL_HIGH_YIELD_SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )
            raw_out = response.text or ""
            mcq_items = parse_mcq_json(raw_out)

            if not mcq_items:
                await wait_msg.edit_text("لم أتمكن من تنسيق أسئلة الـ Poll بشكل صحيح. حاول مرة أخرى.")
                return

            context.user_data["last_mcq_items"] = mcq_items
            await wait_msg.delete()

            for idx, item in enumerate(mcq_items, start=1):
                await query.message.reply_poll(
                    question=f"{idx}. {item['question']}"[:295],
                    options=item["options"],
                    type="quiz",
                    correct_option_id=item["correct_option_id"],
                    is_anonymous=False,
                )
                if item["explanation"]:
                    await query.message.reply_text(f"💡 شرح سؤال {idx}: {item['explanation']}")

            await query.message.reply_text(
                "تحب تحمّل الأسئلة دي كملف PDF أو تختار حاجة تانية من نفس المحاضرة؟",
                reply_markup=get_choices_keyboard(has_mcq_pdf=True),
            )
            return

        prompt = build_mode_prompt(mode, lecture_text)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
                caption="ملف Anki جاهز للاستيراد المباشر (Separator: |). تحب تعمل حاجة تانية من نفس المحاضرة؟",
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
        await wait_msg.edit_text("حدث خطأ أثناء التوليد. حاول مرة أخرى بعد قليل.")


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
