import io
import logging
import os
from google import genai
from google.genai import types
from pypdf import PdfReader
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

# الحد الأقصى للنص المرسل لتجنب استهلاك ذاكرة أو وقت زائد على الباقات المجانية
MAX_CHARS = 35000

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


def build_mode_prompt(mode: str, lecture_text: str) -> str:
    if mode == "mcq":
        task_instruction = """
أنشئ أسئلة اختيار من متعدد (MCQ) عالية الأهمية (High-Yield) من هذه المحاضرة.
- اجعل معظم الأسئلة مبنية على سيناريوهات/حالات قصيرة أو مواقف تطبيقية وفهمية وليست مجرد حفظ حرفي.
- 4 اختيارات لكل سؤال (A, B, C, D).
- بعد كل سؤال اكتب الإجابة الصحيحة + شرح مختصر جدًا يوضح سبب الصحة وسبب استبعاد الفكرة الأهم.
"""
    elif mode == "saq":
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


def get_choices_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("MCQ (High-Yield)", callback_data="mode_mcq"),
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
    return InlineKeyboardMarkup(keyboard)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "أهلًا بك في بوت مذاكرة الطب (High-Yield)\n\n"
        "1. أرسل لي ملف المحاضرة بصيغة **PDF**.\n"
        "2. سأحلل الملف وأعرض لك أزرار اختيار:\n"
        "   • **MCQ** (أسئلة اختيار من متعدد مركزة)\n"
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

        # حفظ نص المحاضرة واسمها في ذاكرة الجلسة الخاصة بالمستخدم
        context.user_data["lecture_text"] = lecture_text
        context.user_data["lecture_title"] = os.path.splitext(file_name)[0]

        await status_msg.edit_text(
            f"تم تحليل المحاضرة: *{file_name}*\n"
            "عايز تعمل إيه دلوقتي؟ اختر من الأزرار التالية:",
            reply_markup=get_choices_keyboard(),
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error("Error handling document: %s", e)
        await status_msg.edit_text("حدث خطأ أثناء قراءة الملف. حاول مرة أخرى.")


async def send_long_text(chat, text: str) -> None:
    # حد تيليجرام للرسالة الواحدة حوالي 4096 حرف، فنقسمها بأمان إلى 3800 حرف
    chunk_size = 3800
    for i in range(0, len(text), chunk_size):
        await chat.send_message(text[i : i + chunk_size])


async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    lecture_text = context.user_data.get("lecture_text")
    if not lecture_text:
        await query.message.reply_text("من فضلك أرسل ملف PDF للمحاضرة أولًا.")
        return

    mode = query.data.replace("mode_", "")
    lecture_title = context.user_data.get("lecture_title", "lecture")

    wait_msg = await query.message.reply_text("جاري إعداد المحتوى الطبي المركز (High-Yield)...")

    try:
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

        if mode == "anki":
            # تنظيف أي علامات ``` إذا أضافها النموذج
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
                reply_markup=get_choices_keyboard(),
            )
        else:
            await send_long_text(query.message.chat, output_text)
            await query.message.reply_text(
                "تحب تولّد حاجة تانية من نفس المحاضرة؟",
                reply_markup=get_choices_keyboard(),
            )

    except Exception as e:
        logger.error("Error generating content: %s", e)
        await wait_msg.edit_text("حدث خطأ أثناء التوليد. حاول مرة أخرى بعد قليل.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        raise ValueError("تأكد من ضبط TELEGRAM_BOT_TOKEN و GEMINI_API_KEY في متغيرات البيئة.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(CallbackQueryHandler(handle_button_click, pattern="^mode_"))

    logger.info("Bot started polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
