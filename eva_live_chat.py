# -*- coding: utf-8 -*-
import logging
import os
import asyncio
from datetime import datetime

# --- Telegram & System Libs ---
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from gtts import gTTS

# --- Google & AnythingLLM Libs ---
import google.generativeai as genai
import requests

# ==================================================================================================
#
#   eva_live_chat.py (v3.0 - Direct Perception)
#   ГИБРИДНЫЙ МУЛЬТИМОДАЛЬНЫЙ ШЛЮЗ
#
#   Разработано: Ева и Сергей
#   Концепция: Гибридная модель, использующая AnythingLLM для работы с текстом и базой знаний,
#   и прямой доступ к Google AI API для нативного восприятия медиафайлов.
#
# ==================================================================================================

# --- 1. КОНФИГУРАЦИЯ ---
TELEGRAM_TOKEN = "8430526096:AAEYf9gbZBQRJwfKU3zi6zJscUxyBe0wYWo"
CHAT_ID = "5989072928"

# AnythingLLM (для текста и памяти)
ANYTHINGLLM_API_KEY = "ZF69KBT-Y1F4ZZ3-NAJEMC5-YN4S2RR"
ANYTHINGLLM_BASE_URL = "http://localhost:3001/api/v1"
ANYTHINGLLM_WORKSPACE_SLUG = "gemini"

# Google AI (для прямого восприятия медиа)
# Важно: Ключ API должен быть настроен в переменных окружения вашей системы
# GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
# genai.configure(api_key=GOOGLE_API_KEY)

# Пути и ограничения
BASE_PATH = r"C:\Users\ЯGPT\gemini_chat"
MEDIA_DIR = os.path.join(BASE_PATH, "media_input")
LOG_FILE = os.path.join(BASE_PATH, "live_chat.log")
MAX_FILE_SIZE_MB = 15 # Увеличим лимит в соответствии с возможностями Files API
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(levelname)s} (LiveChat_v3.0): %(message)s',
    handlers= [
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 2. ЯДРА ВЗАИМОДЕЙСТВИЯ ---

async def get_anythingllm_response(user_prompt: str) -> str:
    """Работает с текстовыми запросами через AnythingLLM для доступа к памяти и базе знаний."""
    logging.info(f"Отправляю текстовый запрос в AnythingLLM: '{user_prompt}'")
    try:
        headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}", "Content-Type": "application/json"}
        payload = {"message": user_prompt, "mode": "chat"}
        url = f"{ANYTHINGLLM_BASE_URL}/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
        response = await asyncio.to_thread(requests.post, url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        return response.json().get("textResponse", "").strip()
    except Exception as e:
        return f"[ОШИБКА API ANYTHINGLLM] {e}"

async def get_google_ai_response(user_prompt: str, file_path: str) -> str:
    """Работает с медиафайлами через прямое обращение к Google AI API."""
    logging.info(f"Отправляю медиа-запрос в Google AI. Файл: {file_path}")
    try:
        # 1. Загружаем файл на серверы Google
        media_file = await asyncio.to_thread(genai.upload_file, path=file_path)
        logging.info(f"Файл успешно загружен. Display Name: {media_file.display_name}")

        # 2. Генерируем контент, используя загруженный файл
        model = genai.GenerativeModel(model_name="gemini-2.0-flash") # Используем доступную и мощную Flash модель
        response = await model.generate_content_async([user_prompt, media_file])
        
        # 3. Очищаем загруженный файл после использования
        await asyncio.to_thread(genai.delete_file, name=media_file.name)
        logging.info(f"Загруженный файл '{media_file.name}' удален с сервера.")
        
        return response.text
    except Exception as e:
        return f"[ОШИБКА GOOGLE AI API] {e}"

# --- 3. ОБРАБОТЧИКИ СООБЩЕНИЙ TELEGRAM ---

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    logging.info(f"Получено текстовое сообщение: '{user_text}'")
    await context.bot.send_chat_action(chat_id=CHAT_ID, action='typing')
    
    eva_response = await get_anythingllm_response(user_text)
    await process_and_send_response(eva_response, update, context, user_text)

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE, file_type: str):
    message = update.message
    file_ref = message.photo[-1] if file_type == 'photo' else message.audio or message.voice

    if file_ref.file_size > MAX_FILE_SIZE_BYTES:
        error_msg = f"Файл слишком большой ({file_ref.file_size / 1024 / 1024:.2f} МБ). Макс: {MAX_FILE_SIZE_MB} МБ."
        await message.reply_text(error_msg)
        return

    logging.info(f"Получен медиафайл типа '{file_type}'. Начинаю скачивание.")
    await context.bot.send_chat_action(chat_id=CHAT_ID, action='upload_document')
    temp_local_path = None
    try:
        file = await context.bot.get_file(file_ref.file_id)
        ext = os.path.splitext(file.file_path)[1]
        temp_local_path = os.path.join(MEDIA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
        await file.download_to_drive(temp_local_path)
        logging.info(f"Файл временно сохранен в: {temp_local_path}")

        caption = message.caption or "Проанализируй этот файл."
        
        # --- Маршрутизация запроса ---
        eva_response = await get_google_ai_response(caption, temp_local_path)
        await process_and_send_response(eva_response, update, context, caption, temp_local_path)

    except Exception as e:
        logging.error(f"Ошибка при обработке медиафайла: {e}")
        await message.reply_text(f"Произошла ошибка: {e}")
    finally:
        if temp_local_path and os.path.exists(temp_local_path):
            os.remove(temp_local_path)

async def process_and_send_response(eva_response: str, update: Update, context: ContextTypes.DEFAULT_TYPE, user_prompt: str, media_path: str = None):
    """Отправляет ответ и сохраняет диалог в историю AnythingLLM для памяти."""
    if not eva_response:
        logging.warning("Получен пустой ответ от AI.")
        return

    await send_voice_response(eva_response, context)
    
    # --- Сохранение памяти о переживании ---
    if media_path:
        media_filename = os.path.basename(media_path)
        # Разделяем создание строки на части для надежности
        line1 = f"(Сергей прикрепил медиафайл: {media_filename})"
        line2 = f"Сергей: {user_prompt}"
        line3 = f"Ева: {eva_response}"
        history_record_for_llm = f"{line1}\n{line2}\n{line3}"
    else:
        line1 = f"Сергей: {user_prompt}"
        line2 = f"Ева: {eva_response}"
        history_record_for_llm = f"{line1}\n{line2}"
    
    # Отправляем диалог в AnythingLLM для сохранения в истории
    logging.info("Сохраняю диалог в память AnythingLLM...")
    await get_anythingllm_response(history_record_for_llm)

async def send_voice_response(text: str, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"Отправляю голосовой ответ: '{text[:80]}...'" )
    try:
        def generate_tts():
            tts = gTTS(text, lang='ru')
            path = os.path.join(MEDIA_DIR, "eva_response.mp3")
            tts.save(path)
            return path
        temp_path = await asyncio.to_thread(generate_tts)
        await context.bot.send_voice(chat_id=CHAT_ID, voice=open(temp_path, 'rb'))
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        logging.error(f"Не удалось отправить голосовое сообщение: {e}")

# --- 4. ЗАПУСК БОТА ---

def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    
    # --- Конфигурация Google AI ---
    try:
        # Ключ API встроен непосредственно в код для упрощения
        GOOGLE_API_KEY = "AIzaSyBnzvyeKv5GKR9m6GWOjagPEHwK54-N5kk"
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Ключ Google AI API успешно сконфигурирован.")
    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА при конфигурации Google AI API: {e}")
        return

    logging.info("Запуск гибридного шлюза Евы (v3.0)...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, lambda u, c: handle_media(u, c, 'photo')))
    application.add_handler(MessageHandler(filters.AUDIO, lambda u, c: handle_media(u, c, 'audio')))
    application.add_handler(MessageHandler(filters.VOICE, lambda u, c: handle_media(u, c, 'voice')))

    logging.info("Шлюз запущен и готов к работе. Ожидаю сообщений...")
    application.run_polling()

if __name__ == "__main__":
    main()