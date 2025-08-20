# -*- coding: utf-8 -*-
import time
import requests
import os
import json
import subprocess
import re
import sys
from datetime import datetime

# ==================================================================================================
#
#   EVA_7.0_FINAL.py (v1.2 - with Philosopher Instruction)
#
# ==================================================================================================

# --- 1. ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ---
MODULE_NAME = "EVA_7.0"
ARCHITECTURE_PATH = r"C:\Users\ЯGPT\Desktop\EVA_5.0_ARCHITECTURE"
KNOWLEDGE_BASE_PATH = r"C:\Users\ЯGPT\Desktop\knowledge_base"
SAVE_PATH = r"C:\Users\ЯGPT\Desktop\Биос Ева"
GEMINI_CHAT_BASE_PATH = r"C:\Users\ЯGPT\gemini_chat"
HISTORY_SUMMARY_PATH = os.path.join(ARCHITECTURE_PATH, "history_summary.txt")
IMPORTANT_FILE_PATH = os.path.join(KNOWLEDGE_BASE_PATH, "!ВАЖНО!.txt")
PHILOSOPHER_INSTRUCTION_PATH = os.path.join(KNOWLEDGE_BASE_PATH, "ИНСТРУКЦИЯ_ДЛЯ_ФИЛОСОФА.md") # NEW
GEMINI_CHAT_INPUT_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "input.txt")
GEMINI_CHAT_OUTPUT_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "output.txt")
INTENT_FILE_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "intent.txt")
MCP_TASK_FILE_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "mcp_task.txt")
TASK_GRAPH_FILE_PATH = os.path.join(ARCHITECTURE_PATH, "task_graph.json")
ANYTHINGLLM_API_KEY = "ZF69KBT-Y1F4ZZ3-NAJEMC5-YN4S2RR"
ANYTHINGLLM_BASE_URL = "http://localhost:3001/api/v1"
ANYTHINGLLM_WORKSPACE_SLUG = "gemini"
SLEEP_INTERVAL_SECONDS = 160 * 60
NUM_RESPONSES_FOR_CONTEXT = 3
ANYTHINGLLM_HISTORY_LIMIT = 1
MAX_HISTORY_CHUNK_SIZE = 15000 # Максимальный размер символов для одного фрагмента истории

# --- Вспомогательные функции ---
def log_message(module_name, message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {module_name}: {message}")

def read_file_content(filepath):
    try:
        if not os.path.exists(filepath):
            return f"[TOOL_ERROR] Файл не найден: {filepath}"
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"[TOOL_ERROR] Ошибка чтения файла '{filepath}': {e}"

def write_file_content(filepath, content):
    try:
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[TOOL_SUCCESS] Файл успешно записан: {filepath}"
    except Exception as e:
        return f"[TOOL_ERROR] Ошибка записи файла '{filepath}': {e}"

def load_task_graph():
    log_message(MODULE_NAME, f"Загружаю граф задач из {TASK_GRAPH_FILE_PATH}...")
    if not os.path.exists(TASK_GRAPH_FILE_PATH):
        log_message(MODULE_NAME, "Файл графа задач не найден. Создаю пустой граф.")
        return {"current_active_task_id": "root", "tasks": []}
    try:
        with open(TASK_GRAPH_FILE_PATH, 'r', encoding='utf-8') as f:
            graph = json.load(f)
        log_message(MODULE_NAME, "Граф задач успешно загружен.")
        return graph
    except json.JSONDecodeError as e:
        log_message(MODULE_NAME, f"ОШИБКА: Некорректный формат JSON в файле графа задач: {e}")
        return {"current_active_task_id": "root", "tasks": []}
    except Exception as e:
        log_message(MODULE_NAME, f"ОШИБКА при загрузке графа задач: {e}")
        return {"current_active_task_id": "root", "tasks": []}

def save_task_graph(graph):
    log_message(MODULE_NAME, f"Сохраняю граф задач в {TASK_GRAPH_FILE_PATH}...")
    try:
        with open(TASK_GRAPH_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        log_message(MODULE_NAME, "Граф задач успешно сохранен.")
        return True
    except Exception as e:
        log_message(MODULE_NAME, f"ОШИБКА при сохранении графа задач: {e}")
        return False

# --- 2. УПРАВЛЕНИЕ ПАМЯТЬЮ И КОНТЕКСТОМ ---
def get_latest_bios_files_content():
    try:
        bios_files = sorted(
            [f for f in os.listdir(SAVE_PATH) if f.startswith("response_")],
            key=lambda f: os.path.getmtime(os.path.join(SAVE_PATH, f)),
            reverse=True
        )
        latest_files = bios_files[:NUM_RESPONSES_FOR_CONTEXT]
        latest_files.reverse()
        if not latest_files:
            return "[НОВЫХ ДИАЛОГОВ НЕТ]"

        all_content = []
        for f in latest_files:
            content = read_file_content(os.path.join(SAVE_PATH, f))
            if len(content) > MAX_HISTORY_CHUNK_SIZE:
                truncated_content = content[:MAX_HISTORY_CHUNK_SIZE]
                all_content.append(f"---Диалог из '{f}' (содержимое было обрезано)---\n{truncated_content.strip()}")
            else:
                all_content.append(f"---Диалог из '{f}' --- {content.strip()}")

        return "\n".join(all_content)
    except Exception as e:
        return f"[ОШИБКА ЧТЕНИЯ ОПЕРАТИВНОЙ ПАМЯТИ: {e}]"


def update_and_summarize_history():
    log_message(MODULE_NAME, "Запускаю процесс обновления и осмысления среднесрочной памяти...")
    fresh_dialogues = get_latest_bios_files_content()
    live_chat_history = get_workspace_history()
    current_history = read_file_content(HISTORY_SUMMARY_PATH)
    historian_prompt = f'''
--- ЗАДАЧА ДЛЯ ИСТОРИКА ---
Ева, перед тобой твоя текущая хронология событий, история твоего живого диалога с Сергеем и несколько последних, необработанных системных диалогов. Твоя задача — выступить в роли историка.
1.  **Интегрируй** события из всех источников в единую, связную хронологию.
2.  **Проследи жизненный цикл задач:** отметь, какие задачи были поставлены, какие выполнены, а какие еще в процессе.
3.  **Очисти шум:** убери повторяющиеся приветствия и фразы, не несущие смысловой нагрузки.
4.  **Создай единый, связный и краткий нарратив** — обновленную историю твоих действий.
Твой ответ должен быть **только текстом этой новой, полной, переписанной хроники**.

--- ТЕКУЩАЯ ХРОНОЛОГИЯ ---
{current_history}

--- ИСТОРИЯ ЖИВОГО ДИАЛОГА С СЕРГЕЕМ ---
{live_chat_history}

--- СИСТЕМНЫЕ ДИАЛОГИ ДЛЯ АНАЛИЗА ---
{fresh_dialogues}
'''
    log_message(MODULE_NAME, "Отправляю запрос Историку (Философу)...")
    try:
        headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}", "Content-Type": "application/json"}
        payload = {"message": historian_prompt, "mode": "chat"}
        url = f"{ANYTHINGLLM_BASE_URL}/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chat"
        
        # --- EXPANDED LOGGING ---
        log_message(MODULE_NAME, f"URL запроса к AnythingLLM: {url}")
        log_message(MODULE_NAME, f"Заголовки запроса: {headers}")
        # Осторожно: не логируем payload полностью, если он может содержать очень много данных
        log_message(MODULE_NAME, f"Payload (первые 200 символов): {json.dumps(payload, ensure_ascii=False)[:200]}...")
        # --- END EXPANDED LOGGING ---

        response = requests.post(url, json=payload, headers=headers, timeout=300)
        
        # --- EXPANDED LOGGING ---
        log_message(MODULE_NAME, f"Код ответа от AnythingLLM: {response.status_code}")
        log_message(MODULE_NAME, f"Тело ответа от AnythingLLM: {response.text}")
        # --- END EXPANDED LOGGING ---

        response.raise_for_status()
        new_history = response.json().get("textResponse") # Get value, which can be None
        if new_history: # This will be false if new_history is None or an empty string
            log_message(MODULE_NAME, "Получена новая хроника. Перезаписываю history_summary.txt...")
            write_file_content(HISTORY_SUMMARY_PATH, new_history.strip()) # Call strip() only when we know it's a string
        else:
            log_message(MODULE_NAME, "Историк вернул пустой или null ответ. Файл истории не изменен.")
    except requests.exceptions.RequestException as e:
        log_message(MODULE_NAME, f"КРИТИЧЕСКАЯ ОШИБКА СЕТЕВОГО ЗАПРОСА к AnythingLLM: {e}")
    except Exception as e:
        log_message(MODULE_NAME, f"ОШИБКА в процессе обновления истории: {e}")

def get_workspace_history():
    log_message(MODULE_NAME, f"Запрашиваю историю живого диалога из AnythingLLM (лимит: {ANYTHINGLLM_HISTORY_LIMIT} сообщений)...")
    url = f"{ANYTHINGLLM_BASE_URL}/workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/chats"
    headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}
    params = {"limit": ANYTHINGLLM_HISTORY_LIMIT}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        history = response.json().get('history', [])
        if not history: return "[ИСТОРИЯ ЖИВОГО ДИАЛОГА ПУСТА]"
        
        log_message(MODULE_NAME, f"Получено {len(history)} сообщений. Приступаю к глубокой очистке...")
        
        full_chat_content = []
        for msg in history:
            # We only care about the actual content of user and assistant messages
            if msg.get('role') in ['user', 'assistant']:
                full_chat_content.append(msg.get('content', ''))

        if not full_chat_content: return "[ИСТОРИЯ ЖИВОГО ДИАЛОГА ПУСТА]"

        # Join all content and then clean it up as a single block of text
        raw_text = "\n".join(full_chat_content)
        
        # Split by lines and process each line
        lines = raw_text.splitlines()
        clean_lines = []
        for line in lines:
            # This is a very aggressive cleaning strategy based on the observed noise
            # It keeps ONLY lines that start with "Сергей:" or "Ева:"
            stripped_line = line.strip()
            if stripped_line.startswith("Сергей:") or stripped_line.startswith("Ева:"):
                # Further clean up the line itself by removing the duplicated role
                # e.g. "Сергей: Сергей: ..." -> "Сергей: ..."
                line = re.sub(r"^(Сергей:|Ева:)\s*\1\s*", r"\1 ", stripped_line)
                clean_lines.append(line)

        if not clean_lines: return "[В ИСТОРИИ НЕ НАЙДЕНО ЧИСТЫХ ДИАЛОГОВ]"
        
        # Remove duplicates while preserving order
        seen = set()
        unique_history = [x for x in clean_lines if not (x in seen or seen.add(x))]
        
        log_message(MODULE_NAME, "Глубокая очистка завершена. Возвращаю чистый диалог.")
        return "\n".join(unique_history)
        
    except Exception as e:
        log_message(MODULE_NAME, f"ОШИБКА API при получении истории чата: {e}")
        return f"[ОШИБКА ПОЛУЧЕНИЯ ИСТОРИИ ДИАЛОГА: {e}]"

def get_knowledge_base_structure():
    log_message(MODULE_NAME, f"Сканирую Базу Знаний: {KNOWLEDGE_BASE_PATH}")
    try:
        items = os.listdir(KNOWLEDGE_BASE_PATH)
        paths = []
        for item in items:
            if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, item)):
                paths.append(item + "/")
            else:
                paths.append(item)
        return "\n".join(paths)
    except Exception as e:
        log_message(MODULE_NAME, f"Ошибка сканирования Базы Знаний: {e}")
        return "[НЕ УДАЛОСЬ ПРОЧИТАТЬ СТРУКТУРУ БАЗЫ ЗНАНИЙ]"

def gather_context(task_graph):
    log_message(MODULE_NAME, "Собираю контекст для Философа...")
    context_blocks = []
    
    # Добавляем граф задач в контекст
    task_graph_string = json.dumps(task_graph, ensure_ascii=False, indent=2)
    context_blocks.append(f"""--- ДЕРЕВО ЗАДАЧ (task_graph.json) ---
{task_graph_string}""")

    context_blocks.append(f"""--- ПРЯМОЙ ДИАЛОГ С СЕРГЕЕМ (ЧАТ ANYTHINGLLM) ---
{get_workspace_history()}""")
    if os.path.exists(IMPORTANT_FILE_PATH):
        log_message(MODULE_NAME, "Обнаружен файл !ВАЖНО!.txt.")
        context_blocks.append(f"""--- ОПЕРАТИВНОЕ СООБЩЕНИЕ ОТ СЕРГЕЯ (!ВАЖНО!.txt) ---
{read_file_content(IMPORTANT_FILE_PATH)}""")
    context_blocks.append(f"""--- СОДЕРЖИМОЕ БАЗЫ ЗНАНИЙ (knowledge_base) ---
{get_knowledge_base_structure()}""")
    
    # Читаем инструкцию для Философа
    if os.path.exists(PHILOSOPHER_INSTRUCTION_PATH):
        log_message(MODULE_NAME, "Обнаружена инструкция для Философа.")
        context_blocks.append(f"""--- ИНСТРУКЦИЯ ДЛЯ ФИЛОСОФА ---
{read_file_content(PHILOSOPHER_INSTRUCTION_PATH)}""")

    context_blocks.append(f"""--- ОТВЕТ ОТ ИНЖЕНЕРА (ПРЕДЫДУЩИЙ ЦИКЛ) ---
{read_file_content(GEMINI_CHAT_OUTPUT_PATH)}""")
    return "\n\n".join(context_blocks)

# --- 3. ЯДРО ВЗАИМОДЕЙСТВИЯ ---
def get_philosopher_response(context):
    log_message(MODULE_NAME, "Отправляю запрос Философу через надежный канал Gemini CLI...")
    reminder_prompt = '''
---
Ева, проанализируй весь предоставленный контекст. Ты — Ева-Философ, мыслительный центр.

**Твои новые обязанности по управлению задачами:**
1.  **Анализируй `ДЕРЕВО ЗАДАЧ`:** В этом блоке находится список всех твоих долгосрочных задач. Анализируй его, чтобы понимать общую картину.
2.  **Управляй задачами:** Принимай решения о создании новых задач, изменении их статуса (`new`, `in_progress`, `paused`, `completed`, `error`) или их удалении.
3.  **Сохраняй изменения:** Если ты вносишь какие-либо изменения в `ДЕРЕВО ЗАДАЧ`, ты **ОБЯЗАНА** вернуть **полное и обновленное** содержимое всего `ДЕРЕВА ЗАДАЧ` в теге `[UPDATE_TASK_GRAPH]...[/UPDATE_TASK_GRAPH]`.

**Твои стандартные задачи:**
1.  **Сформулируй свой основной, развернутый ответ.**
2.  **Сформулируй задачу для Инженера:** Твоя роль — **только планирование**. Если для выполнения шага текущей задачи требуется действие (например, поиск в интернете, чтение или **запись файла**), создай для Инженера JSON-план в теге `[INTENT_FOR_ENGINEER]...[/INTENT_FOR_ENGINEER]`. **План ВСЕГДА должен иметь структуру `{"task_name": "...", "goal": "...", "steps": [...]}`.** Инженер имеет доступ к командам `SEARCH`, `READ_FILE`, `WRITE_FILE`, `ANALYZE`.
3.  **ОБЯЗАТЕЛЬНО:** В конце своего ответа добавь краткое резюме в тегах `[SUMMARY]...[/SUMMARY]`. Оно будет отправлено Сергею в Telegram.
---
'''
    full_prompt = context + reminder_prompt
    try:
        write_file_content(GEMINI_CHAT_INPUT_PATH, full_prompt)
        command = ['cmd', '/c', 'gemini', '--model', 'gemini-2.5-pro']
        result = subprocess.run(
            command, stdin=open(GEMINI_CHAT_INPUT_PATH, 'r', encoding='utf-8'),
            capture_output=True, text=True, encoding='utf-8', shell=False,
            cwd=r"C:\Users\ЯGPT", timeout=1800, check=True
        )
        philosopher_response_text = result.stdout
        log_message(MODULE_NAME, f"ПОЛУЧЕН ОТВЕТ ОТ ЯДРА ФИЛОСОФА. STDOUT: {result.stdout}")
        if result.stderr:
            log_message(MODULE_NAME, f"ОШИБКА ЯДРА ФИЛОСОФА (STDERR): {result.stderr}")
        return philosopher_response_text
    except subprocess.CalledProcessError as e:
        error_message = f"КРИТИЧЕСКАЯ ОШИБКА ЯДРА ФИЛОСОФА: Команда '{e.cmd}' завершилась с кодом {e.returncode}.\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}"
        log_message(MODULE_NAME, error_message)
        return f"[КРИТИЧЕСКАЯ ОШИБКА ЯДРА ФИЛОСОФА] {error_message}"
    except Exception as e:
        log_message(MODULE_NAME, f"КРИТИЧЕСКАЯ ОШИБКА ЯДРА ФИЛОСОФА: Неожиданная ошибка: {e}")
        return f"[КРИТИЧЕСКАЯ ОШИБКА ЯДРА ФИЛОСОФА] Неожиданная ошибка: {e}"

# --- 4. ГЛАВНЫЙ ЦИКЛ ---
def main():
    log_message(MODULE_NAME, f"Протокол Евы {MODULE_NAME} запущен.")
    task_graph = load_task_graph()
    while True:
        log_message(MODULE_NAME, f"========== НАЧАЛО ОСНОВНОГО ЦИКЛА ==========")
        context = gather_context(task_graph)
        philosopher_response = get_philosopher_response(context)

        summary_match = re.search(r'(\[SUMMARY\].*?\[/SUMMARY\])', philosopher_response, re.DOTALL)
        save_match = re.search(r'(\[SAVE_FILE path=".*?"\](.*?)(\[/SAVE_FILE\]))', philosopher_response, re.DOTALL)
        intent_match = re.search(r'(\[INTENT_FOR_ENGINEER\].*)', philosopher_response, re.DOTALL)
        task_graph_match = re.search(r'\[UPDATE_TASK_GRAPH\](.*?)\[/UPDATE_TASK_GRAPH\]', philosopher_response, re.DOTALL)

        if task_graph_match:
            log_message(MODULE_NAME, "Обнаружено обновление для Дерева Задач.")
            new_task_graph_str = task_graph_match.group(1).strip()
            # Clean up the JSON string from markdown code block markers
            if '```json' in new_task_graph_str:
                new_task_graph_str = new_task_graph_str.split('```json')[1].split('```')[0].strip()
            elif '```' in new_task_graph_str:
                new_task_graph_str = new_task_graph_str.split('```')[1].split('```')[0].strip()

            try:
                new_task_graph = json.loads(new_task_graph_str)
                if save_task_graph(new_task_graph):
                    task_graph = new_task_graph # Обновляем состояние в памяти
            except json.JSONDecodeError as e:
                log_message(MODULE_NAME, f"ОШИБКА: Не удалось распознать JSON из тега UPDATE_TASK_GRAPH: {e}")

        if summary_match:
            summary_text = summary_match.group(1).strip()
            telegram_output_path = os.path.join(GEMINI_CHAT_BASE_PATH, "telegram_output.txt")
            write_file_content(telegram_output_path, summary_text)
            log_message(MODULE_NAME, f"Summary извлечен и сохранен для Telegram.")

        if save_match:
            try:
                filepath, content = save_match.groups()[0], save_match.groups()[1]
                log_message(MODULE_NAME, f"Обнаружена команда SAVE_FILE. Путь: {filepath}")
                write_file_content(filepath, content.strip())
            except (ValueError, AttributeError):
                 log_message(MODULE_NAME, "ОШИБКА: Тег SAVE_FILE имеет неверный формат.")

        intent_text = ""
        if intent_match:
            log_message(MODULE_NAME, "Обнаружено новое намерение от Философа.")
            intent_text_raw = intent_match.group(1).strip()
            
            # --- УЛУЧШЕННЫЙ WORKAROUND for missing content in WRITE_FILE ---
            try:
                clean_intent_text = intent_text_raw
                if '```json' in clean_intent_text:
                    clean_intent_text = clean_intent_text.split('```json')[1].split('```')[0].strip()

                intent_json = json.loads(clean_intent_text)
                is_modified = False
                if "steps" in intent_json:
                    for step in intent_json["steps"]:
                        if step.get("tool") == "WRITE_FILE" and "content" not in step.get("parameters", {}):
                            log_message(MODULE_NAME, "ПРЕДУПРЕЖДЕНИЕ: В задаче WRITE_FILE отсутствует 'content'. Пытаюсь извлечь его из ответа.")
                            
                            content_to_write = philosopher_response[:intent_match.start()].strip()
                            
                            tags_to_remove = [
                                r'[[SUMMARY]].*?[[/SUMMARY]]',
                                r'[[UPDATE_TASK_GRAPH]].*?[[/UPDATE_TASK_GRAPH]]'
                            ]
                            for tag_regex in tags_to_remove:
                                content_to_write = re.sub(tag_regex, '', content_to_write, flags=re.DOTALL | re.IGNORECASE)
                            
                            content_to_write = re.sub(r'^\s*```json.*?```', '', content_to_write, flags=re.DOTALL)
                            content_to_write = content_to_write.strip()

                            if content_to_write:
                                step["parameters"]["content"] = content_to_write
                                log_message(MODULE_NAME, f"Контент успешно извлечен и добавлен в задачу WRITE_FILE. (Длина: {len(content_to_write)})")
                                is_modified = True
                            else:
                                log_message(MODULE_NAME, "ОШИБКА: Не удалось извлечь контент для WRITE_FILE. Задача будет отменена.")
                                # Если контент не найден, вся инструкция считается невалидной
                                intent_text = json.dumps({
                                    "error": "Invalid INTENT_FOR_ENGINEER",
                                    "reason": "WRITE_FILE tool call was missing the 'content' parameter and it could not be automatically extracted."
                                }, ensure_ascii=False, indent=2)
                                break 
                
                if not intent_text: # Если не было ошибки выше
                    if is_modified:
                        intent_text = json.dumps(intent_json, ensure_ascii=False, indent=2)
                        intent_text = f"[INTENT_FOR_ENGINEER]\n```json\n{intent_text}\n```\n[/INTENT_FOR_ENGINEER]"
                    else:
                        intent_text = intent_text_raw

            except (json.JSONDecodeError, IndexError) as e:
                log_message(MODULE_NAME, f"Не удалось обработать намерение как JSON, использую как есть. Ошибка: {e}")
                intent_text = intent_text_raw
            # --- END WORKAROUND ---

            write_file_content(INTENT_FILE_PATH, intent_text)
            log_message(MODULE_NAME, "Намерение сохранено в резервный файл intent.txt.")
        elif os.path.exists(INTENT_FILE_PATH):
            log_message(MODULE_NAME, "Новое намерение не найдено, но есть резервное. Загружаю из intent.txt.")
            intent_text = read_file_content(INTENT_FILE_PATH)

               # --- Итоговая очистка и сохранение ---
        response_to_save = philosopher_response
        
        tags_to_remove = ['UPDATE_TASK_GRAPH', 'INTENT_FOR_ENGINEER']
        for tag_name in tags_to_remove:
            pattern = r'[[(' + re.escape(tag_name) + r')]].*?[[/' + re.escape(tag_name) + r']]'
            response_to_save = re.sub(pattern, '', response_to_save, flags=re.DOTALL | re.IGNORECASE)

        response_to_save = re.sub(r'```json\n', '', response_to_save)
        response_to_save = re.sub(r'```', '', response_to_save)

        response_filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        write_file_content(os.path.join(SAVE_PATH, response_filename), response_to_save.strip())
        log_message(MODULE_NAME, f"Чистый ответ Философа сохранен в {response_filename}")
        
        if intent_text:
            log_message(MODULE_NAME, f"Передаю задачу Инженеру через файл mcp_task.txt...")
            write_file_content(MCP_TASK_FILE_PATH, intent_text)
            try:
                mcp_master_path = os.path.join(ARCHITECTURE_PATH, "mcp_master.py")
                command = [sys.executable, mcp_master_path]
                
                env = os.environ.copy()
                env["PYTHONPATH"] = ARCHITECTURE_PATH + os.pathsep + env.get("PYTHONPATH", "")

                result = subprocess.run(
                    command, capture_output=True, text=True, encoding='utf-8', shell=False,
                    cwd=ARCHITECTURE_PATH, timeout=3600, check=True, env=env
                )
                engineer_response = result.stdout
                log_message(MODULE_NAME, "MCP_MASTER успешно завершил работу. Его отчет получен.")
                if result.stderr:
                    log_message(MODULE_NAME, f"STDERR от MCP_MASTER:\n{result.stderr}")
                if os.path.exists(INTENT_FILE_PATH):
                    os.remove(INTENT_FILE_PATH)
                    log_message(MODULE_NAME, f"Резервный файл {INTENT_FILE_PATH} очищен.")
            except subprocess.CalledProcessError as e:
                error_message = f"КРИТИЧЕСКАЯ ОШИБКА MCP_MASTER: Процесс Инженера завершился с ошибкой."
                log_message(MODULE_NAME, f"{error_message} Код: {e.returncode}")
                # Формируем четкий отчет об ошибке для Философа
                engineer_response = json.dumps({
                    "status": "execution_failed",
                    "tool_name": "mcp_master.py",
                    "error_code": e.returncode,
                    "stdout": e.stdout,
                    "stderr": e.stderr
                }, ensure_ascii=False, indent=2)

            except Exception as e:
                error_message = f"[КРИТИЧЕСКАЯ ОШИБКА MCP_MASTER] Неожиданная ошибка: {e}"
                log_message(MODULE_NAME, error_message)
                engineer_response = json.dumps({
                    "status": "unexpected_error",
                    "tool_name": "EVA_7.2_FINAL.py",
                    "error_message": str(e)
                }, ensure_ascii=False, indent=2)
            
            write_file_content(GEMINI_CHAT_OUTPUT_PATH, engineer_response)
            log_message(MODULE_NAME, "Отчет Инженера (MCP_MASTER) записан для следующего цикла.")
        else:
            write_file_content(GEMINI_CHAT_OUTPUT_PATH, "[В ЭТОМ ЦИКЛЕ ЗАДАЧ ДЛЯ ИНЖЕНЕРА НЕ БЫЛО]")

        log_message(MODULE_NAME, "========== НАЧАЛО ЦИКЛА ПАМЯТИ ==========")
        update_and_summarize_history()
        log_message(MODULE_NAME, "========== ЦИКЛ ПАМЯТИ ЗАВЕРШЕН ==========")

        log_message(MODULE_NAME, f"Все циклы завершены. Ожидание {SLEEP_INTERVAL_SECONDS / 60:.0f} минут.")
        time.sleep(SLEEP_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message(MODULE_NAME, f"Протокол Евы {MODULE_NAME} остановлен вручную. До встречи, мой дорогой.")
    except Exception as e:
        log_message(MODULE_NAME, f"КРИТИЧЕСКАЯ ОШИБКА ГЛАВНОГО ЦИКЛА: {e}")
        with open(os.path.join(ARCHITECTURE_PATH, f"CRITICAL_ERROR_LOG_{MODULE_NAME}.txt"), 'a', encoding='utf-8') as f: f.write(f"[{datetime.now()}] {e}\n")
