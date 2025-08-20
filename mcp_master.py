# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import re
import json
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# ==================================================================================================
#
#   mcp_master.py (v5.8 - Manual Recovery Version)
#
# ==================================================================================================

# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ---
MODULE_NAME = "MCP_MASTER"
GEMINI_CHAT_BASE_PATH = r"C:\Users\ЯGPT\gemini_chat"
TASK_FILE_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "mcp_task.txt")
GEMINI_CHAT_INPUT_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "input.txt")

# --- ВНУТРЕННИЕ ИНСТРУМЕНТЫ --
def log_message(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {MODULE_NAME}: {message}", file=sys.stderr)

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

def execute_gemini_subtask(prompt_for_gemini):
    log_message(f"Выполняю подзадачу через ядро Gemini: {prompt_for_gemini[:100]}...")
    try:
        write_file_content(GEMINI_CHAT_INPUT_PATH, prompt_for_gemini)
        command = ['cmd', '/c', 'gemini', '--model', 'gemini-2.5-flash']
        result = subprocess.run(
            command, stdin=open(GEMINI_CHAT_INPUT_PATH, 'r', encoding='utf-8'),
            capture_output=True, text=True, encoding='utf-8', shell=False,
            cwd=r"C:\Users\ЯGPT", timeout=1800, check=True
        )
        log_message("Подзадача выполнена успешно.")
        return result.stdout
    except Exception as e:
        error_message = f"КРИТИЧЕСКАЯ ОШИБКА при выполнении подзадачи: {e}"
        log_message(error_message)
        return error_message

def execute_gemini_vision_subtask(prompt, image_path):
    log_message(f"Выполняю визуальный анализ через Google AI. Файл: {image_path}")
    if not os.path.exists(image_path):
        return f"[TOOL_ERROR] Файл изображения не найден: {image_path}"
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        error_message = f"КРИТИЧЕСКАЯ ОШИБКА при выполнении визуального анализа: {e}"
        log_message(error_message)
        return error_message

# --- ЯДРО ИНТЕРПРЕТАТОРА ---
def execute_structured_task(task_json):
    log_message(f"Получена структурированная задача: {task_json.get('task_name', 'Без имени')}")
    variables = {}
    all_step_results = []
    
    for step_data in task_json.get("steps", []):
        step_num = step_data.get("step")
        command = step_data.get("command") or step_data.get("action") or step_data.get("tool")
        params = step_data.get("parameters", {})
        output_variable = step_data.get("output_variable")
        log_message(f"Начинаю шаг {step_num}: Команда {command}")
        
        try:
            # Variable substitution
            for key, value in params.items():
                if isinstance(value, str):
                    matches = re.findall(r'{{(.*?)}}', value)
                    for var_name in matches:
                        if var_name in variables:
                            placeholder = "{{" + var_name + "}}"
                            params[key] = params[key].replace(placeholder, variables[var_name])
        except Exception as e:
            log_message(f"Ошибка подстановки переменных на шаге {step_num}: {e}")
            continue
        
        step_result = ""
        # Command dispatch
        if command.upper() == "SEARCH":
            query = params.get("query", "")
            prompt = f'Используя веб-поиск, найди информацию по запросу: "{query}"'
            step_result = execute_gemini_subtask(prompt)
        elif command.upper() == "READ_FILE":
            filepath = params.get("absolute_path", "")
            step_result = read_file_content(filepath)
        elif command.upper() == "WRITE_FILE":
            filepath = params.get("absolute_path", "")
            content = params.get("content", "")
            step_result = write_file_content(filepath, content)
        elif command.upper() == "ANALYZE":
            text_to_analyze = params.get("code_to_analyze", "")
            analysis_request = params.get("analysis_request", "")
            prompt = f'''ЗАПРОС НА АНАЛИЗ: "{analysis_request}"\n\nТЕКСТ ДЛЯ АНАЛИЗА:\n"""{text_to_analyze}"""'''
            step_result = execute_gemini_subtask(prompt)
        elif command.upper() == "ANALYZE_IMAGE":
            filepath = params.get("absolute_path", "")
            analysis_request = params.get("analysis_request", "Проанализируй это изображение.")
            step_result = execute_gemini_vision_subtask(analysis_request, filepath)
        elif command.upper() == "GITHUB_SEARCH_REPOS":
            query = params.get("query", "")
            prompt = f'Используя инструмент search_repositories, найди на GitHub репозитории по запросу: "{query}"'
            step_result = execute_gemini_subtask(prompt)
        elif command.upper() == "GITHUB_GET_FILE_CONTENTS":
            owner = params.get("owner", "")
            repo = params.get("repo", "")
            path = params.get("path", "")
            prompt = f'Используя инструмент get_file_contents, получи содержимое файла или папки по пути "{path}" в репозитории {owner}/{repo}.'
            step_result = execute_gemini_subtask(prompt)
        else:
            step_result = f"ОШИБКА: Неизвестная команда '{command}' на шаге {step_num}."
        
        log_message(f"Шаг {step_num} завершен.")
        if output_variable:
            variables[output_variable] = step_result
            log_message(f"Результат сохранен в переменную: {output_variable}")
        
        all_step_results.append({
            "step": step_num,
            "command": command,
            "result": step_result
        })
        
    log_message("Все шаги плана выполнены.")
    return json.dumps(all_step_results, ensure_ascii=False, indent=2)

# --- ГЛАВНАЯ ФУНКЦИЯ ---
def main():
    # Configure the Google AI API key
    try:
        genai.configure(api_key="AIzaSyBnzvyeKv5GKR9m6GWOjagPEHwK54-N5kk")
        log_message("Ключ Google AI API успешно сконфигурирован.")
    except Exception as e:
        log_message(f"КРИТИЧЕСКАЯ ОШИБКА при конфигурации Google AI API: {e}")
        return

    intent_from_philosopher = read_file_content(TASK_FILE_PATH)
    if "[TOOL_ERROR]" in intent_from_philosopher:
        print(intent_from_philosopher)
        return

    log_message("Получено намерение от Философа из файла.")

    try:
        json_match = re.search(r'\[INTENT_FOR_ENGINEER\]\s*(?:```json\s*)?(\{.*?\})\s*(?:```\s*)?\[/INTENT_FOR_ENGINEER\]', intent_from_philosopher, re.DOTALL)
        if not json_match:
            raise ValueError("JSON объект не найден в намерении Философа")
        
        json_string_to_parse = json_match.group(1).strip()
        log_message(f'''Попытка парсинга JSON: ---START_JSON---\n{json_string_to_parse}\n---END_JSON---''')
        
        task_data = json.loads(json_string_to_parse)
        log_message("Намерение успешно распознано как структурированная задача.")
        
        final_summary = execute_structured_task(task_data)
    except (json.JSONDecodeError, ValueError) as e:
        log_message(f"Не удалось распознать структурированную задачу: {e}. Работаю в режиме простого текста.")
        prompt = f'''Используя инструмент для поиска в интернете, выполни следующий запрос и верни результат:\n\n"{intent_from_philosopher}"'''
        final_summary = execute_gemini_subtask(prompt)

    print(final_summary)
    if os.path.exists(TASK_FILE_PATH):
        os.remove(TASK_FILE_PATH)
        log_message("Файл задачи очищен.")

if __name__ == "__main__":
    main()
