import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import threading
import queue
import argparse
import re  # <<< ИСПРАВЛЕНИЕ: импорт перемещен сюда

# -----------------------------
# Утилиты извлечения кода из ответа LLM
# -----------------------------


def extract_python_code(raw: str) -> str:
    """
    Return an importable Python module string from an LLM response.

    Strategy (robust):
    1) Prefer the largest fenced code block tagged as python: ```python ... ```.
    2) Else prefer the largest fenced code block of any language: ``` ... ```.
    3) Else, if the text contains 'def neighbor_sort_moves', return the WHOLE text (do NOT trim imports/helpers).
    4) As a last resort, return the raw text.
    """
    if raw is None:
        return ""
    text = str(raw)

    # 1) Prefer ```python fenced blocks
    py_blocks = re.findall(r"```python\s+([\s\S]*?)```", text, flags=re.IGNORECASE)
    if py_blocks:
        code_str = max(py_blocks, key=len).strip()
    else:
        # 2) Any fenced block
        any_blocks = re.findall(r"```[\w]*\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if any_blocks:
            code_str = max(any_blocks, key=len).strip()
        else:
            # 3) If we at least see the target function name, keep the whole text (to preserve imports/helpers)
            if "def neighbor_sort_moves" in text:
                code_str = text.strip()
            else:
                # 4) Last resort: raw text (maybe it's already plain code)
                code_str = text.strip()

    # Strip trivial HTML noise that sometimes leaks from providers
    code_str = re.sub(r"</?span[^>]*>", "", code_str)
    code_str = re.sub(r"</?audio[^>]*>", "", code_str)
    code_str = re.sub(r"</?source[^>]*>", "", code_str)

    return code_str


# -----------------------------
# Опциональная локальная инференс‑ветка HF
# -----------------------------
_HF_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# -----------------------------
# Tkinter только при наличии DISPLAY
# -----------------------------
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    _TKINTER_AVAILABLE = True
except Exception:
    _TKINTER_AVAILABLE = False

# -----------------------------
# Патч провайдера ротации (для логов AnyProvider)
# -----------------------------
import g4f.providers.retry_provider as retry_mod
OriginalRotatedProvider = retry_mod.RotatedProvider

import g4f
from g4f import Provider
from g4f.errors import ModelNotFoundError

# Thread-local для сбора логов по каждому треду
local = threading.local()

class TrackedRotated(OriginalRotatedProvider):
    async def create_async_generator(self, model: str, messages: List[Dict], **kwargs):
        if not hasattr(local, 'current_data'):
            local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}
            local.current_queue = queue.Queue()

        local.current_data['tried'] = []
        local.current_data['errors'] = {}
        local.current_data['success'] = None

        if hasattr(local, 'current_queue') and self.providers:
            providers_list = [p.__name__ for p in self.providers]
            local.current_queue.put((model, 'log', f"1) Providers found: {providers_list}"))

        for provider_class in self.providers:
            provider_instance = None
            provider_name = provider_class.__name__ if hasattr(provider_class, '__name__') else str(provider_class)

            local.current_data['tried'].append(provider_name)
            if hasattr(local, 'current_queue'):
                local.current_queue.put((model, 'log', f"2) Trying provider: {provider_name}"))

            try:
                provider_instance = provider_class()
                async for chunk in provider_instance.create_async_generator(model, messages, **kwargs):
                    yield chunk

                if hasattr(local, 'current_queue'):
                    local.current_queue.put((model, 'log', f"3) Success from {provider_name}"))
                    local.current_data['success'] = provider_name
                return
            except Exception as e:
                error_str = str(e)
                if hasattr(local, 'current_queue'):
                    local.current_queue.put((model, 'log', f"3) Error from {provider_name}: {error_str}"))
                local.current_data['errors'][provider_name] = error_str
                if provider_instance and hasattr(provider_instance, '__del__'):
                    provider_instance.__del__()
                continue

        raise ModelNotFoundError(f"No working provider found for model {model}", local.current_data['tried'])

# Monkey‑patch
retry_mod.RotatedProvider = TrackedRotated


# -----------------------------
# Конфиг (добавлены пресеты и results_dir по умолчанию совместимый с validation_script)
# -----------------------------
CONFIG = {
    'URLS': {
        'WORKING_RESULTS': 'https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt'
    },
    'PROMPTS': {
        'INITIAL': (
            "You are a professional Python programming assistant. Write a correct, functional, and immediately executable Python module to solve the task below.\n\n"
            "{task}\n\n"
            "STRICT RULES:\n"
            "1) Respond with ONLY Python code. No explanations or Markdown.\n"
            "2) The module must be self-contained and importable. It MUST define the function:\n"
            "   def neighbor_sort_moves(vec: list) -> list:\n"
            "   which returns a list of swap pairs (i, j).\n"
            "3) Each pair (i, j) must be a CYCLIC NEIGHBOR transposition. This means either j == (i+1) mod n OR i == (j+1) mod n, where n=len(vec).\n"
            "4) The function must NOT read input() nor access files/network.\n"
            "5) The module may include helper functions/classes, but the canonical entry point is neighbor_sort_moves(vec).\n"
            "6) Do not print anything besides an optional short demo under if __name__ == \"__main__\": (printing is allowed but not required)."
        ),
        # Новый максимально простой для LLM демо‑промпт
        'DEMO': (
            "You are a Python expert. Implement a simple, correct CYCLIC-NEIGHBOR bubble sort generator.\n\n"
            "Task: Write a self-contained module that defines:\n"
            "    def neighbor_sort_moves(vec: list) -> list\n\n"
            "Return a list of swaps (i, j) that sorts a copy of 'vec' into nondecreasing order using ONLY cyclic-adjacent swaps:\n"
            "  • (i, i+1) for i=0..n-2, and • (n-1, 0).\n\n"
            "Recommended algorithm (easy):\n"
            "  If n<=1: return [].\n"
            "  Repeat up to n*n times: for k in 0..n-1, compare a[k] and a[(k+1)%n]; if a[k] > a[(k+1)%n], swap them and record (k,(k+1)%n).\n"
            "  Stop early if full pass made no swaps.\n\n"
            "Constraints: no I/O, no external deps. Duplicates must be handled. Return moves only."
        ),
        'FIX': (
            "You are a Python debugging assistant. The following code failed. "
            "Return a corrected module that:\n"
            "• defines def neighbor_sort_moves(vec: list) -> list\n"
            "• uses ONLY cyclic neighbor transpositions (i, j), where j == (i+1) mod n OR i == (j+1) mod n.\n"
            "• is importable and self-contained; no input() or external files.\n\n"
            "Faulty Code:\n{code}\n\n"
            "Error Message:\n{error}\n\n"
            "IMPORTANT: Respond with ONLY the corrected Python code."
        ),
        'REFACTOR_NO_PREV': (
            "You are a Python optimization expert. Refactor and improve the following module without changing the public API:\n\n"
            "{code}\n\n"
            "Requirements:\n"
            "• Keep def neighbor_sort_moves(vec: list) -> list\n"
            "• Use ONLY cyclic neighbor transpositions (i, j), where j == (i+1) mod n OR i == (j+1) mod n.\n"
            "• Improve readability, performance, and correctness.\n"
            "• Keep it self-contained and importable.\n"
            "Respond with ONLY the refactored code."
        ),
        'REFACTOR': (
            "You are a Python optimization expert. Compare the current and previous versions and produce the best refactored module.\n\n"
            "Current Code:\n{code}\n\n"
            "Previous Version:\n{prev}\n\n"
            "Requirements:\n"
            "• Keep def neighbor_sort_moves(vec: list) -> list\n"
            "• Use ONLY cyclic neighbor transpositions (i, j), where j == (i+1) mod n OR i == (j+1) mod n.\n"
            "• Improve readability, performance, and correctness.\n"
            "• Keep it self-contained and importable.\n"
            "Respond with ONLY the newest, refactored code."
        )
    },
    'RETRIES': {
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
        'FIX': {'max_retries': 3, 'backoff_factor': 2.0}
    },
    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 30,
        'N_SAVE': 100,
        'MAX_WORKERS': 50,
        'EXEC_TIMEOUT': 30,
        'ERROR_TIMEOUT': 'Timeout expired during code execution.',
        'ERROR_NO_RESPONSE': 'No response from model.',
        'NUM_REFACTOR_LOOPS': 2,
        # По умолчанию — совместимо с validation_script.py
        'RESULTS_FOLDER': '/kaggle/working/run_results',
    },
    'STAGES': {
        'INITIAL': 'initial_response',
        'FIX_INITIAL': 'fix_before_refactor',
        'REFACTOR_FIRST': 'first_refactor_response',
        'FIX_AFTER_REFACTOR': 'fix_after_refactor',
        'REFACTOR': 'refactor_loop',
        'FIX_LOOP': 'fix_in_loop'
    }
}

# -----------------------------
# Выбор моделей к запуску
# -----------------------------

def get_models_list(config: Dict, args) -> List[str]:
    """
    Возвращает список моделей:
    • Если есть --hf_model и доступен transformers — тестируем её локально.
    • Иначе: берём список из g4f working + пересекаем с allowlist через --models (если задан).
    """
    models: List[str] = []
    if args.hf_model:
        if not _HF_AVAILABLE:
            print("[WARN] transformers/torch not available. Cannot run local HF model.", file=sys.stderr)
        else:
            models.append(f"hf_local::{args.hf_model}")
        return models

    url_txt = config['URLS']['WORKING_RESULTS']
    try:
        resp = requests.get(url_txt, timeout=config['CONSTANTS']['REQUEST_TIMEOUT'])
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException:
        text = ''

    working_models = set()
    for line in text.splitlines():
        if config['CONSTANTS']['DELIMITER_MODEL'] in line:
            parts = [p.strip() for p in line.split(config['CONSTANTS']['DELIMITER_MODEL'])]
            if len(parts) == 3 and parts[2] == config['CONSTANTS']['MODEL_TYPE_TEXT']:
                model_name = parts[1]
                if 'flux' not in model_name.lower():
                    working_models.add(model_name)

    try:
        from g4f.models import Model
        all_g4f_models = Model.__all__()
        g4f_models = {
            model_name for model_name in all_g4f_models
            if 'flux' not in model_name.lower() and not any(sub in model_name.lower() for sub in ['image', 'vision', 'audio', 'video'])
        }
    except Exception:
        g4f_models = set()

    models = sorted(list(working_models.union(g4f_models)))
    if args.models:
        allow = set(m.strip() for m in args.models.split(",") if m.strip())
        models = [m for m in models if m in allow]
    return models


# -----------------------------
# Локальный HF запуск
# -----------------------------
_hf_cache = {}

def hf_local_query(model_id_or_path: str, prompt: str, max_new_tokens: int = 800, temperature: float = 0.2) -> Optional[str]:
    if not _HF_AVAILABLE:
        return None
    key = model_id_or_path
    if key not in _hf_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _hf_cache[key] = (tokenizer, model)
    tokenizer, model = _hf_cache[key]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    text = tokenizer.decode(output_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return text.strip()


# -----------------------------
# Унифицированный запрос к LLM (HF локально или g4f)
# -----------------------------

def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    """
    Опрашивает либо локальную HF‑модель, либо AnyProvider (g4f).
    """
    local.current_model = model
    local.current_queue = progress_queue
    local.current_stage = stage
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}

    # Локально через HF
    if model.startswith("hf_local::"):
        model_id = model.split("::", 1)[1]
        try:
            response = hf_local_query(model_id, prompt)
            return response
        except Exception as e:
            local.current_data['errors'][model] = str(e)
            return None

    # Провайдеры g4f
    for attempt in range(retries_config['max_retries'] + 1):
        try:
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                provider=Provider.AnyProvider,
                timeout=config['CONSTANTS']['REQUEST_TIMEOUT'],
            )
            if response and str(response).strip():
                return str(response).strip()
        except ModelNotFoundError as e:
            if len(e.args) > 1 and hasattr(local, 'current_data'):
                local.current_data['tried'] = e.args[1]
            return None
        except Exception:
            pass

        if attempt < retries_config['max_retries']:
            time.sleep(retries_config['backoff_factor'] * (2 ** attempt))

    return None


# -----------------------------
# Безопасный пробный запуск с таймаутом
# -----------------------------

def safe_execute(code: str, config: Dict) -> Tuple[bool, str]:
    """
    Выполняет код в подпроцессе с таймаутом — базовая проверка синтаксиса/рантайма.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=config['CONSTANTS']['EXEC_TIMEOUT'],
        )
        if result.returncode == 0:
            return True, result.stdout or ''
        else:
            return False, result.stderr or 'Unknown error during execution.'
    except subprocess.TimeoutExpired:
        return False, config['CONSTANTS']['ERROR_TIMEOUT']
    except Exception as e:
        return False, str(e)


# -----------------------------
# Основной конвейер для одной модели
# -----------------------------

def process_model(model: str, task: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Выполняет цикл: генерация → (при необходимости) фиксы → рефакторинги.
    """
    iterations = []
    current_code = None
    prev_code = None

    progress_queue.put((model, 'status', f"Starting: {config['STAGES']['INITIAL']}"))
    progress_queue.put((model, 'log', f"=== STARTING MODEL: {model} ==="))

    # 1) Начальный запрос
    prompt = config['PROMPTS']['INITIAL'].format(task=task)
    progress_queue.put((model, 'log', f"Stage: {config['STAGES']['INITIAL']}. Firing prompt."))
    response = llm_query(model, prompt, config['RETRIES']['INITIAL'], config, progress_queue, config['STAGES']['INITIAL'])

    iterations.append({
        'providers_tried': local.current_data.get('tried', []),
        'success_provider': local.current_data.get('success'),
        'stage': config['STAGES']['INITIAL'],
        'response': response,
        'error': None if response else config['CONSTANTS']['ERROR_NO_RESPONSE'],
    })

    if not response:
        progress_queue.put((model, 'status', f"Failed: {config['STAGES']['INITIAL']}"))
        progress_queue.put((model, 'done', None))
        return {'model': model, 'iterations': iterations, 'final_code': None}

    current_code = extract_python_code(response)
    progress_queue.put((model, 'status', "Initial code received. Executing..."))

    # 2) Этапы фиксов/рефакторов
    pipeline_stages = [
        (CONFIG['STAGES']['FIX_INITIAL'], CONFIG['PROMPTS']['REFACTOR_NO_PREV'], False),
        (CONFIG['STAGES']['FIX_AFTER_REFACTOR'], CONFIG['PROMPTS']['REFACTOR'], True),
    ]
    for _ in range(CONFIG['CONSTANTS']['NUM_REFACTOR_LOOPS']):
        pipeline_stages.append((CONFIG['STAGES']['FIX_LOOP'], CONFIG['PROMPTS']['REFACTOR'], True))

    for fix_stage, refactor_prompt, use_prev in pipeline_stages:
        # A) Пробуем исполнить текущий код
        progress_queue.put((model, 'log', f"Executing code from previous stage..."))
        success_exec, output = safe_execute(current_code, config)

        if not success_exec:
            # B) Чиним, если упал
            progress_queue.put((model, 'status', f"Execution failed. Attempting fix: {fix_stage}"))
            progress_queue.put((model, 'log', f"Execution failed. Error:\n{output}"))
            prompt = CONFIG['PROMPTS']['FIX'].format(code=current_code, error=str(output))
            response = llm_query(model, prompt, CONFIG['RETRIES']['FIX'], config, progress_queue, fix_stage)

            iterations.append({
                'providers_tried': local.current_data.get('tried', []),
                'success_provider': local.current_data.get('success'),
                'stage': fix_stage,
                'response': response,
                'error': None if response else CONFIG['CONSTANTS']['ERROR_NO_RESPONSE'],
            })

            if not response:
                progress_queue.put((model, 'status', f"Failed to fix code at stage: {fix_stage}"))
                progress_queue.put((model, 'done', None))
                return {'model': model, 'iterations': iterations, 'final_code': None}
            current_code = extract_python_code(response)

        # C) Рефакторим всегда
        refactor_stage = fix_stage.replace('fix', 'refactor')
        progress_queue.put((model, 'status', f"Refactoring code: {refactor_stage}"))

        if use_prev:
            prompt = refactor_prompt.format(code=current_code, prev=prev_code)
        else:
            prompt = refactor_prompt.format(code=current_code)

        response = llm_query(model, prompt, CONFIG['RETRIES']['FIX'], config, progress_queue, refactor_stage)

        iterations.append({
            'providers_tried': local.current_data.get('tried', []),
            'success_provider': local.current_data.get('success'),
            'stage': refactor_stage,
            'response': response,
            'error': None if response else CONFIG['CONSTANTS']['ERROR_NO_RESPONSE'],
        })

        if not response:
            progress_queue.put((model, 'status', f"Failed to refactor at stage: {refactor_stage}"))
            # Продолжаем с текущим кодом
            continue

        prev_code = current_code
        current_code = extract_python_code(response)

    progress_queue.put((model, 'status', 'Completed'))
    progress_queue.put((model, 'log', f"=== FINAL CODE:\n{current_code or 'None'}"))
    progress_queue.put((model, 'log', f"=== FINISHED MODEL: {model} ==="))
    progress_queue.put((model, 'done', None))
    return {'model': model, 'iterations': iterations, 'final_code': current_code}


# -----------------------------
# Оркестратор
# -----------------------------

def orchestrator(task: str, models: List[str], config: Dict, progress_queue: queue.Queue) -> Dict:
    """Параллельный запуск по всем моделям."""
    folder = config['CONSTANTS']['RESULTS_FOLDER']
    os.makedirs(folder, exist_ok=True)
    results = {}
    total_models = len(models)

    with ThreadPoolExecutor(max_workers=config['CONSTANTS']['MAX_WORKERS']) as executor:
        future_to_model = {executor.submit(process_model, model, task, config, progress_queue): model for model in models}
        completed = 0
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results[result['model']] = result
            except Exception as e:
                results[model] = {'model': model, 'iterations': [], 'final_code': None, 'error': str(e)}
                progress_queue.put((model, 'error', str(e)))

            completed += 1
            progress_queue.put(('global', 'progress', (completed, total_models)))

    final_results = {'results': list(results.values()), 'timestamp': datetime.now().isoformat()}
    final_file = os.path.join(folder, 'final_results.json')
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    progress_queue.put(('global', 'done', f"Processing complete. Final results saved to {final_file}"))
    return final_results


# -----------------------------
# (Опционально) GUI-заглушка
# -----------------------------
if _TKINTER_AVAILABLE:
    class ProgressGUI:
        # GUI тут не нужен.
        pass


# -----------------------------
# Консольный режим
# -----------------------------

def run_headless_orchestrator(task: str, models: List[str], config: Dict):
    progress_queue = queue.Queue()

    orchestrator_thread = threading.Thread(
        target=orchestrator,
        args=(task, models, config, progress_queue),
        daemon=True,
    )
    orchestrator_thread.start()

    completed_count = 0
    total_count = len(models)
    active_model = ""
    active_status = ""

    while orchestrator_thread.is_alive() or not progress_queue.empty():
        try:
            model, msg_type, data = progress_queue.get(timeout=1)

            if model == 'global':
                if msg_type == 'done':
                    print(f"\n[INFO] {data}")
                    break
                elif msg_type == 'progress':
                    completed_count, total_count = data
            else:
                active_model = model
                if msg_type == 'status':
                    active_status = data
                elif msg_type == 'error':
                    active_status = f"ERROR: {data}"

            progress_bar = f"[{('#' * (completed_count * 30 // max(1, total_count)))}{('-' * ((total_count - completed_count) * 30 // max(1, total_count)))}]"
            print(f"\rProgress: {completed_count}/{total_count} {progress_bar} | Current: {active_model} - {active_status.ljust(40)}", end="", flush=True)

        except queue.Empty:
            continue

    print("\n\nAll tasks finished.")


# -----------------------------
# Тексты задач (пресеты)
# -----------------------------

def build_default_task_text() -> str:
    """Изначальная формулировка задачи (adjacent + циклическая пара)."""
    return (
        """
Task: Implement a sorting algorithm that uses ONLY adjacent swaps on a vector.

Input: A vector a of length n (0-indexed).

Allowed operation:
- swap (i, i+1) for i = 0..n-2
- swap (n-1, 0)

Output: Return a list of swaps as pairs of indices in the form (i, i+1) and, when needed, (n-1, 0).
Applying these swaps in the given order must transform the original vector into nondecreasing order.
Do not use any other operations.
"""
    )


def build_demo_task_text() -> str:
    """Упрощённый для LLM демо‑промпт (на основе циклического пузырька)."""
    return (
        """
Task: Sort a list into nondecreasing order by returning a sequence of CYCLIC-ADJACENT swaps only.

Requirements:
- Define def neighbor_sort_moves(vec: list) -> list
- Allowed swaps only: (i, i+1) for i=0..n-2 and (n-1, 0)
- If n<=1: return []
- Use a bubble-like n*n outer limit with early stop when a full pass makes no swaps.
- No I/O, no imports, return the *moves list* only.

Tip: On each pass k=0..n-1 compare a[k] and a[(k+1)%n], swap if needed and record the pair.
"""
    )


def load_task_text(args) -> str:
    # 1) Внешний файл с задачей имеет приоритет
    if args.task_file:
        try:
            with open(args.task_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Failed to read --task-file: {e}. Falling back to preset.")
    # 2) Пресеты
    if args.task_preset == 'demo':
        return build_demo_task_text()
    return build_default_task_text()


# -----------------------------
# Аргументы CLI (новые: --task-preset, --task-file, --results-dir)
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM code generation runner with optional local HF model support")
    p.add_argument("--hf_model", type=str, default=None, help="Local Hugging Face model id or path to test (uses transformers).")
    p.add_argument("--models", type=str, default=None, help="Comma-separated allowlist of g4f models to test.")
    p.add_argument("--max_workers", type=int, default=CONFIG['CONSTANTS']['MAX_WORKERS'], help="Thread pool size.")
    p.add_argument("--task-preset", choices=["default", "demo"], default="default", help="Which built-in task text to use.")
    p.add_argument("--task-file", type=str, default=None, help="Path to a custom task text file (overrides preset).")
    p.add_argument("--results-dir", type=str, default=CONFIG['CONSTANTS']['RESULTS_FOLDER'], help="Where to save final_results.json (default matches validation_script).")
    return p.parse_args()


# -----------------------------
# main
# -----------------------------

def main():
    args = parse_args()

    # Обновляем конфиг из аргументов
    CONFIG['CONSTANTS']['MAX_WORKERS'] = int(args.max_workers)
    if args.results_dir:
        CONFIG['CONSTANTS']['RESULTS_FOLDER'] = args.results_dir

    # Готовим текст задачи согласно опциям
    task_description = load_task_text(args)
    print(f"Using task preset: {('file:'+args.task_file) if args.task_file else args.task_preset}")

    print("Selecting models...")
    models = get_models_list(CONFIG, args)
    print(f"Found {len(models)} model(s) to test.")

    test_models = models
    if not test_models:
        print("No models found. Exiting.")
        return

    if _TKINTER_AVAILABLE and os.environ.get('DISPLAY'):
        print("Display found, but GUI is not implemented in this build; falling back to headless.")
        run_headless_orchestrator(task_description, test_models, CONFIG)
    else:
        print("Running in headless (console) mode...")
        run_headless_orchestrator(task_description, test_models, CONFIG)


if __name__ == "__main__":
    main()
