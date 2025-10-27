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
import re
import tempfile

# -----------------------------
# Code extraction utilities from LLM response
# -----------------------------

def extract_python_code(raw: str) -> str:
    """
    Return an importable Python module string from an LLM response.
    (Код не изменен)
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
            # 3) If we at least see a function definition, keep the whole text
            if "def " in text:
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
# Optional local HF inference branch
# (Код не изменен)
# -----------------------------
_HF_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# -----------------------------
# Provider rotation patch (for AnyProvider logs)
# (Код не изменен)
# -----------------------------
import g4f.providers.retry_provider as retry_mod
OriginalRotatedProvider = retry_mod.RotatedProvider

import g4f
from g4f import Provider
from g4f.errors import ModelNotFoundError

# Thread-local for collecting logs per thread
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

# Monkey-patch
retry_mod.RotatedProvider = TrackedRotated


# -----------------------------
# Config (Generic Prompts)
# --- УПРОЩЕНО ---
# -----------------------------
CONFIG = {
    'URLS': {
        'WORKING_RESULTS': '[https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt](https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt)'
    },
    'PROMPTS': {
        # Оставляем только один промпт, который будет отформатирован заданием
        'INITIAL': (
            "You are a professional Python programming assistant. Write a correct, functional, and immediately executable Python module to solve the task below.\n\n"
            "## Full Task Description\n"
            "{task}\n\n"
            "## Strict Rules\n"
            "1. Respond with ONLY the complete Python code module. Do not add any explanations, preamble, or markdown formatting around the code block.\n"
            "2. The code must be self-contained, importable, and executable.\n"
            "3. The code MUST implement all requirements, functions, and behaviors specified in the task description.\n"
        ),
    },
    'RETRIES': {
        # Оставляем только одну конфигурацию попыток
        'INITIAL': {'max_retries': 1, 'backoff_factor': 1.0},
    },
    'CONSTANTS': {
        'DELIMITER_MODEL': '|',
        'MODEL_TYPE_TEXT': 'text',
        'REQUEST_TIMEOUT': 30,
        'MAX_WORKERS': 50,
        'ERROR_NO_RESPONSE': 'No response from model.',
        'RESULTS_FOLDER': './run_results', # Используем относительный путь
    },
    'STAGES': {
        # Оставляем только один этап
        'INITIAL': 'initial_response',
    }
}

# -----------------------------
# Model selection
# (Код не изменен)
# -----------------------------

def get_models_list(config: Dict, args) -> List[str]:
    """
    Returns a list of models:
    • If --hf_model is provided and transformers is available, test it locally.
    • Otherwise: get the list from g4f working + intersect with allowlist via --models (if given).
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
# Local HF execution
# (Код не изменен)
# -----------------------------
_hf_cache = {}

def hf_local_query(model_id_or_path: str, prompt: str, max_new_tokens: int = 1500, temperature: float = 0.2) -> Optional[str]:
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
# Unified query to LLM (local HF or g4f)
# (Код не изменен)
# -----------------------------

def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    """
    Queries either a local HF model or AnyProvider (g4f).
    """
    local.current_model = model
    local.current_queue = progress_queue
    local.current_stage = stage
    local.current_data = {'tried': [], 'errors': {}, 'success': None, 'model': model}

    # Locally via HF
    if model.startswith("hf_local::"):
        model_id = model.split("::", 1)[1]
        try:
            response = hf_local_query(model_id, prompt)
            return response
        except Exception as e:
            local.current_data['errors'][model] = str(e)
            return None

    # g4f providers
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
# safe_execute (REMOVED)
# -----------------------------
# Функция safe_execute удалена, так как проверка
# теперь выполняется исключительно в validation_script.py


# -----------------------------
# Main pipeline for a single model
# --- СИЛЬНО УПРОЩЕНО ---
# -----------------------------

def process_model(model: str, task: str, config: Dict, progress_queue: queue.Queue) -> Dict:
    """
    Executes the simple cycle: generate -> extract.
    No more fix or refactor loops.
    """
    iterations = []
    current_code = None

    stage = config['STAGES']['INITIAL']
    progress_queue.put((model, 'status', f"Starting: {stage}"))
    progress_queue.put((model, 'log', f"=== STARTING MODEL: {model} ==="))

    # 1) Initial request
    prompt = config['PROMPTS']['INITIAL'].format(task=task)
    progress_queue.put((model, 'log', f"Stage: {stage}. Firing prompt."))
    response = llm_query(model, prompt, config['RETRIES']['INITIAL'], config, progress_queue, stage)

    iteration_data = {
        'providers_tried': local.current_data.get('tried', []),
        'success_provider': local.current_data.get('success'),
        'stage': stage,
        'response': response,
        'error': None if response else config['CONSTANTS']['ERROR_NO_RESPONSE'],
    }
    iterations.append(iteration_data)

    if not response:
        progress_queue.put((model, 'status', f"Failed: {stage}"))
        progress_queue.put((model, 'done', None))
        return {'model': model, 'iterations': iterations, 'final_code': None}

    # 2) Extract code
    current_code = extract_python_code(response)
    progress_queue.put((model, 'status', "Initial code received."))
    
    # 3) Done
    progress_queue.put((model, 'status', 'Completed'))
    progress_queue.put((model, 'log', f"=== FINAL CODE:\n{current_code or 'None'}"))
    progress_queue.put((model, 'log', f"=== FINISHED MODEL: {model} ==="))
    progress_queue.put((model, 'done', None))
    
    return {'model': model, 'iterations': iterations, 'final_code': current_code}


# -----------------------------
# Orchestrator
# (Код не изменен)
# -----------------------------

def orchestrator(task: str, models: List[str], config: Dict, progress_queue: queue.Queue) -> Dict:
    """Parallel launch for all models."""
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

    # *** УЛУЧШЕНИЕ: Добавляем 'task_description' в итоговый JSON ***
    final_results = {
        'task_description': task, # Это важно для validation_script
        'results': list(results.values()), 
        'timestamp': datetime.now().isoformat()
    }
    
    final_file = os.path.join(folder, 'final_results.json')
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    progress_queue.put(('global', 'done', f"Processing complete. Final results saved to {final_file}"))
    return final_results


# -----------------------------
# (Optional) GUI stub
# -----------------------------
# (Удалено для простоты, т.к. не использовалось)


# -----------------------------
# Console mode
# (Код не изменен)
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
# Task texts (presets)
# --- УПРОЩЕНО ---
# -----------------------------

def build_default_task_text() -> str:
    """The 'neighbor_sort_moves' task (from task.txt)."""
    # Этот текст больше не просит LLM генерировать тесты
    return (
        """
You are a senior Python engineer. Implement a simple algorithm in Python that produces a sequence of CYCLIC-ADJACENT swaps which, when applied in order, sorts a list in nondecreasing order.

## Task
Write a standalone Python module that defines exactly:
    def neighbor_sort_moves(vec: list) -> list:
This function must return a list of index pairs (i, j) representing swaps.
Applying all swaps in order to a copy of vec must result in a nondecreasing list.

## Allowed swaps ONLY
- Adjacent neighbors: (i, i+1) for i = 0 .. n-2
- Wrap-around neighbors: (n-1, 0)
No other swap pairs are permitted.

## Constraints & Behavior
- If n == 0 or n == 1, return [].
- Handle duplicates correctly.
- Do NOT print anything. No file/network I/O. No external imports.
- Return only the moves list.
- Indices must always be valid (0 <= i, j < n).
- Keep swaps strictly to the allowed neighbor pairs above.
- Do NOT use built-ins that solve the task directly for you (e.g., calling sort() to derive the move sequence).
- Time/termination: use a simple bubble-like process around the ring;
stop early if a full pass makes no swaps, and in any case cap the process to at most n*n steps to prevent infinite loops.

## Code quality
- Include type hints and a clear docstring (describe allowed swaps and behavior).
- Keep the implementation short and readable; add minimal comments.
- Follow PEP 8.

## Answer format
- The final answer MUST be a single Python code block with no extra commentary.
"""
    )


def build_demo_task_text() -> str:
    """The 'find_max' demo task."""
    # Этот текст также больше не просит LLM генерировать тесты
    return (
        """
You are a senior Python engineer. Implement a very simple algorithm in Python: find the maximum value in a list using a single linear scan.

## Task
Write a standalone Python module that defines:
    def find_max(nums: list[int]) -> int:

## Requirements
- Do NOT use built-ins that solve the task directly (e.g., max(), sorted(), numpy).
- Time complexity: O(n). Extra space: O(1) aside from variables.
- Behavior:
  - If nums is empty, raise ValueError("empty sequence").
  - Support negative numbers, duplicates, and single-element lists.
- Code quality:
  - Include type hints and a clear docstring with examples.
  - Keep the implementation short and readable; add minimal explanatory comments.
  - Follow PEP 8.

## Answer format
- The final answer MUST be a single Python code block with no extra commentary.
"""
    )


def load_task_text(args) -> str:
    # 1) External task file has priority
    if args.task_file:
        try:
            with open(args.task_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Failed to read --task-file: {e}. Falling back to preset.")
    # 2) Presets
    if args.task_preset == 'demo':
        return build_demo_task_text()
    return build_default_task_text()


# -----------------------------
# CLI arguments
# (Код не изменен, но --task-file теперь важнее)
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM code generation runner (Simplified)")
    p.add_argument("--hf_model", type=str, default=None, help="Local Hugging Face model id or path to test (uses transformers).")
    p.add_argument("--models", type=str, default=None, help="Comma-separated allowlist of g4f models to test.")
    p.add_argument("--max_workers", type=int, default=CONFIG['CONSTANTS']['MAX_WORKERS'], help="Thread pool size.")
    p.add_argument("--task-preset", choices=["default", "demo"], default="default", help="Which built-in task text to use. 'default' is neighbor_sort_moves, 'demo' is find_max.")
    p.add_argument("--task-file", type=str, default=None, help="Path to a custom task text file (overrides preset). **THIS IS THE RECOMMENDED WAY**")
    p.add_argument("--results-dir", type=str, default=CONFIG['CONSTANTS']['RESULTS_FOLDER'], help="Where to save final_results.json (default matches validation_script).")
    return p.parse_args()


# -----------------------------
# main
# (Код не изменен)
# -----------------------------

def main():
    args = parse_args()

    # Update config from arguments
    CONFIG['CONSTANTS']['MAX_WORKERS'] = int(args.max_workers)
    if args.results_dir:
        CONFIG['CONSTANTS']['RESULTS_FOLDER'] = args.results_dir

    # Prepare the task text according to options
    task_description = load_task_text(args)
    if args.task_file:
         print(f"Using task from file: {args.task_file}")
    else:
         print(f"Using built-in task preset: {args.task_preset}")


    print("Selecting models...")
    models = get_models_list(CONFIG, args)
    print(f"Found {len(models)} model(s) to test.")

    test_models = models
    if not test_models:
        print("No models found. Exiting.")
        return

    print("Running in headless (console) mode...")
    run_headless_orchestrator(task_description, test_models, CONFIG)


if __name__ == "__main__":
    main()
