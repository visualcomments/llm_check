# -*- coding: utf-8 -*-
import json
import subprocess
import sys
import os
import time
import re
import unittest
import importlib.util
import io
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
RESULTS_DIR = "./run_results"
VALIDATION_OUTPUT_FOLDER = "./validation_output"

# ----------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ----------------------------------------------------------------------

def ensure_output_folder(folder_name: str = VALIDATION_OUTPUT_FOLDER) -> str:
    """Гарантирует существование каталога для артефактов валидации."""
    path = os.path.abspath(folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def extract_python_code(raw: Optional[str]) -> str:
    """
    Извлекает python-код из LLM-ответа.
    Приоритет:
      1) самый большой fenced-блок с ```python ... ```
      2) самый большой fenced-блок с ``` ... ```
      3) если в тексте есть 'def ', возвращаем текст как есть
      4) иначе — сырой текст (strip)
    Дополнительно чистим тривиальные HTML-теги, которые иногда протекают.
    """
    if not raw:
        return ""
    text = str(raw)

    # 1) ```python ... ```
    py_blocks = re.findall(r"```python\s+([\s\S]*?)```", text, flags=re.IGNORECASE)
    if py_blocks:
        code = max(py_blocks, key=len).strip()
    else:
        # 2) ``` ... ```
        any_blocks = re.findall(r"```[\w]*\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if any_blocks:
            code = max(any_blocks, key=len).strip()
        else:
            # 3) есть сигнатуры?
            if "def " in text:
                code = text.strip()
            else:
                # 4) сырой текст
                code = text.strip()

    # Очистка шума
    code = re.sub(r"</?span[^>]*>", "", code)
    code = re.sub(r"</?audio[^>]*>", "", code)
    code = re.sub(r"</?source[^>]*>", "", code)
    return code


def load_function_from_code(code: str, temp_file_path: str, function_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Сохраняет code во временный .py, импортирует модуль и возвращает ссылку на функцию.
    """
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
        if spec is None or spec.loader is None:
            return None, "Failed to create module spec."
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)  # type: ignore[attr-defined]

        if not hasattr(temp_module, function_name):
            return None, f"Function '{function_name}' not found in module."
        return getattr(temp_module, function_name), None
    except Exception as e:
        return None, f"Import error: {e}"


# ----------------------------------------------------------------------
# ТЕСТОВЫЕ ДРАЙВЕРЫ (исполняются в подпроцессе через -c)
# ----------------------------------------------------------------------

TEST_DRIVER_SORT = r"""
import sys, json, unittest, importlib.util

TMP_FILE = r"{tmp_file}"
FUNC_NAME = "{FUNCTION_TO_TEST}"

# Тесты
def is_cyclic_neighbor(i: int, j: int, n: int) -> bool:
    return j == (i + 1) % n or i == (j + 1) % n

def apply_moves(vec, moves):
    a = list(vec)
    n = len(a)
    for (i, j) in moves:
        assert 0 <= i < n and 0 <= j < n, "Out-of-range index"
        assert is_cyclic_neighbor(i, j, n), "Non-cyclic-neighbor swap"
        a[i], a[j] = a[j], a[i]
    return a

class TestGoldenSuiteSort(unittest.TestCase):
    func_to_test = None

    def setUp(self):
        self.assertIsNotNone(self.func_to_test, "neighbor_sort_moves not loaded")

    def check_case(self, vec):
        moves = self.func_to_test(list(vec))
        self.assertIsInstance(moves, list, "Result must be a list of swaps")
        out = apply_moves(vec, moves)
        self.assertEqual(out, sorted(vec), "Sequence must be sorted ascending")

    def test_small(self):
        self.check_case([3,2,1])

    def test_wrap(self):
        self.check_case([2,1,0,3])

    def test_dups(self):
        self.check_case([2,2,1,3,3,0])

    def test_negatives(self):
        self.check_case([5,-1,4,-2,0])

def load_func(path: str, name: str):
    spec = importlib.util.spec_from_file_location("tmp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, name)

def run():
    suite = unittest.TestSuite()
    try:
        func = load_func(TMP_FILE, FUNC_NAME)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Import error: {e}"}))
        return
    TestGoldenSuiteSort.func_to_test = func
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestGoldenSuiteSort))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=0)
    result = runner.run(suite)
    payload = {
        "ok": result.wasSuccessful(),
        "failures": [(str(t[0]), t[1]) for t in result.failures],
        "errors": [(str(t[0]), t[1]) for t in result.errors],
    }
    print(json.dumps(payload))

if __name__ == "__main__":
    run()
"""

TEST_DRIVER_FIND_MAX = r"""
import sys, json, unittest, importlib.util

TMP_FILE = r"{tmp_file}"
FUNC_NAME = "{FUNCTION_TO_TEST}"

class TestFindMax(unittest.TestCase):
    func_to_test = None

    def setUp(self):
        self.assertIsNotNone(self.func_to_test, "find_max not loaded")

    def test_empty(self):
        with self.assertRaises(ValueError):
            self.func_to_test([])

    def test_single(self):
        self.assertEqual(self.func_to_test([7]), 7)

    def test_negatives(self):
        self.assertEqual(self.func_to_test([-3, -9, -1]), -1)

    def test_mix_dups(self):
        self.assertEqual(self.func_to_test([1, 3, 2, 3]), 3)

def load_func(path: str, name: str):
    spec = importlib.util.spec_from_file_location("tmp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, name)

def run():
    suite = unittest.TestSuite()
    try:
        func = load_func(TMP_FILE, FUNC_NAME)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Import error: {e}"}))
        return
    TestFindMax.func_to_test = func
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestFindMax))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=0)
    result = runner.run(suite)
    payload = {
        "ok": result.wasSuccessful(),
        "failures": [(str(t[0]), t[1]) for t in result.failures],
        "errors": [(str(t[0]), t[1]) for t in result.errors],
    }
    print(json.dumps(payload))

if __name__ == "__main__":
    run()
"""

# ----------------------------------------------------------------------
# ОСНОВНАЯ ВАЛИДАЦИЯ ОДНОЙ МОДЕЛИ
# ----------------------------------------------------------------------

def validate_model_code(
    code: str,
    model_name: str,
    validation_folder: str,
    function_name: str,
    test_driver_template: str,
    timeout: int = 40,
) -> Dict[str, Any]:
    """
    Валидирует код: пишет во временный .py, импортирует функцию и гоняет нужный golden suite.
    Запускает тесты в отдельном Python-процессе и парсит JSON с результатом.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_]+', '_', model_name)
    tmp_file = os.path.join(validation_folder, f"temp_validate_{sanitized}_{int(time.time())}.py")

    result: Dict[str, Any] = {
        "model": model_name,
        "overall_success": False,
        "error_log": None,
        "test_results": []
    }

    # 1) Предварительная проверка импорта
    _, import_error = load_function_from_code(code, tmp_file, function_name)
    if import_error:
        result["error_log"] = import_error
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return result

    # 2) Запуск драйвера тестов в подпроцессе
    driver_code = test_driver_template.format(tmp_file=tmp_file, FUNCTION_TO_TEST=function_name)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", driver_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result["error_log"] = "Execution timed out."
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return result
    except Exception as e:
        result["error_log"] = f"Subprocess error: {e}"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return result

    # 3) Парсим вывод драйвера
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # Драйвер печатает РОВНО один JSON объект. На всякий случай берём последний {...}
    m = re.findall(r"\{[\s\S]*\}", stdout)
    driver_payload: Dict[str, Any]
    if m:
        try:
            driver_payload = json.loads(m[-1])
        except Exception as e:
            result["error_log"] = f"Failed to parse driver JSON: {e}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            return result
    else:
        result["error_log"] = f"No JSON from driver.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        return result

    ok = bool(driver_payload.get("ok"))
    result["overall_success"] = ok
    if not ok:
        fails = driver_payload.get("failures", [])
        errs = driver_payload.get("errors", [])
        text = []
        if fails:
            text.append("Failures:\n" + "\n".join(f"- {name}\n{tb}" for name, tb in fails))
        if errs:
            text.append("Errors:\n" + "\n".join(f"- {name}\n{tb}" for name, tb in errs))
        if not text:
            text.append("Unknown test failure.")
        result["error_log"] = "\n\n".join(text)

    # 4) Убираем временный файл
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    return result


# ----------------------------------------------------------------------
# ОСНОВНОЙ СЦЕНАРИЙ
# ----------------------------------------------------------------------

def main():
    final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
    validation_folder = ensure_output_folder()
    output_file = os.path.join(validation_folder, "comprehensive_validation.json")

    if not os.path.exists(final_results_file):
        print(f"Error: The results file was not found at '{final_results_file}'")
        print("Please run the CallLLM script first to generate the results.")
        return

    print(f"Loading results from {final_results_file}...")
    with open(final_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Определяем, какие тесты запускать ---
    task_description: str = data.get("task_description", "") or ""
    if "find_max" in task_description:
        print("Detected 'find_max' task.")
        function_name = "find_max"
        test_driver = TEST_DRIVER_FIND_MAX
    else:
        print("Defaulting to 'neighbor_sort_moves' task.")
        function_name = "neighbor_sort_moves"
        test_driver = TEST_DRIVER_SORT

    # --- Результирующий контейнер ---
    summary: Dict[str, Any] = {
        "validation_timestamp": datetime.now().isoformat(),
        "source_file": final_results_file,
        "validation_task": task_description,
        "models": []
    }

    results = data.get("results", [])
    print(f"Starting validation for {len(results)} models...")

    for idx, item in enumerate(results, 1):
        model_name = item.get("model", f"Unknown_{idx}")

        # 1) Сначала пробуем canonical поле
        final_code: Optional[str] = item.get("final_code")
        code_to_test = (final_code or "").strip()

        # 2) Если кода нет — пробуем вытащить из последнего удачного response
        if not code_to_test:
            iters = item.get("iterations", []) or []
            last_response = ""
            for it in reversed(iters):
                resp = (it or {}).get("response")
                if resp and str(resp).strip():
                    last_response = str(resp).strip()
                    break
            extracted = extract_python_code(last_response)
            if extracted.strip():
                code_to_test = extracted.strip()

        print(f"\r({idx}/{len(results)}) Validating model: {model_name.ljust(36)}", end="", flush=True)

        if not code_to_test:
            summary["models"].append({
                "model": model_name,
                "overall_success": False,
                "error_log": "No final_code and no usable response found.",
                "test_results": []
            })
            continue

        model_validation = validate_model_code(
            code=code_to_test,
            model_name=model_name,
            validation_folder=validation_folder,
            function_name=function_name,
            test_driver_template=test_driver,
            timeout=40,
        )
        summary["models"].append(model_validation)

    print("\nValidation complete. Saving detailed report...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Для быстрой сводки в CSV — сделаем лёгкую агрегацию
    try:
        import pandas as pd
        rows = []
        for m in summary["models"]:
            rows.append({
                "model": m["model"],
                "overall_success": bool(m["overall_success"]),
                "error_log": m.get("error_log") or "",
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(validation_folder, "analysis_data.csv"), index=False, encoding="utf-8")
    except Exception:
        pass

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
