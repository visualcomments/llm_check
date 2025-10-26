import json
import subprocess
import sys
import os
import time
import re
from typing import Dict, List, Any
from datetime import datetime

# --- CONFIGURATION ---
RESULTS_DIR = "./run_results"
VALIDATION_OUTPUT_FOLDER = "./validation_output"

def ensure_output_folder(folder_name: str = VALIDATION_OUTPUT_FOLDER) -> str:
    """Ensures the output folder exists relative to the script's execution path."""
    path = os.path.abspath(folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def parse_unittest_output(stderr: str, stdout: str) -> Dict[str, Any]:
    """
    Парсит вывод stderr/stdout из unittest для извлечения результатов по каждому тесту.
    """
    results = []
    overall_success = False
    error_summary = ""

    # Паттерны Regex
    # Ловит: test_basic (test_module.TestClass) ... ok
    re_ok = re.compile(r"^(test_.*?) \((.*?)\) \.\.\. ok", re.MULTILINE)
    # Ловит: FAIL: test_fail (test_module.TestClass) ИЛИ ERROR: test_error (...)
    # И захватывает весь блок ошибки до следующего разделителя
    re_fail_header = re.compile(
        r"^=+\n(FAIL|ERROR): (.*?) \((.*?)\)\n-+\n([\s\S]*?)(?=\n(?:=|\-){20,})", 
        re.MULTILINE
    )
    # Ловит: Ran X tests... OK
    re_final_ok = re.compile(r"^Ran \d+ test.*?\n\nOK\s*$", re.DOTALL)
    # Ловит: Ran X tests... FAILED (failures=X, errors=Y)
    re_final_fail = re.compile(r"FAILED \((.*?)\)\s*$", re.DOTALL)

    # 1. Проверяем общий статус (OK, FAILED, или синтаксическая ошибка)
    
    if re_final_ok.search(stderr):
        # Все тесты прошли
        overall_success = True
        error_summary = "OK"
    else:
        match_fail = re_final_fail.search(stderr)
        if match_fail:
            # Тесты запускались, но провалились
            overall_success = False
            error_summary = f"FAILED ({match_fail.group(1)})"
        else:
            # Это не вывод unittest. Вероятно, SyntaxError или другая ошибка времени выполнения.
            overall_success = False
            error_summary = stderr.strip() if stderr.strip() else stdout.strip()
            # Возвращаемся раньше, т.к. парсить отдельные тесты нет смысла
            return {"test_results": [], "overall_success": False, "error_log": error_summary}

    # 2. Парсим результаты отдельных тестов

    # Собираем тесты, которые провалились (FAIL/ERROR)
    failed_tests = {}
    for m in re_fail_header.finditer(stderr):
        test_name = m.group(2)
        error_detail = m.group(4).strip()
        failed_tests[test_name] = error_detail
        results.append({"test_name": test_name, "success": False, "error": error_detail})

    # Собираем тесты, которые прошли (ok)
    ok_tests_found = set()
    for m in re_ok.finditer(stderr):
        test_name = m.group(1)
        ok_tests_found.add(test_name)
        if test_name not in failed_tests:
            results.append({"test_name": test_name, "success": True, "error": None})

    # Если общий статус "OK", но мы не нашли 'ok' тестов (напр. Ran 0 tests)
    if overall_success and not results:
        error_summary = stdout or "OK (Ran 0 tests)"
    
    # Если мы нашли тесты, но 'error_summary' все еще 'OK', используем stdout
    if overall_success:
        error_summary = stdout or "OK"

    return {"test_results": results, "overall_success": overall_success, "error_log": error_summary}


def validate_model_code(code: str, model_name: str, validation_folder: str, timeout: int = 40) -> Dict[str, Any]:
    """
    Валидирует код, запуская его unittests и парся результаты.
    """
    sanitized_model_name = re.sub(r'[^A-Za-z0-9_]+', '_', model_name)
    tmp_file = os.path.join(validation_folder, f"temp_validate_{sanitized_model_name}_{int(time.time())}.py")
    
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(code)

    result = {
        "model": model_name,
        "overall_success": False,
        "error_log": None,
        "test_results": []  # <--- НОВОЕ ПОЛЕ ДЛЯ АНАЛИЗА
    }

    try:
        proc = subprocess.run(
            [sys.executable, tmp_file], 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace',
            timeout=timeout
        )
        
        full_stderr = proc.stderr or ""
        full_stdout = proc.stdout or ""
        
        # *** УЛУЧШЕНИЕ: Вызываем новый парсер ***
        parsed_data = parse_unittest_output(full_stderr, full_stdout)
        
        result.update(parsed_data)
        
        # Если парсер ничего не нашел (напр. SyntaxError), но код вернул ошибку,
        # то 'error_log' уже был установлен парсером как полный stderr.
        if not parsed_data["test_results"] and proc.returncode != 0:
            result["overall_success"] = False
            # Убедимся, что error_log не пустой
            if not result["error_log"]:
                 result["error_log"] = full_stderr if full_stderr else full_stdout
        
        # Если код выполнился (returncode 0), но парсер не нашел тестов
        elif proc.returncode == 0 and not parsed_data["test_results"]:
            result["overall_success"] = True # Считаем успехом (скрипт отработал)
            result["error_log"] = full_stdout or "Script ran successfully (no tests found/run)."

            
    except subprocess.TimeoutExpired:
        result["overall_success"] = False
        result["error_log"] = "Execution timed out."
        result["test_results"] = []
    except Exception as e:
        result["overall_success"] = False
        result["error_log"] = f"Validation harness failed: {str(e)}"
        result["test_results"] = []
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return result

def main():
    final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
    output_filename = "comprehensive_validation.json"
    
    validation_folder = ensure_output_folder()
    output_file = os.path.join(validation_folder, output_filename)
    
    if not os.path.exists(final_results_file):
        print(f"Error: The results file was not found at '{final_results_file}'")
        print("Please run the CallLLM script first to generate the results.")
        return

    print(f"Loading results from {final_results_file}...")
    with open(final_results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # *** УЛУЧШЕНИЕ: Читаем описание задачи из файла CallLLM ***
    task_description = data.get("task_description", "Generic validation by executing the script's unittests.")

    all_validation_results = {
        "validation_timestamp": datetime.now().isoformat(),
        "source_file": final_results_file,
        "validation_task": task_description, # <--- Используем реальную задачу
        "models": []
    }

    results = data.get('results', [])
    print(f"Starting validation for {len(results)} models...")

    for i, result in enumerate(results, 1):
        model_name = result.get('model', f'Unknown_{i}')
        final_code = result.get('final_code')
        
        print(f"\r({i}/{len(results)}) Validating model: {model_name.ljust(40)}", end="", flush=True)
        
        if not final_code or not final_code.strip():
            model_validation = {
                "model": model_name,
                "overall_success": False,
                "error_log": "No final_code was generated by the model.",
                "test_results": [] # <--- Добавляем пустое поле для согласованности
            }
        else:
            # Вызываем обновленную функцию валидации
            model_validation = validate_model_code(final_code, model_name, validation_folder)
        
        all_validation_results["models"].append(model_validation)

    print(f"\nValidation complete. Saving detailed report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validation_results, f, ensure_ascii=False, indent=4)
        
    print(f"Validation report saved to {output_file}")

if __name__ == "__main__":
    main()