import json
import subprocess
import sys
import os
import time
import re
import unittest
import importlib.util
import io
from typing import Dict, List, Any, Tuple
from datetime import datetime

# --- CONFIGURATION ---
RESULTS_DIR = "./run_results"
VALIDATION_OUTPUT_FOLDER = "./validation_output"
# DEFAULT_FUNCTION_TO_TEST = "neighbor_sort_moves" # Больше не одна константа


def ensure_output_folder(folder_name: str = VALIDATION_OUTPUT_FOLDER) -> str:
    """Ensures the output folder exists relative to the script's execution path."""
    path = os.path.abspath(folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def apply_moves(original_vec: List, moves: List[Tuple[int, int]]) -> List:
    """Helper function to apply a sequence of swaps to a copy of a list. (Used by Sort Test)"""
    if not original_vec:
        return []
    
    vec = list(original_vec)  # Work on a copy
    n = len(vec)
    if n == 0:
        return []

    for i, j in moves:
        if not (0 <= i < n and 0 <= j < n):
            return vec
        vec[i], vec[j] = vec[j], vec[i]
        
    return vec


# ----------------------------------------------------------------------
# --- GOLDEN TEST SUITE (SORT TASK) ---
# (Код не изменен)
# ----------------------------------------------------------------------

class TestGoldenSuiteSort(unittest.TestCase):
    """
    This test suite is run by the validation script for the 'neighbor_sort_moves' task.
    """
    # This will be populated by the validator before running tests
    func_to_test = None

    def setUp(self):
        """Ensure the function to test is available."""
        self.assertIsNotNone(
            self.func_to_test,
            "The function to test was not loaded correctly."
        )

    def run_test_case(self, vec):
        """Helper to run a full test on a vector."""
        n = len(vec)
        moves = self.func_to_test(vec)
        
        # 1. Apply moves
        result_vec = apply_moves(vec, moves)
        
        # 2. Check if sorted
        expected_vec = sorted(vec)
        self.assertEqual(
            result_vec, 
            expected_vec,
            f"Failed to sort: {vec}. Got: {result_vec}, Expected: {expected_vec}"
        )

        # 3. Check move legality
        if n > 1:
            for i, j in moves:
                self.assertTrue(0 <= i < n, f"Invalid move index i: {i} for n={n}")
                self.assertTrue(0 <= j < n, f"Invalid move index j: {j} for n={n}")
                
                is_adjacent = (j == (i + 1))
                is_wrap = (i == (n - 1) and j == 0)
                is_rev_adjacent = (i == (j + 1))
                is_rev_wrap = (j == (n - 1) and i == 0)
                
                self.assertTrue(
                    is_adjacent or is_wrap or is_rev_adjacent or is_rev_wrap,
                    f"Illegal move: ({i}, {j}). Not cyclic-adjacent."
                )

        # 4. Check safety bound
        if n > 1:
            self.assertLessEqual(
                len(moves),
                n * n,
                f"Exceeded safety bound: {len(moves)} moves for n={n}"
            )

    def test_empty_and_single(self):
        self.assertEqual(self.func_to_test([]), [])
        self.assertEqual(self.func_to_test([5]), [])
    def test_already_sorted(self):
        self.run_test_case([1, 1, 2, 3, 3])
    def test_duplicates(self):
        self.run_test_case([2, 2, 1, 1, 3])
    def test_reverse_order(self):
        self.run_test_case([5, 4, 3, 2, 1])
    def test_mixed_random(self):
        self.run_test_case([3, 1, 4, 1, 5, 9, 2])


# ----------------------------------------------------------------------
# --- GOLDEN TEST SUITE (FIND_MAX TASK) ---
# (Код не изменен)
# ----------------------------------------------------------------------

class TestGoldenSuiteFindMax(unittest.TestCase):
    """
    This test suite is run by the validation script for the 'find_max' task.
    """
    # This will be populated by the validator before running tests
    func_to_test = None

    def setUp(self):
        """Ensure the function to test is available."""
        self.assertIsNotNone(
            self.func_to_test,
            "The function 'find_max' was not loaded correctly."
        )

    def test_basic_case(self):
        self.assertEqual(self.func_to_test([3, 1, 7, 2]), 7)

    def test_all_negatives(self):
        self.assertEqual(self.func_to_test([-5, -2, -9]), -2)

    def test_single_element(self):
        self.assertEqual(self.func_to_test([42]), 42)

    def test_duplicates(self):
        self.assertEqual(self.func_to_test([5, 5, 5]), 5)

    def test_empty_list_raises_error(self):
        """Test if the function raises ValueError for an empty list."""
        # Use assertRaisesRegex for more specific error message matching
        with self.assertRaisesRegex(ValueError, "empty sequence"):
            self.func_to_test([])


# ----------------------------------------------------------------------
# --- Validation Logic ---
# (Код не изменен)
# ----------------------------------------------------------------------

def load_function_from_code(code: str, temp_file_path: str, function_name: str) -> Tuple[Any, str]:
    """
    Saves code to a file, imports it as a module, and returns the specified function.
    """
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
        if spec is None:
            return None, "Could not create module spec."
            
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Use the dynamic function_name
        if not hasattr(temp_module, function_name):
            return None, f"Code does not define function: {function_name}"
            
        return getattr(temp_module, function_name), None
        
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except ImportError as e:
        return None, f"ImportError: {e}"
    except Exception as e:
        return None, f"Failed to load module: {e}"

# ----------------------------------------------------------------------
# --- Test Driver Strings for Subprocess ---
# ----------------------------------------------------------------------

# This driver is for the 'neighbor_sort_moves' task
# (Код не изменен)
TEST_DRIVER_SORT = """
import unittest
import importlib.util
import io
import sys
import json

# --- Helper function for this test driver ---
def apply_moves(original_vec: list, moves: list) -> list:
    if not original_vec: return []
    vec = list(original_vec)
    n = len(vec)
    if n == 0: return []
    for i, j in moves:
        if not (0 <= i < n and 0 <= j < n): return vec
        vec[i], vec[j] = vec[j], vec[i]
    return vec

# --- Definition of the Test Suite for the subprocess ---
class TestGoldenSuiteSort(unittest.TestCase):
    func_to_test = None
    
    def setUp(self):
        self.assertIsNotNone(self.func_to_test, "Func not loaded")

    def run_test_case(self, vec):
        n = len(vec)
        moves = self.func_to_test(vec)
        result_vec = apply_moves(vec, moves)
        expected_vec = sorted(vec)
        self.assertEqual(result_vec, expected_vec, f"Failed: {{vec}}. Got: {{result_vec}}, Exp: {{expected_vec}}")
        if n > 1:
            for i, j in moves:
                self.assertTrue(0 <= i < n and 0 <= j < n, f"Invalid index")
                is_adj = (j == (i + 1))
                is_wrap = (i == (n - 1) and j == 0)
                is_rev_adj = (i == (j + 1))
                is_rev_wrap = (j == (n - 1) and i == 0)
                self.assertTrue(is_adj or is_wrap or is_rev_adj or is_rev_wrap, f"Illegal move: ({{i}}, {{j}})")
            self.assertLessEqual(len(moves), n * n, f"Bound exceeded")

    def test_empty_and_single(self):
        self.assertEqual(self.func_to_test([]), [])
        self.assertEqual(self.func_to_test([5]), [])
    def test_already_sorted(self):
        self.run_test_case([1, 1, 2, 3, 3])
    def test_duplicates(self):
        self.run_test_case([2, 2, 1, 1, 3])
    def test_reverse_order(self):
        self.run_test_case([5, 4, 3, 2, 1])
    def test_mixed_random(self):
        self.run_test_case([3, 1, 4, 1, 5, 9, 2])

# --- Main execution logic for subprocess ---
try:
    spec = importlib.util.spec_from_file_location("temp_module", r"{tmp_file}")
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)
    # {FUNCTION_TO_TEST} is formatted by the caller
    TestGoldenSuiteSort.func_to_test = getattr(temp_module, "{FUNCTION_TO_TEST}")
except Exception as e:
    print(json.dumps({{"overall_success": False, "error_log": f"Subprocess load failed: {{e}}", "test_results": []}}))
    sys.exit(0)

loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(TestGoldenSuiteSort)
all_test_names = loader.getTestCaseNames(TestGoldenSuiteSort) # <-- ПОЛУЧАЕМ ИМЕНА

stream = io.StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=0)
result = runner.run(suite)
output = stream.getvalue()

test_results = []
failed_tests = set() # <-- ИСПОЛЬЗУЕМ SET
for test, err in result.failures + result.errors:
    test_name = test.id().split('.')[-1]
    test_results.append({{"test_name": test_name, "success": False, "error": str(err)}})
    failed_tests.add(test_name)

for test_name in all_test_names: # <-- ИТЕРИРУЕМ ПО ИМЕНАМ
    if test_name not in failed_tests:
        test_results.append({{"test_name": test_name, "success": True, "error": None}})

print(json.dumps({{
    "overall_success": result.wasSuccessful(),
    "error_log": "OK" if result.wasSuccessful() else output,
    "test_results": test_results
}}))
"""


# This driver is for the 'find_max' task
# (Код не изменен)
TEST_DRIVER_FIND_MAX = """
import unittest
import importlib.util
import io
import sys
import json

# --- Definition of the Test Suite for the subprocess ---
class TestGoldenSuiteFindMax(unittest.TestCase):
    func_to_test = None
    
    def setUp(self):
        self.assertIsNotNone(self.func_to_test, "Func 'find_max' not loaded")

    def test_basic_case(self):
        self.assertEqual(self.func_to_test([3, 1, 7, 2]), 7)

    def test_all_negatives(self):
        self.assertEqual(self.func_to_test([-5, -2, -9]), -2)

    def test_single_element(self):
        self.assertEqual(self.func_to_test([42]), 42)

    def test_duplicates(self):
        self.assertEqual(self.func_to_test([5, 5, 5]), 5)

    def test_empty_list_raises_error(self):
        with self.assertRaisesRegex(ValueError, "empty sequence"):
            self.func_to_test([])

# --- Main execution logic for subprocess ---
try:
    spec = importlib.util.spec_from_file_location("temp_module", r"{tmp_file}")
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)
    # {FUNCTION_TO_TEST} is formatted by the caller
    TestGoldenSuiteFindMax.func_to_test = getattr(temp_module, "{FUNCTION_TO_TEST}")
except Exception as e:
    print(json.dumps({{"overall_success": False, "error_log": f"Subprocess load failed: {{e}}", "test_results": []}}))
    sys.exit(0)

# --- ИСПРАВЛЕННАЯ ЛОГИКА СБОРА ТЕСТОВ ---
loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(TestGoldenSuiteFindMax)
all_test_names = loader.getTestCaseNames(TestGoldenSuiteFindMax) # <-- ПОЛУЧАЕМ ИМЕНА

stream = io.StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=0)
result = runner.run(suite)
output = stream.getvalue()

test_results = []
failed_tests = set() # <-- ИСПОЛЬЗУЕМ SET
for test, err in result.failures + result.errors:
    test_name = test.id().split('.')[-1]
    test_results.append({{"test_name": test_name, "success": False, "error": str(err)}})
    failed_tests.add(test_name)

for test_name in all_test_names: # <-- ИТЕРИРУЕМ ПО ИМЕНАМ
    if test_name not in failed_tests:
        test_results.append({{"test_name": test_name, "success": True, "error": None}})
# --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

print(json.dumps({{
    "overall_success": result.wasSuccessful(),
    "error_log": "OK" if result.wasSuccessful() else output,
    "test_results": test_results
}}))
"""


def validate_model_code(
    code: str, 
    model_name: str, 
    validation_folder: str, 
    function_name: str,               # <-- DYNAMIC
    test_driver_template: str,        # <-- DYNAMIC
    timeout: int = 40
) -> Dict[str, Any]:
    """
    Validates code by importing it and running the correct golden test suite.
    (Код не изменен)
    """
    sanitized_model_name = re.sub(r'[^A-Za-z0-9_]+', '_', model_name)
    tmp_file = os.path.join(validation_folder, f"temp_validate_{sanitized_model_name}_{int(time.time())}.py")
    
    result = {
        "model": model_name,
        "overall_success": False,
        "error_log": None,
        "test_results": []
    }

    try:
        # 1. Load the function from the code string (pre-check)
        # Pass the dynamic function_name
        _, error = load_function_from_code(code, tmp_file, function_name)
        
        if error:
            result["error_log"] = error
            # Clean up the temp file even on pre-check failure
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            return result
            
        # 2. Run the golden test suite in a subprocess
        
        # Format the correct test driver string
        test_driver_code = test_driver_template.format(
            tmp_file=tmp_file, 
            FUNCTION_TO_TEST=function_name
        )
        
        proc = subprocess.run(
            [sys.executable, "-c", test_driver_code],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        
        # Parse the JSON output from the subprocess
        try:
            parsed_data = json.loads(proc.stdout)
            result.update(parsed_data)
        except json.JSONDecodeError:
            result["overall_success"] = False
            result["error_log"] = "Failed to parse subprocess JSON output.\n" + (proc.stderr or proc.stdout)

    except subprocess.TimeoutExpired:
        result["overall_success"] = False
        result["error_log"] = "Execution timed out."
    except Exception as e:
        result["overall_success"] = False
        result["error_log"] = f"Validation harness failed: {str(e)}"
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return result


def main():
    # (Код не изменен)
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

    # --- DYNAMIC TASK SELECTION ---
    task_description = data.get('task_description', "") 
    
    current_function_name: str
    current_test_driver: str

    if "find_max" in task_description:
        print("Detected 'find_max' task.")
        current_function_name = "find_max"
        current_test_driver = TEST_DRIVER_FIND_MAX
    else:
        # 'neighbor_sort_moves' или любой другой текст из prompt.txt
        # будет по умолчанию использовать тесты 'neighbor_sort_moves'.
        # Это ЛОГИКА, которую нужно будет расширить, если вы добавите больше задач.
        print("Defaulting to 'neighbor_sort_moves' task logic.")
        current_function_name = "neighbor_sort_moves"
        current_test_driver = TEST_DRIVER_SORT
    # --------------------------------

    all_validation_results = {
        "validation_timestamp": datetime.now().isoformat(),
        "source_file": final_results_file,
        "validation_task": task_description,
        "models": []
    }

    results = data.get('results', [])
    print(f"Starting validation for {len(results)} models...")

    for i, result in enumerate(results, 1):
        model_name = result.get('model', f'Unknown_{i}')
        # Мы берем final_code, который теперь является
        # единственным сгенерированным кодом
        final_code = result.get('final_code') 
        
        print(f"\r({i}/{len(results)}) Validating model: {model_name.ljust(40)}", end="", flush=True)
        
        if not final_code or not final_code.strip():
            model_validation = {
                "model": model_name,
                "overall_success": False,
                "error_log": "No final_code was generated by the model.",
                "test_results": []
            }
        else:
            # Pass the dynamic function name and test driver
            model_validation = validate_model_code(
                final_code, 
                model_name, 
                validation_folder, 
                current_function_name,  # <-- Pass dynamic name
                current_test_driver     # <-- Pass dynamic driver
            )
        
        all_validation_results["models"].append(model_validation)

    print(f"\nValidation complete. Saving detailed report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validation_results, f, ensure_ascii=False, indent=4)
        
    print(f"Validation report saved to {output_file}")

if __name__ == "__main__":
    main()
