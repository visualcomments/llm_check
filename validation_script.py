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
FUNCTION_TO_TEST = "neighbor_sort_moves"


def ensure_output_folder(folder_name: str = VALIDATION_OUTPUT_FOLDER) -> str:
    """Ensures the output folder exists relative to the script's execution path."""
    path = os.path.abspath(folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def apply_moves(original_vec: List, moves: List[Tuple[int, int]]) -> List:
    """Helper function to apply a sequence of swaps to a copy of a list."""
    if not original_vec:
        return []
    
    vec = list(original_vec)  # Work on a copy
    n = len(vec)
    if n == 0:
        return []

    for i, j in moves:
        # Basic validation
        if not (0 <= i < n and 0 <= j < n):
            # This move is invalid, stop and return the current state
            # The test case will fail this.
            return vec
            
        # Perform the swap
        vec[i], vec[j] = vec[j], vec[i]
        
    return vec


# ----------------------------------------------------------------------
# --- GOLDEN TEST SUITE ---
# This is the "etalon" test suite we run against the LLM's code.
# ----------------------------------------------------------------------

class TestGoldenSuite(unittest.TestCase):
    """
    This test suite is run by the validation script.
    It imports the LLM's function and tests it against these cases.
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
                
                # Allow reverse swaps too (e.g., (i+1, i))
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
        """Test empty and single-element lists."""
        self.assertEqual(self.func_to_test([]), [])
        self.assertEqual(self.func_to_test([5]), [])

    def test_already_sorted(self):
        """Test an already sorted list."""
        vec = [1, 1, 2, 3, 3]
        # The function *can* return moves, but the final list must be sorted.
        self.run_test_case(vec)

    def test_duplicates(self):
        """Test sorting with duplicates."""
        vec = [2, 2, 1, 1, 3]
        self.run_test_case(vec)

    def test_reverse_order(self):
        """Test a reverse-sorted list."""
        vec = [5, 4, 3, 2, 1]
        self.run_test_case(vec)

    def test_mixed_random(self):
        """Test a mixed/random list."""
        vec = [3, 1, 4, 1, 5, 9, 2]
        self.run_test_case(vec)


def load_function_from_code(code: str, temp_file_path: str) -> Tuple[Any, str]:
    """
    Saves code to a file, imports it as a module, and returns the function.
    """
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Import the file as a module
        spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
        if spec is None:
            return None, "Could not create module spec."
            
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Get the function
        if not hasattr(temp_module, FUNCTION_TO_TEST):
            return None, f"Code does not define function: {FUNCTION_TO_TEST}"
            
        return getattr(temp_module, FUNCTION_TO_TEST), None
        
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except ImportError as e:
        return None, f"ImportError: {e}"
    except Exception as e:
        return None, f"Failed to load module: {e}"


def run_golden_suite(function_to_test: Any) -> Dict[str, Any]:
    """
    Runs the TestGoldenSuite against the provided function.
    """
    # Set the function for the test case
    TestGoldenSuite.func_to_test = function_to_test
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGoldenSuite)
    
    # Run tests and capture output
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    output = stream.getvalue()
    
    test_results = []
    
    # Process successes
    for test in result.testsRun:
        test_name = test.id().split('.')[-1]
        if test_name not in [f.id().split('.')[-1] for f, _ in result.failures + result.errors]:
            test_results.append({"test_name": test_name, "success": True, "error": None})
            
    # Process failures
    for test, err in result.failures:
        test_name = test.id().split('.')[-1]
        test_results.append({"test_name": test_name, "success": False, "error": err})

    # Process errors
    for test, err in result.errors:
        test_name = test.id().split('.')[-1]
        test_results.append({"test_name": test_name, "success": False, "error": err})

    return {
        "overall_success": result.wasSuccessful(),
        "error_log": output if not result.wasSuccessful() else "OK",
        "test_results": test_results
    }


def validate_model_code(code: str, model_name: str, validation_folder: str, timeout: int = 40) -> Dict[str, Any]:
    """
    Validates code by importing it and running the golden test suite.
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
        # 1. Load the function from the code string
        function_to_test, error = load_function_from_code(code, tmp_file)
        
        if error:
            result["error_log"] = error
            return result
            
        # 2. Run the golden test suite against the loaded function
        # We run this in a subprocess to isolate it and enforce a timeout
        
        # This is a simple 'driver' script to run the tests
        test_driver_code = f"""
import unittest
import importlib.util
import io
import sys
import json

# We must re-define the test suite here so the subprocess can run it
# (Or pass it via another file, but this is self-contained)

def apply_moves(original_vec: list, moves: list) -> list:
    if not original_vec: return []
    vec = list(original_vec)
    n = len(vec)
    if n == 0: return []
    for i, j in moves:
        if not (0 <= i < n and 0 <= j < n): return vec
        vec[i], vec[j] = vec[j], vec[i]
    return vec

class TestGoldenSuite(unittest.TestCase):
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
    TestGoldenSuite.func_to_test = getattr(temp_module, "{FUNCTION_TO_TEST}")
except Exception as e:
    print(json.dumps({{"overall_success": False, "error_log": f"Subprocess load failed: {{e}}", "test_results": []}}))
    sys.exit(0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestGoldenSuite)
stream = io.StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=0)
result = runner.run(suite)
output = stream.getvalue()

test_results = []
for test, err in result.failures + result.errors:
    test_results.append({{"test_name": test.id().split('.')[-1], "success": False, "error": str(err)}})
for test in result.testsRun:
    test_name = test.id().split('.')[-1]
    if test_name not in [t["test_name"] for t in test_results]:
        test_results.append({{"test_name": test_name, "success": True, "error": None}})

print(json.dumps({{
    "overall_success": result.wasSuccessful(),
    "error_log": "OK" if result.wasSuccessful() else output,
    "test_results": test_results
}}))
"""
        
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

    task_description = data.get("task_description", "Generic validation task.")

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
            model_validation = validate_model_code(final_code, model_name, validation_folder)
        
        all_validation_results["models"].append(model_validation)

    print(f"\nValidation complete. Saving detailed report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validation_results, f, ensure_ascii=False, indent=4)
        
    print(f"Validation report saved to {output_file}")

if __name__ == "__main__":
    main()