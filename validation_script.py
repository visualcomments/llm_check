import json
import subprocess
import sys
import os
import time
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import random

def ensure_output_folder(folder_name: str = "validation_output") -> str:
    BASE_DIR = os.environ.get("RESULTS_BASE", os.getcwd())
    path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_expected_behavior() -> Dict[str, Any]:
    return {
        "description": "Sorting via ONLY cyclic neighbor transpositions. The model must define def neighbor_sort_moves(vec: list) -> list.",
        "validation_criteria": [
            "Code must execute without errors and be importable.",
            "neighbor_sort_moves(vec) must exist and return a list of pairs (i, j).",
            "Each pair (i, j) must be a cyclic neighbor: (j == (i+1) % n) OR (i == (j+1) % n).",
            "Applying the returned moves to the original vector must produce sorted(vec) (nondecreasing).",
            "All indices must be valid for the original vector length."
        ]
    }

def is_cyclic_neighbor_pair(i: int, j: int, n: int) -> bool:
    """
    Checks if i and j are neighbors on a cycle of length n.
    This is symmetric: (i, j) is valid if (j, i) is.
    """
    if n <= 1:
        return False
    # Check for (i, i+1) or (j, j+1) allowing for wrap-around
    return (j == (i + 1) % n) or (i == (j + 1) % n)

def apply_moves(vec: List[Any], moves: List[Tuple[int, int]]) -> List[Any]:
    """Applies a list of swaps to a copy of the vector."""
    n = len(vec)
    a = list(vec) # Work on a copy
    for (i, j) in moves:
        if not (0 <= i < n and 0 <= j < n):
            # Handle n=0 case gracefully
            if n == 0 and i == 0 and j == 0:
                 raise IndexError(f"Invalid indices for n=0: {(i,j)}")
            if n > 0:
                raise IndexError(f"Swap indices out of range: {(i,j)} for n={n}")
            # if n=0, and moves are empty, this loop won't run.
            # if n=0 and moves are not empty, it's an error.
            
        if not is_cyclic_neighbor_pair(i, j, n):
            raise ValueError(f"Illegal move: {(i,j)} is not a cyclic neighbor swap for n={n}")
        a[i], a[j] = a[j], a[i]
    return a

def make_test_vectors() -> Dict[int, List[List[int]]]:
    """Generates a comprehensive set of test vectors for n=0..9."""
    random.seed(42)
    tests: Dict[int, List[List[int]]] = {}
    
    # Edge cases
    tests[0] = [[]]
    tests[1] = [[42], [1]]
    tests[2] = [[1, 0], [0, 1], [5, 5]]
    
    # Standard cases
    for n in range(3, 10):  # lengths 3..9
        vecs = []
        vecs.append(list(range(n)))                          # already sorted
        vecs.append(list(range(n-1, -1, -1)))                 # strictly decreasing
        v = list(range(n))
        random.shuffle(v); vecs.append(v)                      # random shuffle 1
        v = list(range(n))
        random.shuffle(v); vecs.append(v)                      # random shuffle 2
        if n >= 5:
            dup = list(range(n))
            dup[0] = dup[1]
            dup[-1] = dup[-2]
            random.shuffle(dup)
            vecs.append(dup)                                   # with duplicates
        tests[n] = vecs
    return tests

def adapt_code_for_testing(original_code: str, vectors_by_n: Dict[int, List[List[int]]]) -> str:
    """
    Builds a test harness that silences any prints from the candidate module,
    then imports its neighbor_sort_moves and runs it on our test vectors.
    Each line printed is a JSON with fields: n, vec, moves.
    """
    # We pass the vectors as a JSON string to the harness
    test_vectors_json = json.dumps(vectors_by_n, ensure_ascii=False)
    
    test_harness = f"""
import json, sys, os, traceback

# --- Silence any print during import/execution in the candidate ---
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    # ---- BEGIN CANDIDATE CODE ----
{original_code}
    # ---- END CANDIDATE CODE ----
    
    # Try to access the required function
    fn = neighbor_sort_moves
except Exception as e:
    # Restore stdout to report the import error
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    print(json.dumps({{"error": "Failed to import or find neighbor_sort_moves: " + str(e), "traceback": traceback.format_exc()}}))
    raise SystemExit(1)
finally:
    # Ensure streams are restored even if import fails
    if not sys.stdout.closed:
        sys.stdout.close()
    if not sys.stderr.closed:
        sys.stderr.close()
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr

def _run_tests(vectors_by_n):
    for n_str, vecs in vectors_by_n.items():
        n = int(n_str)
        for vec in vecs:
            try:
                # Pass a COPY of the vector, as required by the prompt
                moves = fn(list(vec))
                print(json.dumps({{"n": n, "vec": vec, "moves": moves}}, ensure_ascii=False))
            except Exception as e:
                print(json.dumps({{"n": n, "vec": vec, "error": str(e), "traceback": traceback.format_exc()}}))

if __name__ == "__main__":
    _vectors = json.loads(\"\"\"{test_vectors_json}\"\"\")
    _run_tests(_vectors)
"""
    return test_harness

def parse_runner_output(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # Ignore lines that aren't valid JSON
            pass
    return out



def _sanitize(code: str) -> str:
    """Make candidate code importable by stripping markdown fences/HTML and trimming to the entry function."""
    if not code:
        return ""
    CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    text = str(code)
    blocks = CODE_BLOCK_RE.findall(text)
    text = max(blocks, key=len) if blocks else text
    text = re.sub(r"</?span[^>]*>", "", text)
    if "def neighbor_sort_moves" in text:
        text = text[text.index("def neighbor_sort_moves"):]
    return text.strip()
def validate_model_code(code: str, model_name: str, vectors_by_n: Dict[int, List[List[int]]], timeout: int = 40) -> Dict[str, Any]:
    validation_folder = ensure_output_folder()

    # Create a unique-ish temp file for the harness
    tmp_file = os.path.join(validation_folder, f"temp_validate_{re.sub(r'[^A-Za-z0-9_]+', '_', model_name)}_{int(time.time())}.py")
    code = _sanitize(code)
    harness = adapt_code_for_testing(code, vectors_by_n)

    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(harness)

    result = {
        "model": model_name,
        "overall_success": False,
        "results_by_n": {},
        "successful_n_count": 0,
        "failed_n_count": 0
    }
    
    all_ns_keys = [str(n) for n in vectors_by_n.keys()]

    try:
        proc = subprocess.run([sys.executable, tmp_file], capture_output=True, text=True, encoding='utf-8', timeout=timeout)
        stdout = proc.stdout
        stderr = proc.stderr
        
        # Clean up the temp file immediately
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
            
        if proc.returncode != 0 and not stdout:
            # Harness script itself failed, likely an import error
            error_msg = stderr.strip() or "Execution error with no output."
            if "Failed to import" in stdout: # Our harness pipes import errors to stdout
                 error_msg = stdout.strip()
            for n_str in all_ns_keys:
                result["results_by_n"][n_str] = {"validation_passed": False, "error": error_msg}
            result["failed_n_count"] = len(all_ns_keys)
            return result
            
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        for n_str in all_ns_keys:
            result["results_by_n"][n_str] = {"validation_passed": False, "error": "Execution timed out."}
        result["failed_n_count"] = len(all_ns_keys)
        return result

    records = parse_runner_output(stdout)
    per_n = {n: {"total": 0, "passed": 0, "errors": []} for n in vectors_by_n.keys()}

    for rec in records:
        if "error" in rec and "n" not in rec:
            # This was a global error, like failing to find the function
            for n in vectors_by_n.keys():
                per_n[n]["errors"].append(rec["error"])
        else:
            n = rec.get("n")
            vec = rec.get("vec")
            moves = rec.get("moves")
            
            if n not in per_n:
                # Should not happen if vectors_by_n is correct
                per_n[n] = {"total": 0, "passed": 0, "errors": []}
                
            per_n[n]["total"] += 1
            
            try:
                if "error" in rec:
                    raise Exception(rec["error"])
                if moves is None or not isinstance(moves, list):
                    raise ValueError("neighbor_sort_moves must return a list.")
                
                # Check format for all moves *before* applying
                for mv in moves:
                    if not (isinstance(mv, (list, tuple)) and len(mv) == 2 and all(isinstance(x, int) for x in mv)):
                        raise ValueError(f"Illegal move format: {mv}. Must be pair of ints.")
                    i, j = mv
                    if not is_cyclic_neighbor_pair(i, j, len(vec)):
                        raise ValueError(f"Illegal move (not a cyclic neighbor): {mv}")
                    if not (0 <= i < len(vec) and 0 <= j < len(vec)):
                        # Allow for n=0 case
                        if len(vec) == 0 and not (i==0 and j==0):
                             raise IndexError(f"Index out of bounds: {mv} for n=0")
                        if len(vec) > 0:
                            raise IndexError(f"Index out of bounds: {mv} for n={len(vec)}")
                
                # Apply moves
                after = apply_moves(vec, [(int(i), int(j)) for i, j in moves])
                
                # Check sorted
                if after == sorted(vec):
                    per_n[n]["passed"] += 1
                else:
                    raise AssertionError(f"Final array is not sorted. Got: {after}, Expected: {sorted(vec)}")
            except Exception as e:
                per_n[n]["errors"].append(str(e))

    for n, agg in per_n.items():
        total_vectors_for_n = len(vectors_by_n[n])
        passed = agg["passed"]
        
        # Check if all *expected* vectors were run
        if agg["total"] != total_vectors_for_n:
             agg["errors"].append(f"Expected to run {total_vectors_for_n} vectors, but only got {agg['total']} results.")
        
        ok = (total_vectors_for_n > 0 and passed == total_vectors_for_n)
        
        if ok:
            result["successful_n_count"] += 1
        else:
            result["failed_n_count"] += 1
            
        result["results_by_n"][str(n)] = {
            "vectors_total": total_vectors_for_n,
            "vectors_run": agg["total"],
            "vectors_passed": passed,
            "validation_passed": ok,
            "errors": agg["errors"][:5] # Limit to first 5 errors
        }

    result["overall_success"] = (result["successful_n_count"] > 0) and (result["failed_n_count"] == 0)
    return result

def main():
    BASE_DIR = os.environ.get("RESULTS_BASE", os.getcwd())
    results_dir = os.path.join(BASE_DIR, "run_results")
    final_results_file = os.path.join(results_dir, "final_results.json")
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

    # Generate test vectors *once*
    print("Generating test vectors (n=0..9)...")
    vectors_by_n = make_test_vectors()
    all_ns_keys = [str(n) for n in vectors_by_n.keys()]

    all_validation_results = {
        "validation_timestamp": datetime.now().isoformat(),
        "source_file": final_results_file,
        "expected_behavior": get_expected_behavior(),
        "models": []
    }

    results = data.get('results', [])
    print(f"Starting validation for {len(results)} models...")

    for i, result in enumerate(results, 1):
        model_name = result.get('model', f'Unknown_{i}')
        final_code = result.get('final_code')
        
        print(f"\r({i}/{len(results)}) Validating model: {model_name.ljust(40)}", end="", flush=True)
        
        if not final_code or "def neighbor_sort_moves" not in final_code:
            model_validation = {
                "model": model_name, 
                "overall_success": False, 
                "results_by_n": {n_str: {"validation_passed": False, "vectors_total": len(vectors_by_n[int(n_str)]), "vectors_run": 0, "vectors_passed": 0, "errors": ["No valid neighbor_sort_moves() found in final_code."]} for n_str in all_ns_keys},
                "successful_n_count": 0,
                "failed_n_count": len(all_ns_keys)
            }
        else:
            model_validation = validate_model_code(final_code, model_name, vectors_by_n)
        
        all_validation_results["models"].append(model_validation)

    print(f"\nValidation complete. Saving detailed report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validation_results, f, ensure_ascii=False, indent=4)
        
    print(f"Validation report saved to {output_file}")

if __name__ == "__main__":
    main()
