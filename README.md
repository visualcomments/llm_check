# Neighbor-Transposition Validation — Fixed

**Date:** 2025-10-25

## What changed

1. **Response normalization** in `CallLLM.py`:
   - Added `extract_python_code()` to strip Markdown/HTML/JSON shells.
   - All `current_code = response` → `current_code = extract_python_code(response)`.

2. **Defensive sanitation** in `validation_script.py`:
   - Added `_sanitize()` and applied before building the test harness.
   - Harness exits with non-zero code on import failure.
   - Portable paths via `RESULTS_BASE` env var (fallback to `os.getcwd()`).

3. **Portable output paths** in `validation_script.py` & `analysis_script.py`:
   - Removed hardcoded `/kaggle/working` usage.

## How to run

```bash
# optional: set a base for results
export RESULTS_BASE="$(pwd)"

# run validation (expects run_results/final_results.json to exist)
python validation_script.py

# run analysis
python analysis_script.py
```
