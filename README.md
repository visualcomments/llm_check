# LLM Code Generation & Validation Pipeline

This repository packages a **repeatable pipeline** to prompt Large Language Models (LLMs) to generate Python code, run lightweight self-checks, and then **validate** and **analyze** those results with dedicated scripts.

The repository is intentionally minimal and self-contained:
- **`CallLLM.py`** – Orchestrates LLM code generation (AnyProvider via `g4f`, or an optional local Hugging Face model), auto-fixes/refactors code, and saves a canonical results JSON.
- **`validation_script.py`** – Executes a robust validation harness against the generated code and exports a detailed report.
- **`analysis_script.py`** – Summarizes validation results into a machine- and human-friendly analysis (JSON + CSV).
- **`task_demo.txt`** – A simple example task file you can pass with `--task-file`.
- **`requirements.txt`** – Minimal runtime dependencies.
- **This `README.md`** – You are here.


---

## 1) Repository Structure

```
.
├── CallLLM.py                 # LLM orchestration and codegen pipeline
├── validation_script.py       # Validates generated code on curated test vectors
├── analysis_script.py         # Aggregates & analyzes validation results
├── task_demo.txt              # Simple English task for --task-file
├── requirements.txt           # Python dependencies
└── README.md                  # This guide
```

---

## 2) Quick Start

### Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate code (demo task) and place results where the validator expects them
```bash
python CallLLM.py --task-preset demo --results-dir /kaggle/working/run_results
```

### Validate
```bash
python validation_script.py
```

### Analyze
```bash
python analysis_script.py
```

All outputs are written beneath `/kaggle/working/` to simplify reproducibility (this path is used both locally and on common hosted notebooks).

---

## 3) `CallLLM.py` – Orchestrator

### What it does
1. **Task building** – chooses a task description from a preset or a user-provided file.
2. **Model selection** –
   - If `--hf_model` is provided, uses a local Hugging Face model (optional feature).
   - Otherwise queries `g4f` for available text models and intersects with an optional allowlist (`--models`).
3. **Prompting** – sends a strict instruction asking the LLM to return **Python code only**, defining
   ```python
   def neighbor_sort_moves(vec: list) -> list
   ```
   The function must return **only** a list of **cyclic-adjacent** swap pairs (either `(i, i+1)` or `(n-1, 0)`).
4. **Self-check + repair loop** –
   - Extracts a code block from the response.
   - Runs a lightweight syntax/runtime check in a sandboxed subprocess.
   - If a failure occurs, asks the model to **FIX** the code (with the captured error).
   - Optionally **REFACTORs** to improve readability/robustness.
5. **Outputs** – saves a single `final_results.json` with all models’ attempts and the final code per model to the directory you configured via `--results-dir`.

### CLI Options
| Option | Type | Default | Description |
|---|---|---:|---|
| `--task-preset` | `default`\|`demo` | `default` | Built-in task text. `demo` is intentionally the **easiest** for LLMs. |
| `--task-file` | `str` | `None` | Path to a custom task file (overrides `--task-preset`). |
| `--results-dir` | `str` | `/kaggle/working/run_results` | Where to write `final_results.json` (must exist or be creatable). |
| `--hf_model` | `str` | `None` | (Optional) Local HF model id/path; requires `transformers` and `torch`. |
| `--models` | `str` | `None` | Comma-separated allowlist of `g4f` model names to test. |
| `--max_workers` | `int` | `50` | Parallel worker count for multi-model runs. |

### Task Sources
- **Preset** (English, strict but short):  
  - `default`: classic “adjacent + wrap-around” sorting problem.  
  - `demo`: **simplified** cyclic bubble-like process (recommended for quick smoke tests).
- **Custom file:** `--task-file /path/to/task.txt` (see included `task_demo.txt`).

### Internals (Important Functions)
- **`extract_python_code(raw: str) -> str`** – Tolerant extractor for JSON `{ "answer": "..." }` or markdown code blocks. Strips HTML fragments and optionally trims to `def neighbor_sort_moves`.
- **`hf_local_query(model_id, prompt, ...)`** – Optional local inference via `transformers` (no internet).
- **`llm_query(model, prompt, ...)`** – Unified query against local HF or `g4f.Provider.AnyProvider`, with retry and a **provider-rotation patch** (`TrackedRotated`) to log which providers succeeded/failed.
- **`safe_execute(code, config)`** – Runs code in a subprocess with a timeout to catch obvious syntax/runtime errors.
- **`process_model(model, task, config, progress_queue)`** – Full per-model pipeline: initial → fix-if-fail → refactor (loop).
- **`orchestrator(task, models, config, progress_queue)`** – Parallel execution across models; writes `final_results.json` with a timestamp.
- **`build_default_task_text()` / `build_demo_task_text()`** – Built-in task bodies.
- **`parse_args() / main()`** – CLI plumbing.

### Output: `final_results.json` (schema sketch)
```jsonc
{
  "timestamp": "2025-10-26T12:34:56.789012",
  "results": [
    {
      "model": "gpt-4o-mini",
      "iterations": [
        {
          "stage": "initial_response",
          "providers_tried": ["ProviderA", "ProviderB", "..."],
          "success_provider": "ProviderB",
          "response": "Python code ...",
          "error": null
        },
        {
          "stage": "refactor_response",
          "response": "Python code ...",
          "error": null
        }
        // ... more iterations
      ],
      "final_code": "def neighbor_sort_moves(vec: list) -> list: ..."
    }
    // ... one object per model
  ]
}
```

> **Tip:** If you want a single model run, use `--models "model_name"` or provide `--hf_model`.

---

## 4) `validation_script.py` – What it Validates

**Inputs**
- Reads `final_results.json` from **`/kaggle/working/run_results/final_results.json`** (or wherever you pointed `--results-dir` when running `CallLLM.py`).

**Process**
1. Builds a **test harness** around each model’s `final_code` (function `adapt_code_for_testing`).
2. Imports the candidate’s `neighbor_sort_moves` safely while suppressing side effects.
3. Feeds curated **test vectors grouped by `n`** (different input sizes, duplicates included).
4. Checks two invariants per vector:
   - **Correctness:** applying the returned swap sequence to a working copy yields a **nondecreasing** list.
   - **Legality:** **every** swap is **cyclic-adjacent** only (`(i, i+1)` or `(n-1, 0)`), with wrap-around allowed.
5. Aggregates per-`n` stats and computes an `overall_success` flag.

**Outputs**
- Writes a comprehensive JSON report to **`/kaggle/working/validation_output/comprehensive_validation.json`** with, for each model:
  - `results_by_n`: totals, passes, and sample errors
  - `successful_n_count`, `failed_n_count`, `overall_success`
  - metadata for traceability (e.g., temporary harness filename)
- Prints a short console summary.

---

## 5) `analysis_script.py` – What it Produces

**Inputs**
- Expects **`/kaggle/working/validation_output/comprehensive_validation.json`**.

**Outputs**
- **`/kaggle/working/validation_output/analysis_report.json`** – high-level KPIs (per-model success, pass rates by `n`, etc.).
- **`/kaggle/working/validation_output/analysis_data.csv`** – flat table (one row per model), suitable for spreadsheets/dashboards.
- Console logs with file paths.

---

## 6) Advanced Usage Examples

### Run a specific set of `g4f` models
```bash
python CallLLM.py   --task-preset demo   --results-dir /kaggle/working/run_results   --models "gpt-4o-mini,gpt-3.5-turbo"
```

### Use a local Hugging Face model
```bash
python CallLLM.py   --task-file ./task_demo.txt   --results-dir /kaggle/working/run_results   --hf_model microsoft/Phi-3-mini-4k-instruct
```
> Local mode requires GPU-friendly `torch` + `transformers`. CPU works but is slower.

### Provide your own task
```bash
python CallLLM.py --task-file ./my_task.txt --results-dir /kaggle/working/run_results
```

---

## 7) Troubleshooting

- **No models found**: Provide `--models` or `--hf_model`; ensure `g4f` is installed and accessible.
- **Timeouts during self-check**: Increase `EXEC_TIMEOUT` inside `CallLLM.py`’s `CONFIG['CONSTANTS']`, or simplify the task.
- **Validation can’t find results**: Make sure `final_results.json` is at the expected path. Re-run codegen with `--results-dir /kaggle/working/run_results`.
- **Import errors in validation**: The harness captures and reports the exception; generally indicates the generated module didn’t define `neighbor_sort_moves` or had syntax errors.
- **Large runs**: Tune parallelism with `--max_workers`.

---

## 8) Requirements

```
requests
g4f
# Optional (only for --hf_model):
transformers
torch
```

Python **3.10+** is recommended.

---

## 9) License

This repository is provided **as-is** for experimentation and evaluation of LLM-based code generation workflows.
