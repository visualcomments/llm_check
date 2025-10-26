## Быстрый старт

1) Генерация кода с демо-пресетом и путём результатов, ожидаемым валидатором:
```bash
python CallLLM.py --task-preset demo --results-dir /kaggle/working/run_results
```

Опционально:
- Ограничить модели:
```bash
python CallLLM.py --task-preset demo --results-dir /kaggle/working/run_results --models "gpt-4o-mini,gpt-3.5-turbo"
```
- Запустить локальную HF-модель:
```bash
python CallLLM.py --task-preset demo --results-dir /kaggle/working/run_results --hf_model microsoft/Phi-3-mini-4k-instruct
```

2) Валидация результатов:
```bash
python validation_script.py
```

3) Аналитика:
```bash
python analysis_script.py
```

## Файлы
- `CallLLM.py` — обновлённый оркестратор с пресетами (`--task-preset demo|default`), `--task-file`, `--results-dir`.
- `validation_script.py` — валидатор результатов.
- `analysis_script.py` — анализатор отчёта валидации.

## Требования
- Python 3.10+
- `pip install -r requirements.txt`

> Примечание: `transformers`/`torch` нужны только если вы используете `--hf_model`.
