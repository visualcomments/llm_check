import json
import pandas as pd
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Новые зависимости ---
# Пожалуйста, установите matplotlib и seaborn:
# pip install matplotlib seaborn
# -------------------------

# --- КОНФИГУРАЦИЯ ---
VALIDATION_OUTPUT_FOLDER = "./validation_output"
TOP_N_ERRORS = 10 # Количество самых частых ошибок для отображения

def ensure_output_folder(folder_name: str = VALIDATION_OUTPUT_FOLDER) -> str:
    """Гарантирует, что папка для вывода существует."""
    path = os.path.abspath(os.path.join(os.getcwd(), folder_name))
    os.makedirs(path, exist_ok=True)
    return path

def main():
    output_folder = ensure_output_folder()
    validation_file = os.path.join(output_folder, "comprehensive_validation.json")
    
    if not os.path.exists(validation_file):
        print(f"Ошибка: Файл валидации не найден по пути '{validation_file}'", file=sys.stderr)
        print("Пожалуйста, сначала запустите validation_script для генерации результатов.", file=sys.stderr)
        return

    print(f"Загрузка данных валидации из {validation_file}...")
    try:
        with open(validation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка: Не удалось прочитать JSON файл. Проверьте его формат. {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при чтении файла: {e}", file=sys.stderr)
        return

    models_data = data.get('models', [])
    
    if not models_data:
        print("Данные моделей для анализа не найдены.")
        return
        
    df = pd.DataFrame(models_data)
    
    # --- Сводная статистика ---
    print("\n" + "="*70)
    print("АНАЛИЗ: Валидация (Unittest Execution)")
    print(f"Задача: {data.get('validation_task', 'Unknown')}")
    print("="*70)
    
    total_models = len(df)
    successful_models = df['overall_success'].sum()
    success_rate = (successful_models / total_models) if total_models else 0
    
    print(f"\nОбщая успешность: {successful_models}/{total_models} ({success_rate:.1%}) моделей прошли все тесты.")
    
    print(f"\nТоп {min(10, int(successful_models))} успешных моделей (первые 10):")
    successful_df = df[df['overall_success'] == True]
    print(successful_df.head(10)[['model', 'overall_success']].to_string(index=False))

    # --- УЛУЧШЕНИЕ: Анализ ошибок ---
    print("\n" + "="*70)
    print(f"Анализ {total_models - successful_models} проваленных моделей")
    print("="*70)
    failed_df = df[df['overall_success'] == False].copy()
    error_counts = pd.Series(dtype=int)

    if not failed_df.empty:
        # Ваш фикс для получения сводки ошибок - он хороший и здесь сохранен
        failed_df['error_summary'] = failed_df['error_log'].astype(str).apply(str.splitlines).str[-1].str.slice(0, 100)
        error_counts = failed_df['error_summary'].value_counts().head(TOP_N_ERRORS)
        
        print(f"\nТоп-{len(error_counts)} самых частых ошибок:")
        print(error_counts.to_string(name="Count"))
        print(f"\nПолный список ошибок см. в 'analysis_data.csv'")
    else:
        print("\nВсе модели прошли тесты. Ошибок нет. 🎉")

    # --- УЛУЧШЕНИЕ: Анализ по каждому тесту (если данные доступны) ---
    print("\n" + "="*70)
    print("Анализ по отдельным тест-кейсам")
    print("="*70)
    test_success_rates = pd.Series(dtype=float)
    
    try:
        # Проверяем, есть ли у нас вложенные данные по тестам
        if 'test_results' not in df.columns or df['test_results'].isnull().all():
            print("В JSON нет детальных 'test_results'. Пропускаем анализ по тестам.")
        else:
            # "Разворачиваем" вложенные данные: одна строка = один тест одной модели
            test_df = pd.json_normalize(models_data, record_path='test_results', meta=['model'])
            
            if not test_df.empty:
                # Считаем процент успеха для каждого теста
                test_success_rates = test_df.groupby('test_name')['success'].mean().sort_values(ascending=False)
                print("\nПроцент успеха по каждому тест-кейсу:")
                print(test_success_rates.to_string(float_format='{:.1%}'.format))
            else:
                print("Список 'test_results' пуст. Нет данных для анализа по тестам.")

    except KeyError:
        print("Ошибка: 'test_results' найден, но имеет неожиданную структуру (напр., отсутствует 'test_name' или 'success').")
    except Exception as e:
        print(f"Неожиданная ошибка при анализе 'test_results': {e}")


    # --- УЛУЧШЕНИЕ: Генерация Визуализаций ---
    print("\n" + "="*70)
    print("Генерация графиков...")
    print("="*70)
    
    sns.set_theme(style="whitegrid")

    # 1. График: Общая успешность (Pie Chart)
    try:
        success_counts = df['overall_success'].value_counts().sort_index()
        labels = ['Failed', 'Passed'] if len(success_counts) == 2 else (['Passed'] if success_counts.index[0] else ['Failed'])
        colors = ['#FF6B6B', '#6BFF6B'] if len(success_counts) == 2 else (['#6BFF6B'] if success_counts.index[0] else ['#FF6B6B'])

        plt.figure(figsize=(8, 6))
        plt.pie(success_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Общая успешность моделей')
        plot_path = os.path.join(output_folder, 'plot_overall_success.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  [+] График общей успешности сохранен: {plot_path}")
    except Exception as e:
        print(f"  [!] Не удалось создать график общей успешности: {e}")

    # 2. График: Успешность по тестам (Bar Chart)
    if not test_success_rates.empty:
        try:
            plt.figure(figsize=(10, max(6, len(test_success_rates) * 0.5))) # Динамическая высота
            sns.barplot(x=test_success_rates.values * 100, y=test_success_rates.index, orient='h', palette="viridis")
            plt.title('Процент успеха по тест-кейсам')
            plt.xlabel('Успешность (%)')
            plt.ylabel('Название теста')
            plt.xlim(0, 100)
            plt.tight_layout()
            plot_path = os.path.join(output_folder, 'plot_test_success_rates.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"  [+] График по тестам сохранен: {plot_path}")
        except Exception as e:
            print(f"  [!] Не удалось создать график по тестам: {e}")

    # 3. График: Топ ошибок (Bar Chart)
    if not error_counts.empty:
        try:
            plt.figure(figsize=(10, max(6, len(error_counts) * 0.6))) # Динамическая высота
            sns.barplot(x=error_counts.values, y=error_counts.index, orient='h', palette="rocket")
            plt.title(f'Топ-{len(error_counts)} самых частых ошибок')
            plt.xlabel('Количество')
            plt.ylabel('Сводка по ошибке (усечено)')
            plt.tight_layout()
            plot_path = os.path.join(output_folder, 'plot_common_errors.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"  [+] График по ошибкам сохранен: {plot_path}")
        except Exception as e:
            print(f"  [!] Не удалось создать график по ошибкам: {e}")

    
    # --- Сохранение отчетов ---
    print("\n" + "="*70)
    print("Сохранение отчетов...")
    print("="*70)
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models_analyzed': total_models,
            'fully_successful_models': int(successful_models),
            'failed_models': total_models - int(successful_models),
            'overall_success_rate': success_rate,
            'task': data.get('validation_task', 'Unknown'),
            'top_errors': error_counts.to_dict(), # Добавляем новые данные
            'test_case_success_rates': test_success_rates.to_dict() # Добавляем новые данные
        },
        'all_model_results': models_data
    }
    
    report_file = os.path.join(output_folder, 'analysis_report.json')
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print(f"JSON отчет сохранен в {report_file}")
    except Exception as e:
        print(f"Ошибка сохранения JSON отчета: {e}", file=sys.stderr)
    
    csv_file = os.path.join(output_folder, 'analysis_data.csv')
    try:
        # Убираем сложные колонки (как 'test_results') перед сохранением в CSV для лучшей читаемости
        df_to_csv = df.drop(columns=['test_results'], errors='ignore')
        df_to_csv.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV сохранен в {csv_file}")
    except Exception as e:
        print(f"Ошибка сохранения CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
