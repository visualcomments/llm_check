import json
import pandas as pd
from datetime import datetime
import os
import sys

def ensure_output_folder(folder_name: str = "validation_output") -> str:
    path = os.path.join("/kaggle/working/", folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def main():
    output_folder = ensure_output_folder()
    validation_file = os.path.join(output_folder, "comprehensive_validation.json")
    
    if not os.path.exists(validation_file):
        print(f"Error: The validation file was not found at '{validation_file}'", file=sys.stderr)
        print("Please run the validation_script first to generate the validation results.", file=sys.stderr)
        return

    print(f"Loading validation data from {validation_file}...")
    with open(validation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models_data = []
    n_values = []
    
    if data.get('models'):
        # Find all 'n' values tested by inspecting the first model result
        sample = data['models'][0].get('results_by_n', {})
        n_values = sorted(int(k) for k in sample.keys())

    print(f"Found test data for n values: {n_values}")

    for model_results in data.get('models', []):
        if not model_results.get('results_by_n'):
            continue
        
        total_ns = len(model_results['results_by_n'])
        succ_ns = sum(1 for n in model_results['results_by_n'].values() if n.get('validation_passed'))
        rate = succ_ns / total_ns if total_ns else 0
        
        model_info = {
            'model': model_results['model'],
            'overall_success': model_results['overall_success'],
            'successful_n_count': succ_ns,
            'total_n_tested': total_ns,
            'success_rate': round(rate, 3)
        }
        
        # Aggregate stats per 'n'
        for n in n_values:
            n_str = str(n)
            n_result = model_results['results_by_n'].get(n_str, {})
            model_info[f'n_{n}_passed'] = n_result.get('validation_passed', False)
            model_info[f'n_{n}_vectors_total'] = n_result.get('vectors_total', 0)
            model_info[f'n_{n}_vectors_ok'] = n_result.get('vectors_passed', 0)
            model_info[f'n_{n}_vectors_run'] = n_result.get('vectors_run', 0)
        
        models_data.append(model_info)
    
    if not models_data:
        print("No model data found to analyze.")
        return
        
    df = pd.DataFrame(models_data)
    
    print("\n" + "="*70)
    print("ANALYSIS: Sorting-by-Neighbor-Transpositions Task")
    print("="*70)
    
    total_models = len(df)
    successful_models = df['overall_success'].sum()
    print(f"\nOverall Success Rate: {successful_models}/{total_models} ({(successful_models/total_models if total_models else 0):.1%}) models passed all n.")
    
    print("\nPer-n pass rates (percent of models that passed all vectors for n):")
    for n in n_values:
        col = f'n_{n}_passed'
        if col in df.columns:
            rate = df[col].mean()
            print(f"  n={n}: {rate:.1%}")
    
    print("\nTop 10 Models by success_rate (passed all vectors for the most n's):")
    top = df.nlargest(10, 'success_rate')[['model', 'success_rate', 'successful_n_count', 'total_n_tested']]
    print(top.to_string(index=False))
    
    print("\nModels that failed all tests (success_rate == 0):")
    failed = df[df['success_rate'] == 0]
    print(f"  {len(failed)}/{total_models} models failed all n values.")
    if not failed.empty and len(failed) < 20:
        print("\n".join(failed['model'].tolist()))
    
    # Save reports
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models_analyzed': total_models,
            'fully_successful_models': int(successful_models),
            'overall_success_rate': (successful_models/total_models if total_models else 0),
            'n_values_tested': n_values
        },
        'all_model_results': models_data
    }
    report_file = os.path.join(output_folder, 'analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"\nJSON report saved to {report_file}")
    
    csv_file = os.path.join(output_folder, 'analysis_data.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV saved to {csv_file}")

if __name__ == "__main__":
    main()
