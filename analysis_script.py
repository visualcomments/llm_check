import json
import pandas as pd
from datetime import datetime
import os
import sys

def ensure_output_folder(folder_name: str = "validation_output") -> str:
    """Ensures the output folder exists."""
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

    models_data = data.get('models', [])
    
    if not models_data:
        print("No model data found to analyze.")
        return
        
    df = pd.DataFrame(models_data)
    
    print("\n" + "="*70)
    print("ANALYSIS: Generic Code Validation (Unittest Execution)")
    print(f"Task: {data.get('validation_task', 'Unknown')}")
    print("="*70)
    
    total_models = len(df)
    successful_models = df['overall_success'].sum()
    success_rate = (successful_models / total_models) if total_models else 0
    
    print(f"\nOverall Success Rate: {successful_models}/{total_models} ({success_rate:.1%}) models passed all tests.")
    
    print(f"\nTop {min(10, successful_models)} Successful Models (random sample):")
    successful_df = df[df['overall_success'] == True]
    print(successful_df.head(10)[['model', 'overall_success']].to_string(index=False))

    print(f"\n{total_models - successful_models} Failed Models:")
    failed_df = df[df['overall_success'] == False]
    
    # Clean up error logs for display
    failed_df['error_summary'] = failed_df['error_log'].str.splitlines().str[-1].str.slice(0, 100)
    print(failed_df[['model', 'error_summary']].to_string(index=False))

    
    # Save reports
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models_analyzed': total_models,
            'fully_successful_models': int(successful_models),
            'overall_success_rate': success_rate,
            'task': data.get('validation_task', 'Unknown')
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