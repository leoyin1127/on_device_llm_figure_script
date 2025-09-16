"""
Script to calculate exact match accuracy for model predictions against ground truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV data and filter to valid cases only."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Filter to valid case data only (exclude summary/empty rows)
        valid_mask = df['case_id'].notna()
        df_clean = df[valid_mask].copy()
        
        invalid_rows = len(df) - len(df_clean)
        if invalid_rows > 0:
            print(f"Filtered out {invalid_rows} invalid/summary rows")
        
        print(f"Using {len(df_clean)} valid cases for analysis")
        
        # Also show Excel formula range for comparison
        excel_range = df.iloc[1:210]  # Excel rows 2-210
        excel_valid = excel_range['case_id'].notna().sum()
        print(f"Note: Excel formula range (rows 2-210) would use {excel_valid} cases")
        
        return df_clean
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise


def get_model_columns(df: pd.DataFrame) -> List[str]:
    """Extract model prediction column names (excluding metadata columns)."""
    # Skip the first 7 columns which are metadata
    # (case_id, Section, OriginalDescription, PostDescription, 
    #  DifferentialDiagnosisList, FinalDiagnosis, DiseaseLeak)
    model_columns = df.columns[7:].tolist()
    return model_columns


def calculate_exact_match_accuracy(predictions: pd.Series, ground_truth: pd.Series, 
                                 include_empty_predictions: bool = True) -> Tuple[float, int, int]:
    """
    Calculate exact match accuracy between predictions and ground truth.
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        include_empty_predictions: If True, count empty predictions as incorrect.
                                 If False, exclude cases with empty predictions.
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    if include_empty_predictions:
        # Include all cases where ground truth exists, treat empty predictions as wrong
        mask = ~ground_truth.isna()
        clean_ground_truth = ground_truth[mask]
        clean_predictions = predictions[mask]
        
        if len(clean_ground_truth) == 0:
            return 0.0, 0, 0
        
        # Replace NaN predictions with empty string for comparison
        clean_predictions = clean_predictions.fillna('')
        
        # Calculate exact matches (case-sensitive)
        exact_matches = (clean_predictions.astype(str).str.strip() == 
                        clean_ground_truth.astype(str).str.strip())
        
        correct_count = exact_matches.sum()
        total_count = len(clean_ground_truth)
        
    else:
        # Exclude cases where either prediction or ground truth is NaN
        mask = ~(predictions.isna() | ground_truth.isna())
        clean_predictions = predictions[mask]
        clean_ground_truth = ground_truth[mask]
        
        if len(clean_predictions) == 0:
            return 0.0, 0, 0
        
        # Calculate exact matches (case-sensitive)
        exact_matches = (clean_predictions.astype(str).str.strip() == 
                        clean_ground_truth.astype(str).str.strip())
        
        correct_count = exact_matches.sum()
        total_count = len(clean_predictions)
    
    accuracy = correct_count / total_count * 100
    return accuracy, correct_count, total_count


def calculate_all_accuracies(df: pd.DataFrame, ground_truth_col: str = 'FinalDiagnosis', 
                           include_empty_predictions: bool = True) -> Dict[str, Dict]:
    """
    Calculate accuracy for all model columns.
    
    Args:
        df: DataFrame containing predictions and ground truth
        ground_truth_col: Name of the ground truth column
        include_empty_predictions: If True, count empty predictions as incorrect
        
    Returns:
        Dictionary with model names as keys and accuracy metrics as values
    """
    if ground_truth_col not in df.columns:
        raise ValueError(f"Ground truth column '{ground_truth_col}' not found in dataset")
    
    ground_truth = df[ground_truth_col]
    model_columns = get_model_columns(df)
    
    results = {}
    
    for model_col in model_columns:
        if model_col in df.columns:
            accuracy, correct, total = calculate_exact_match_accuracy(
                df[model_col], ground_truth, include_empty_predictions
            )
            
            results[model_col] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'accuracy_str': f"{accuracy:.2f}%"
            }
        else:
            print(f"Warning: Column '{model_col}' not found in dataset")
    
    return results


def compare_with_excel_range(original_df: pd.DataFrame, ground_truth_col: str = 'FinalDiagnosis', 
                           include_empty_predictions: bool = True) -> None:
    """Compare our calculation with Excel formula range (rows 2-210)."""
    print("\n" + "="*80)
    print("COMPARISON WITH EXCEL FORMULA RANGE")
    print("="*80)
    
    # Excel range: rows 2-210 (pandas 1:210)
    excel_df = original_df.iloc[1:210]
    
    # First model for comparison
    first_model = get_model_columns(original_df)[0]
    
    # Our approach (all valid cases)
    valid_mask = original_df['case_id'].notna()
    our_df = original_df[valid_mask]
    our_acc, our_correct, our_total = calculate_exact_match_accuracy(
        our_df[first_model], our_df[ground_truth_col], include_empty_predictions
    )
    
    # Excel approach
    excel_acc, excel_correct, excel_total = calculate_exact_match_accuracy(
        excel_df[first_model], excel_df[ground_truth_col], include_empty_predictions
    )
    
    print(f"Model: {first_model}")
    print("-" * 60)
    print(f"Our calculation (all valid cases):     {our_acc:6.2f}% ({our_correct:3}/{our_total:3})")
    print(f"Excel range (rows 2-210):             {excel_acc:6.2f}% ({excel_correct:3}/{excel_total:3})")
    print(f"Difference:                           {our_acc - excel_acc:+6.2f}%")
    
    # Show what cases are different
    our_cases = set(our_df[our_df['case_id'].notna()]['case_id'].astype(int))
    excel_valid = excel_df['case_id'].notna()
    excel_cases = set(excel_df[excel_valid]['case_id'].astype(int))
    
    missing_in_excel = our_cases - excel_cases
    extra_in_excel = excel_cases - our_cases
    
    if missing_in_excel:
        print(f"Cases missing from Excel range: {sorted(missing_in_excel)}")
    if extra_in_excel:
        print(f"Extra cases in Excel range: {sorted(extra_in_excel)}")
    
    evaluation_method = "including" if include_empty_predictions else "excluding"
    print(f"\nEvaluation method: {evaluation_method} empty predictions")
    if include_empty_predictions:
        print("RECOMMENDATION: This method (including empty predictions as incorrect) is better for model evaluation.")


def group_by_model_family(results: Dict[str, Dict]) -> Dict[str, List[Tuple[str, Dict]]]:
    """Group results by model family for better organization."""
    families = {}
    
    for model_name, metrics in results.items():
        # Extract base model name (everything before version indicators)
        base_name = model_name.split(' v')[0].split(' (')[0]
        
        if base_name not in families:
            families[base_name] = []
        
        families[base_name].append((model_name, metrics))
    
    # Sort within each family
    for family in families:
        families[family].sort(key=lambda x: x[0])
    
    return families


def print_results(results: Dict[str, Dict]):
    """Print formatted results."""
    print("\n" + "="*80)
    print("EXACT MATCH ACCURACY RESULTS")
    print("="*80)
    
    # Group by model family
    families = group_by_model_family(results)
    
    for family_name, family_results in families.items():
        print(f"\n{family_name.upper()}:")
        print("-" * 60)
        
        for model_name, metrics in family_results:
            print(f"{model_name:<40} {metrics['accuracy']:>6.2f}% "
                  f"({metrics['correct']:>4}/{metrics['total']:<4})")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    accuracies = [metrics['accuracy'] for metrics in results.values()]
    print(f"Number of models evaluated: {len(accuracies)}")
    print(f"Mean accuracy: {np.mean(accuracies):.2f}%")
    print(f"Median accuracy: {np.median(accuracies):.2f}%")
    print(f"Standard deviation: {np.std(accuracies):.2f}%")
    print(f"Min accuracy: {np.min(accuracies):.2f}%")
    print(f"Max accuracy: {np.max(accuracies):.2f}%")


def save_results_to_csv(results: Dict[str, Dict], output_path: str):
    """Save results to a CSV file."""
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'model'
    results_df = results_df.reset_index()
    results_df = results_df[['model', 'accuracy', 'correct', 'total']]
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calculate exact match accuracy for model predictions')
    parser.add_argument('--input', '-i', type=str, 
                       default='data/OSS Benchmarking Results - Eurorad.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str,
                       default='output/accuracy_results.csv', 
                       help='Output CSV file path for results')
    parser.add_argument('--gt-column', type=str, default='FinalDiagnosis',
                       help='Ground truth column name')
    parser.add_argument('--exclude-empty', action='store_true',
                       help='Exclude cases with empty predictions from evaluation (default: include empty predictions as incorrect)')
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path
    
    # Ensure parent directory exists for input
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load original data for comparison
    original_df = pd.read_csv(csv_path)
    
    # Load cleaned data for analysis  
    df = load_data(csv_path)
    
    include_empty_predictions = not args.exclude_empty
    
    # Print evaluation method
    if include_empty_predictions:
        print("Evaluation method: Including empty predictions as incorrect (recommended for model evaluation)")
    else:
        print("Evaluation method: Excluding cases with empty predictions")
    
    # Show comparison with Excel formula range
    compare_with_excel_range(original_df, args.gt_column, include_empty_predictions)
    
    # Calculate accuracies
    print(f"\nCalculating exact match accuracies against '{args.gt_column}' column...")
    results = calculate_all_accuracies(df, args.gt_column, include_empty_predictions)
    
    # Display results
    print_results(results)
    
    # Save results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_results_to_csv(results, output_path)


if __name__ == "__main__":
    main()
