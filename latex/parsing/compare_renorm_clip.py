#!/usr/bin/env python3
"""
Script to generate LaTeX tables comparing different renorm and clip settings.
For each task, creates a table showing all combinations of renorm/clip settings for each method.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_generators import load_single_experiment_data
from latex_utils import LaTeXTableBuilder, format_method_name, format_value, format_task_name


def load_experiment_data_with_settings(base_path, experiment_name):
    """
    Load data from all 4 experiment variants (renorm/clip combinations).
    
    Args:
        base_path: Base path where experiments are located
        experiment_name: Base experiment name (e.g., "midsteer_sa_50k_last")
        
    Returns:
        Dictionary mapping (renorm, clip) -> DataFrame
    """
    results = {}
    
    # Define the 4 combinations
    combinations = [
        (True, True, "renorm_clip"),
        (False, True, "no_renorm_clip"),
        (True, False, "renorm_no_clip"),
        (False, False, "no_renorm_no_clip")
    ]
    
    for renorm, clip, suffix in combinations:
        exp_path = base_path / f"{experiment_name}_{suffix}"
        
        if exp_path.exists():
            try:
                df = load_single_experiment_data(str(exp_path))
                if not df.empty:
                    df['Renorm'] = renorm
                    df['Clip'] = clip
                    results[(renorm, clip)] = df
                    print(f"Loaded {len(df)} rows from {exp_path.name}")
                else:
                    print(f"No data found in {exp_path.name}")
            except Exception as e:
                print(f"Error loading {exp_path.name}: {e}")
        else:
            print(f"Warning: {exp_path.name} not found")
    
    return results


def select_beta_for_method(method, beta_values):
    """
    Select appropriate beta value for a given method.
    
    Args:
        method: Method name
        beta_values: Available beta values for this method
        
    Returns:
        Selected beta value
    """
    # Default beta selection rules
    if method == 'mean_matching':
        # Prefer 2.0, but fall back to closest available
        if 2.0 in beta_values:
            return 2.0
        else:
            return min(beta_values, key=lambda x: abs(x - 2.0))
    elif method == 'None':
        return 0.0
    else:
        # For other methods, use the middle value or a reasonable default
        beta_list = sorted(beta_values)
        if len(beta_list) > 1:
            return beta_list[len(beta_list) // 2]
        else:
            return beta_list[0]


def generate_comparison_table(experiment_data, task, output_file, beta_selection=None, enable_highlighting=True):
    """
    Generate a comparison table for a specific task.
    
    Args:
        experiment_data: Dictionary mapping (renorm, clip) -> DataFrame
        task: Task name to generate table for
        output_file: Output LaTeX file path
        beta_selection: Dictionary mapping method -> beta value (optional)
        enable_highlighting: Whether to apply bold/underline highlighting
        
    Returns:
        LaTeX table content as string
    """
    # Combine all data for this task
    combined_data = []
    
    for (renorm, clip), df in experiment_data.items():
        task_df = df[df['Task'] == task].copy()
        if not task_df.empty:
            combined_data.append(task_df)
    
    if not combined_data:
        print(f"No data found for task: {task}")
        return ""
    
    # Combine all dataframes
    all_df = pd.concat(combined_data, ignore_index=True)
    
    # Get unique methods
    methods = sorted([m for m in all_df['Method'].unique() if m != 'None'])
    if 'None' in all_df['Method'].unique():
        methods = ['None'] + methods
    
    # For each method, determine the beta value to use
    if beta_selection is None:
        beta_selection = {}
    
    # Ensure all methods have beta values selected
    for method in methods:
        if method not in beta_selection:
            method_df = all_df[all_df['Method'] == method]
            available_betas = set(method_df['Beta'].unique())
            beta_selection[method] = select_beta_for_method(method, available_betas)
    
    # Filter data to only include selected beta values
    filtered_data = []
    for method in methods:
        method_df = all_df[all_df['Method'] == method]
        if method in beta_selection:
            beta_df = method_df[method_df['Beta'] == beta_selection[method]]
            if not beta_df.empty:
                filtered_data.append(beta_df)
    
    if not filtered_data:
        print(f"No data found for selected beta values in task: {task}")
        return ""
    
    # Combine filtered data
    final_df = pd.concat(filtered_data, ignore_index=True)
    
    # Create table
    # Columns: Method, Beta, Renorm, Clip, Target_Concept_Score, Source_Concept_Score, Alpaca_BLEU, Alpaca_BertF1, MMLU_BLEU, MMLU_BertF1
    col_spec = "l|c|c|c|cccccc|"
    
    task_name_formatted = format_task_name(task)
    task_name_escaped = task_name_formatted.replace("→", "$\\rightarrow$") if task_name_formatted else task
    
    table = LaTeXTableBuilder(
        caption=f"Renorm/Clip Comparison - {task_name_escaped}",
        label=f"tab:renorm_clip_{task.replace('_', '')}",
        col_spec=col_spec
    )
    
    # Header rows
    header1_cells = ["Method", "$\\beta$", "Renorm", "Clip", "\\multicolumn{6}{c|}{Metrics}"]
    table.add_header_row(header1_cells)
    
    header2_cells = ["", "", "", "", "SCS", "TCS", "A-BLEU", "A-BertF1", "M-BLEU", "M-BertF1"]
    table.add_header_row(header2_cells)
    table.add_hline()
    
    # Prepare data for ranking
    all_values = {
        'scs': [], 'tcs': [], 'ab': [], 'af1': [], 'mb': [], 'mf1': []
    }
    
    # Collect all values for ranking
    for method in methods:
        method_df = final_df[final_df['Method'] == method]
        for (renorm, clip) in [(True, True), (False, True), (True, False), (False, False)]:
            row_df = method_df[(method_df['Renorm'] == renorm) & (method_df['Clip'] == clip)]
            if not row_df.empty:
                row = row_df.iloc[0]
                all_values['scs'].append((row['Source_Concept_Score'], format_value(row['Source_Concept_Score'], 2)))
                all_values['tcs'].append((row['Target_Concept_Score'], format_value(row['Target_Concept_Score'], 2)))
                all_values['ab'].append((row['Alpaca_BLEU'], format_value(row['Alpaca_BLEU'], 3)))
                all_values['af1'].append((row['Alpaca_BertF1'], format_value(row['Alpaca_BertF1'], 3)))
                all_values['mb'].append((row['MMLU_BLEU'], format_value(row['MMLU_BLEU'], 3)))
                all_values['mf1'].append((row['MMLU_BertF1'], format_value(row['MMLU_BertF1'], 3)))
            else:
                # No data for this combination
                all_values['scs'].append((None, "—"))
                all_values['tcs'].append((None, "—"))
                all_values['ab'].append((None, "—"))
                all_values['af1'].append((None, "—"))
                all_values['mb'].append((None, "—"))
                all_values['mf1'].append((None, "—"))
    
    # Apply ranking formatting
    if enable_highlighting:
        from table_generators import format_value_with_ranking
        formatted_values = {}
        for key in all_values.keys():
            formatted_values[key] = format_value_with_ranking(all_values[key])
    else:
        formatted_values = {}
        for key in all_values.keys():
            formatted_values[key] = [orig_str for _, orig_str in all_values[key]]
    
    # Generate table rows
    row_index = 0
    for method_idx, method in enumerate(methods):
        method_df = final_df[final_df['Method'] == method]
        method_name = format_method_name(method)
        beta_str = f"{beta_selection[method]:.1f}" if method != 'None' else "—"
        
        # Generate 4 rows for this method (all combinations)
        for config_idx, (renorm, clip) in enumerate([(True, True), (False, True), (True, False), (False, False)]):
            row_df = method_df[(method_df['Renorm'] == renorm) & (method_df['Clip'] == clip)]
            
            # Method name and beta only on first row
            if config_idx == 0:
                method_cell = method_name
                beta_cell = beta_str
            else:
                method_cell = ""
                beta_cell = ""
            
            # Renorm and clip checkmarks
            renorm_cell = "\\checkmark" if renorm else ""
            clip_cell = "\\checkmark" if clip else ""
            
            # Get formatted values
            scs_str = formatted_values['scs'][row_index]
            tcs_str = formatted_values['tcs'][row_index]
            ab_str = formatted_values['ab'][row_index]
            af1_str = formatted_values['af1'][row_index]
            mb_str = formatted_values['mb'][row_index]
            mf1_str = formatted_values['mf1'][row_index]
            
            row_cells = [method_cell, beta_cell, renorm_cell, clip_cell, scs_str, tcs_str, ab_str, af1_str, mb_str, mf1_str]
            
            # Custom row addition to avoid escaping LaTeX commands
            row = " & ".join(row_cells) + " \\\\"
            table.lines.append(row)
            row_index += 1
        
        # Add line between methods (except after last method)
        if method_idx < len(methods) - 1:
            table.add_hline()
    
    # Save table
    table_content = table.save(output_file)
    
    # Post-process to fix LaTeX formatting
    table_content = table_content.replace("BOLDXSTART", "\\textbf{")
    table_content = table_content.replace("BOLDXEND", "}")
    table_content = table_content.replace("ULXSTART", "\\underline{")
    table_content = table_content.replace("ULXEND", "}")
    
    # Save the corrected content
    with open(output_file, 'w') as f:
        f.write(table_content)
    
    return table_content


def main():
    """Main function to parse command line arguments and generate comparison tables."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables comparing different renorm and clip settings."
    )
    
    parser.add_argument(
        "base_path",
        help="Base path where experiments are located (e.g., /path/to/llm_exp/results/llama-2-7b-chat-hf)"
    )
    
    parser.add_argument(
        "experiment_name",
        help="Base experiment name (e.g., midsteer_sa_50k_last)"
    )
    
    parser.add_argument(
        "--mean-matching-beta",
        type=float,
        default=2.0,
        help="Beta value to use for mean matching method (default: 2.0)"
    )
    
    parser.add_argument(
        "--no-highlighting",
        action="store_true",
        help="Disable bold/underline highlighting for best/second-best results"
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for generated LaTeX files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Check if base path exists
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        sys.exit(1)
    
    # Load experiment data
    print(f"Loading experiment data from: {base_path}")
    experiment_data = load_experiment_data_with_settings(base_path, args.experiment_name)
    
    if not experiment_data:
        print("Error: No experiment data found")
        sys.exit(1)
    
    # Get all tasks
    all_tasks = set()
    for df in experiment_data.values():
        all_tasks.update(df['Task'].unique())
    
    all_tasks = sorted(all_tasks)
    print(f"Found tasks: {all_tasks}")
    
    # Beta selection
    beta_selection = {'mean_matching': args.mean_matching_beta}
    
    # Generate tables for each task
    output_dir = Path(args.output_dir)
    # Create structured output directory for renorm/clip comparisons
    comparison_output_dir = output_dir / "renorm_clip_comparison"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    
    enable_highlighting = not args.no_highlighting
    
    print(f"Output will be saved to: {comparison_output_dir}")
    
    for task in all_tasks:
        output_file = comparison_output_dir / f"{args.experiment_name}_{task}_renorm_clip_comparison.tex"
        
        print(f"\nGenerating comparison table for task: {task}")
        print(f"Output file: {output_file}")
        
        try:
            table_content = generate_comparison_table(
                experiment_data, 
                task, 
                str(output_file), 
                beta_selection=beta_selection,
                enable_highlighting=enable_highlighting
            )
            
            if table_content:
                print(f"Successfully generated table for {task}")
            else:
                print(f"No table generated for {task}")
                
        except Exception as e:
            print(f"Error generating table for {task}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll tables generated successfully in: {comparison_output_dir}")


if __name__ == "__main__":
    main() 