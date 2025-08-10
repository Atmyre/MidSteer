#!/usr/bin/env python3
"""
Script to generate LaTeX tables comparing different renorm and clip settings.
For each task, creates a table showing all combinations of renorm/clip settings for each method.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latex_utils import load_single_experiment_data
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



def generate_comparison_table(experiment_data, task, output_file, beta_value=2.0, enable_highlighting=True, custom_label=None, caption=None, return_content_only=False):
    """
    Generate a comparison table for a specific task.
    
    Args:
        experiment_data: Dictionary mapping (renorm, clip) -> DataFrame
        task: Task name to generate table for
        output_file: Output LaTeX file path
        beta_value: Single beta value to use for all methods (default: 2.0)
        enable_highlighting: Whether to apply bold/underline highlighting
        custom_label: Custom label to use instead of auto-generated one (optional)
        caption: Table caption (optional)
        
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
    
    # For each method, use the specified beta value (or 0.0 for baseline)
    # Filter data to only include the specified beta value for each method
    filtered_data = []
    for method in methods:
        method_df = all_df[all_df['Method'] == method]
        if method == 'None':
            # Baseline method uses beta=0.0
            beta_df = method_df[method_df['Beta'] == 0.0]
        else:
            # All other methods use the specified beta value
            beta_df = method_df[method_df['Beta'] == beta_value]
        
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
    
    # Use custom label if provided, otherwise generate default
    table_label = custom_label if custom_label else f"tab:renorm_clip_{task.replace('_', '')}"
    
    # Use provided caption or generate default
    if caption is None:
        caption = f"Renorm/Clip Comparison - {task_name_escaped}"
    
    table = LaTeXTableBuilder(
        caption=caption,
        label=table_label,
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
        from latex_utils import format_value_with_ranking
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
        beta_str = f"{beta_value:.1f}" if method != 'None' else "—"
        
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
    
    # Get table content
    if return_content_only:
        table_content = table.finalize()
    else:
        table_content = table.save(output_file)
    
    # Post-process to fix LaTeX formatting
    table_content = table_content.replace("BOLDXSTART", "\\textbf{")
    table_content = table_content.replace("BOLDXEND", "}")
    table_content = table_content.replace("ULXSTART", "\\underline{")
    table_content = table_content.replace("ULXEND", "}")
    
    # Save the corrected content only if not return_content_only
    if not return_content_only:
        with open(output_file, 'w') as f:
            f.write(table_content)
    
    return table_content


try:
    from .artifacts import TableGenerator
except ImportError:
    from artifacts import TableGenerator

class RenormClipTableGenerator(TableGenerator):
    """Generator for renorm/clip comparison tables."""
    
    def generate(self) -> str:
        """Generate the renorm/clip comparison table LaTeX content."""
        def _generate():
            # Extract config parameters from the type-specific config
            type_config = self.get_type_config('renorm_clip_comparison_table')
            
            base_path = Path(type_config.base_path)
            experiment_name = type_config.experiment_name
            
            # Load data from all renorm/clip combinations
            experiment_data = load_experiment_data_with_settings(base_path, experiment_name)
            
            if not experiment_data:
                return f"% No data available for {experiment_name}"
            
            # Combine all data for task filtering
            combined_data = []
            for (renorm, clip), df in experiment_data.items():
                df_copy = df.copy()
                df_copy['Renorm'] = renorm
                df_copy['Clip'] = clip
                combined_data.append(df_copy)
            
            if not combined_data:
                return f"% No data available for {experiment_name}"
                
            data = pd.concat(combined_data, ignore_index=True)
            
            # Get all tasks or filter by task if specified
            if type_config.task_filter:
                tasks = [type_config.task_filter]
            else:
                tasks = sorted(data['Task'].unique())
            
            tables_content = []
            
            for task in tasks:
                # Generate table for this task (output_file not used since return_content_only=True)
                table_content = generate_comparison_table(
                    experiment_data,
                    task,
                    f"unused_{task}.tex",  # Filename not used since return_content_only=True
                    beta_value=type_config.beta_value,
                    enable_highlighting=type_config.enable_highlighting,
                    custom_label=self.label,
                    caption=self.config.caption,
                    return_content_only=True
                )
                
                if table_content:
                    tables_content.append(table_content)
            
            return '\n\n'.join(tables_content)
        
        return self.safe_generate(_generate)
