#!/usr/bin/env python3
"""
High-level table generation functions for different types of LaTeX tables.

This module contains table generation functions that use the utilities from latex_utils.py.
Common data loading and formatting functions have been moved to latex_utils.py.
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# Handle imports for both direct execution and package import
try:
    from .latex_utils import (
        format_method_name, format_value, format_task_name,
        LaTeXTableBuilder, format_value_with_ranking,
        load_single_experiment_data, load_concept_scores_single,
        load_alpaca_metrics_single, load_mmlu_metrics_single,
        parse_method_and_beta, parse_task_name
    )
except ImportError:
    from latex_utils import (
        format_method_name, format_value, format_task_name,
        LaTeXTableBuilder, format_value_with_ranking,
        load_single_experiment_data, load_concept_scores_single,
        load_alpaca_metrics_single, load_mmlu_metrics_single,
        parse_method_and_beta, parse_task_name
    )


# load_single_experiment_data moved to latex_utils.py


# format_value_with_ranking moved to latex_utils.py


# load_concept_scores_single moved to latex_utils.py


# load_alpaca_metrics_single moved to latex_utils.py


# load_mmlu_metrics_single moved to latex_utils.py


def generate_single_experiment_table(experiment_path, output_file, enable_highlighting=True, custom_labels=None, return_content_only=False, task_filter=None, caption=None):
    """
    Generate separate comprehensive tables for each task in a single experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        output_file: Output LaTeX file path
        enable_highlighting: Whether to apply bold/underline highlighting for best/second-best results
        custom_labels: Dictionary mapping task names to custom LaTeX labels
        return_content_only: If True, return content without writing to file
        task_filter: If provided, only generate table for this specific task
        caption: Table caption (optional)
        
    Returns:
        LaTeX table content as string
    """
    # Load data
    df = load_single_experiment_data(experiment_path)
    
    if df.empty:
        print(f"No data found in {experiment_path}")
        return ""
    
    # Get experiment name from path
    exp_name = Path(experiment_path).name
    # Escape backslashes in experiment name for LaTeX
    exp_name_escaped = exp_name.replace("_", "\\_")
    
    # Get unique tasks, filter if task_filter is specified
    if task_filter:
        if task_filter in df['Task'].unique():
            tasks = [task_filter]
        else:
            print(f"Task '{task_filter}' not found in data. Available tasks: {sorted(df['Task'].unique())}")
            return ""
    else:
        tasks = sorted(df['Task'].unique())
    
    all_tables_content = []
    
    # Generate one table per task
    for task_idx, task in enumerate(tasks):
        task_df = df[df['Task'] == task]
        
        # Create column specification - 6 columns (SCS, TCS, A-BLEU, A-BertF1, M-BLEU, M-BertF1)
        col_spec = "l|l|cccccc|"
        
        # Escape task name for LaTeX
        task_name_formatted = format_task_name(task)
        task_name_escaped = task_name_formatted.replace("→", "$\\rightarrow$") if task_name_formatted else task
        
        # Use custom label if provided, otherwise use default
        if custom_labels and task in custom_labels:
            table_label = custom_labels[task]
        else:
            table_label = f"tab:{exp_name.replace('_', '')}_{task.replace('_', '')}"
        
        # Use provided caption or generate default
        if caption is None:
            table_caption = f"Comprehensive Results: {exp_name_escaped} - {task_name_escaped}"
        else:
            table_caption = caption
        
        table = LaTeXTableBuilder(
            caption=table_caption,
            label=table_label,
            col_spec=col_spec
        )
        
        # Header row 1 - Task name
        header1_cells = ["Method", "$\\beta$", f"\\multicolumn{{6}}{{c|}}{{{task_name_escaped}}}"]
        table.add_header_row(header1_cells)
        
        # Header row 2 - Metric names (SCS first, then TCS)
        header2_cells = ["", "", "SCS", "TCS", "A-BLEU", "A-BertF1", "M-BLEU", "M-BertF1"]
        table.add_header_row(header2_cells)
        table.add_hline()
        
        # Prepare data for ranking - collect all values for each column
        task_data = {
            'scs_values': [],
            'tcs_values': [],
            'ab_values': [],
            'af1_values': [],
            'mb_values': [],
            'mf1_values': []
        }
        
        # Collect all values for ranking
        for (method, beta), group in task_df.groupby(['Method', 'Beta']):
            if not group.empty:
                row = group.iloc[0]
                scs_val = row['Source_Concept_Score']
                tcs_val = row['Target_Concept_Score']
                ab_val = row['Alpaca_BLEU']
                af1_val = row['Alpaca_BertF1']
                mb_val = row['MMLU_BLEU']
                mf1_val = row['MMLU_BertF1']
            else:
                scs_val = tcs_val = ab_val = af1_val = mb_val = mf1_val = None
            
            # Store raw values and formatted strings
            task_data['scs_values'].append((scs_val, format_value(scs_val, 2)))
            task_data['tcs_values'].append((tcs_val, format_value(tcs_val, 2)))
            task_data['ab_values'].append((ab_val, format_value(ab_val, 3)))
            task_data['af1_values'].append((af1_val, format_value(af1_val, 3)))
            task_data['mb_values'].append((mb_val, format_value(mb_val, 3)))
            task_data['mf1_values'].append((mf1_val, format_value(mf1_val, 3)))
        
        # Apply ranking formatting
        if enable_highlighting:
            task_data['scs_formatted'] = format_value_with_ranking(task_data['scs_values'])
            task_data['tcs_formatted'] = format_value_with_ranking(task_data['tcs_values'])
            task_data['ab_formatted'] = format_value_with_ranking(task_data['ab_values'])
            task_data['af1_formatted'] = format_value_with_ranking(task_data['af1_values'])
            task_data['mb_formatted'] = format_value_with_ranking(task_data['mb_values'])
            task_data['mf1_formatted'] = format_value_with_ranking(task_data['mf1_values'])
        else:
            # No highlighting - just use the formatted values directly
            task_data['scs_formatted'] = [orig_str for _, orig_str in task_data['scs_values']]
            task_data['tcs_formatted'] = [orig_str for _, orig_str in task_data['tcs_values']]
            task_data['ab_formatted'] = [orig_str for _, orig_str in task_data['ab_values']]
            task_data['af1_formatted'] = [orig_str for _, orig_str in task_data['af1_values']]
            task_data['mb_formatted'] = [orig_str for _, orig_str in task_data['mb_values']]
            task_data['mf1_formatted'] = [orig_str for _, orig_str in task_data['mf1_values']]
        
        # Data rows - Group by method first for better visual organization
        row_index = 0
        
        # Get unique methods in desired order
        methods = sorted([m for m in task_df['Method'].unique() if m != 'None'])
        if 'None' in task_df['Method'].unique():
            methods = ['None'] + methods
        
        for method_idx, method in enumerate(methods):
            method_df = task_df[task_df['Method'] == method]
            method_name = format_method_name(method)
            
            # Get all beta values for this method
            betas = sorted(method_df['Beta'].unique())
            
            # Generate rows for each beta value of this method
            for beta_idx, beta in enumerate(betas):
                beta_group = method_df[method_df['Beta'] == beta]
                
                if not beta_group.empty:
                    beta_str = f"{beta:.1f}" if method != 'None' else "—"
                    
                    # Method name only on first row for each method
                    if beta_idx == 0:
                        method_cell = method_name
                    else:
                        method_cell = ""
                    
                    # Get the formatted values for this row
                    scs_str = task_data['scs_formatted'][row_index]
                    tcs_str = task_data['tcs_formatted'][row_index]
                    ab_str = task_data['ab_formatted'][row_index]
                    af1_str = task_data['af1_formatted'][row_index]
                    mb_str = task_data['mb_formatted'][row_index]
                    mf1_str = task_data['mf1_formatted'][row_index]
                    
                    row_cells = [method_cell, beta_str, scs_str, tcs_str, ab_str, af1_str, mb_str, mf1_str]
                    
                    # Custom row addition to avoid escaping LaTeX commands (like in compare_renorm_clip.py)
                    row = " & ".join(row_cells) + " \\\\"
                    table.lines.append(row)
                    row_index += 1
            
            # Add line between methods (except after last method)
            if method_idx < len(methods) - 1:
                table.add_hline()
        
        # Get table content by using a temporary approach
        temp_filename = f"temp_table_{task_idx}.tex"
        table_content = table.save(temp_filename)
        all_tables_content.append(table_content)
    
    # Combine all tables
    combined_content = "\n\n".join(all_tables_content)
    
    # Clean up temporary files
    for task_idx, task in enumerate(tasks):
        temp_filename = f"temp_table_{task_idx}.tex"
        try:
            os.remove(temp_filename)
        except FileNotFoundError:
            pass  # File might not exist, that's ok
    
    # Post-process to fix LaTeX formatting
    combined_content = combined_content.replace("BOLDXSTART", "\\textbf{")
    combined_content = combined_content.replace("BOLDXEND", "}")
    combined_content = combined_content.replace("ULXSTART", "\\underline{")
    combined_content = combined_content.replace("ULXEND", "}")
    
    # Save the corrected content if not return_content_only
    if not return_content_only:
        with open(output_file, 'w') as f:
            f.write(combined_content)
        print(f"Single experiment comprehensive tables (one per task) saved to {output_file}")
    
    return combined_content


try:
    from .artifacts import TableGenerator
except ImportError:
    from artifacts import TableGenerator

class SingleExperimentTableGenerator(TableGenerator):
    """Generator for single experiment result tables."""
    
    def generate(self) -> str:
        """Generate the single experiment table LaTeX content."""
        def _generate():
            # Get the type-specific config
            type_config = self.get_type_config('single_experiment_result')
            
            # Create a temporary output file path 
            temp_output = self.output_dir / "temp_single_experiment.tex"
            
            # Generate custom labels using self.label (same pattern as RenormClipTableGenerator)
            task = type_config.task_filter
            custom_labels = {task: self.label}
            
            # Generate the table content using the existing function with custom labels
            table_content = generate_single_experiment_table(
                experiment_path=type_config.experiment_path,
                output_file=str(temp_output),
                enable_highlighting=type_config.enable_highlighting,
                custom_labels=custom_labels,
                return_content_only=True,
                task_filter=type_config.task_filter,
                caption=self.config.caption
            )
            
            return table_content
        
        return self.safe_generate(_generate)
