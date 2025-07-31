#!/usr/bin/env python3
"""
Table generation functions for different types of LaTeX tables.
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# Handle imports for both direct execution and package import
try:
    from .latex_utils import (
        format_method_name, format_value, format_task_name,
        LaTeXTableBuilder
    )
    from .data_loader import parse_method_and_beta, parse_task_name
except ImportError:
    from latex_utils import (
        format_method_name, format_value, format_task_name,
        LaTeXTableBuilder
    )
    from data_loader import parse_method_and_beta, parse_task_name


def load_single_experiment_data(experiment_path):
    """
    Load all experimental data from a single experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory (e.g., /path/to/midsteer_sa_10k_last_renorm_clip)
        
    Returns:
        pandas.DataFrame with columns: Method, Beta, Task, Target_Concept_Score, 
                                      Alpaca_BLEU, Alpaca_BertF1, MMLU_BLEU, MMLU_BertF1
    """
    experiment_path = Path(experiment_path)
    evaluation_dir = experiment_path / 'evaluation'
    
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    
    # Collect all data
    all_data = defaultdict(dict)
    
    # Get all task directories
    task_dirs = [d for d in evaluation_dir.iterdir() if d.is_dir()]
    
    for task_dir in task_dirs:
        # Parse task name - only process base task directories (without __ suffix)
        task_name = task_dir.name
        if '_to_' in task_name and '__' not in task_name:
            # Only process base task directories like "dogs_to_cats", not "dogs_to_cats__something"
            source, target = parse_task_name(task_name)
            if not source or not target:
                continue
            task_key = f"{source}_to_{target}"
        else:
            continue
            
        # Load concept scores for both target and source concepts
        target_concept_scores = load_concept_scores_single(task_dir / 'eval', target)
        source_concept_scores = load_concept_scores_single(task_dir / 'eval', source)
        
        # Load Alpaca metrics
        alpaca_metrics = load_alpaca_metrics_single(task_dir / 'alpaca')
        
        # Load MMLU metrics
        mmlu_metrics = load_mmlu_metrics_single(task_dir / 'mmlu')
        
        # Combine all metrics
        all_methods_betas = set()
        all_methods_betas.update(target_concept_scores.keys())
        all_methods_betas.update(source_concept_scores.keys())
        all_methods_betas.update(alpaca_metrics.keys())
        all_methods_betas.update(mmlu_metrics.keys())
        
        for method_beta in all_methods_betas:
            key = (*method_beta, task_key)
            all_data[key]['target_concept_score'] = target_concept_scores.get(method_beta, None)
            all_data[key]['source_concept_score'] = source_concept_scores.get(method_beta, None)
            all_data[key]['alpaca_bleu'] = alpaca_metrics.get(method_beta, {}).get('bleu', None)
            all_data[key]['alpaca_bert_f1'] = alpaca_metrics.get(method_beta, {}).get('bert_f1', None)
            all_data[key]['mmlu_bleu'] = mmlu_metrics.get(method_beta, {}).get('bleu', None)
            all_data[key]['mmlu_bert_f1'] = mmlu_metrics.get(method_beta, {}).get('bert_f1', None)
    
    # Convert to DataFrame
    rows = []
    for (method, beta, task), metrics in all_data.items():
        row = {
            'Method': method,
            'Beta': beta,
            'Task': task,
            'Target_Concept_Score': metrics.get('target_concept_score'),
            'Source_Concept_Score': metrics.get('source_concept_score'),
            'Alpaca_BLEU': metrics.get('alpaca_bleu'),
            'Alpaca_BertF1': metrics.get('alpaca_bert_f1'),
            'MMLU_BLEU': metrics.get('mmlu_bleu'),
            'MMLU_BertF1': metrics.get('mmlu_bert_f1')
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def format_value_with_ranking(values, decimals=2):
    """
    Format a list of values with best (bold) and second best (underscore) highlighting.
    
    Args:
        values: List of (value, original_string) tuples
        decimals: Number of decimal places
        
    Returns:
        List of formatted strings with LaTeX formatting
    """
    # Filter out None values and keep track of indices
    valid_values = []
    for i, (val, orig_str) in enumerate(values):
        if val is not None and pd.notna(val):
            valid_values.append((val, i, orig_str))
    
    if len(valid_values) == 0:
        return [orig_str for _, orig_str in values]
    
    # Sort by value (descending for best first)
    sorted_values = sorted(valid_values, key=lambda x: x[0], reverse=True)
    
    # Create result list
    result = []
    for i, (val, orig_str) in enumerate(values):
        if val is None or pd.isna(val):
            result.append(orig_str)
        else:
            # Find ranking of this value
            rank = None
            for j, (sorted_val, sorted_idx, _) in enumerate(sorted_values):
                if sorted_idx == i:
                    rank = j
                    break
            
            if rank == 0:  # Best
                result.append(f"BOLDXSTART{orig_str}BOLDXEND")
            elif rank == 1:  # Second best
                result.append(f"ULXSTART{orig_str}ULXEND")
            else:
                result.append(orig_str)
    
    return result


def load_concept_scores_single(eval_path, target_concept):
    """Load concept scores for target concept from eval/scores.tsv in a single experiment."""
    scores_file = eval_path / 'scores.tsv'
    if not scores_file.exists():
        return {}
    
    df = pd.read_csv(scores_file, sep='\t')
    target_df = df[df['concept'] == target_concept]
    
    scores = {}
    for _, row in target_df.iterrows():
        method, beta = parse_method_and_beta(row['file'])
        scores[(method, beta)] = row['avg_score']
    
    return scores


def load_alpaca_metrics_single(alpaca_path):
    """Load Alpaca BLEU and Bert F1 scores from alpaca/scores.tsv in a single experiment."""
    scores_file = alpaca_path / 'scores.tsv'
    if not scores_file.exists():
        return {}
    
    df = pd.read_csv(scores_file, sep='\t')
    
    metrics = {}
    for _, row in df.iterrows():
        method, beta = parse_method_and_beta(row['file'])
        metrics[(method, beta)] = {
            'bleu': row['bleu_mean'],
            'bert_f1': row['bert_f1']
        }
    
    return metrics


def load_mmlu_metrics_single(mmlu_path):
    """Load MMLU BLEU and Bert F1 scores from mmlu/scores.tsv in a single experiment."""
    scores_file = mmlu_path / 'scores.tsv'
    if not scores_file.exists():
        return {}
    
    df = pd.read_csv(scores_file, sep='\t')
    
    metrics = {}
    for _, row in df.iterrows():
        method, beta = parse_method_and_beta(row['file'])
        metrics[(method, beta)] = {
            'bleu': row['bleu_mean'],
            'bert_f1': row['bert_f1']
        }
    
    return metrics


def generate_single_experiment_table(experiment_path, output_file, enable_highlighting=True):
    """
    Generate separate comprehensive tables for each task in a single experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        output_file: Output LaTeX file path
        enable_highlighting: Whether to apply bold/underline highlighting for best/second-best results
        
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
    
    # Get unique tasks
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
        
        table = LaTeXTableBuilder(
            caption=f"Comprehensive Results: {exp_name_escaped} - {task_name_escaped}",
            label=f"tab:{exp_name.replace('_', '')}_{task.replace('_', '')}",
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
    
    # Save the corrected content
    with open(output_file, 'w') as f:
        f.write(combined_content)
    
    print(f"Single experiment comprehensive tables (one per task) saved to {output_file}")
    
    return combined_content
