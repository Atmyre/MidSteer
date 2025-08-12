#!/usr/bin/env python3
"""
Unified LaTeX utilities and data loading functions for generating publication-ready tables.

This module consolidates all common utilities previously scattered across multiple files:

LaTeX Formatting:
- format_method_name() - Format method names for LaTeX display
- format_value() - Format numerical values for LaTeX
- format_task_name() - Format task names for LaTeX display
- escape_latex_special_chars() - Escape special LaTeX characters

LaTeX Table Building:
- LaTeXTableBuilder - Helper class for building LaTeX tables incrementally
- create_latex_table_header() - Create standard LaTeX table headers
- create_latex_table_footer() - Create standard LaTeX table footers

Data Loading and Parsing:
- parse_method_and_beta() - Extract method name and beta value from filename
- parse_task_name() - Parse task directory names to extract source/target concepts
- load_single_experiment_data() - Main data loading function for experiments
- load_concept_scores_single() - Load concept scores from eval/scores.tsv
- load_alpaca_metrics_single() - Load Alpaca BLEU and Bert F1 scores
- load_mmlu_metrics_single() - Load MMLU BLEU and Bert F1 scores

Value Formatting:
- format_value_with_ranking() - Format values with best/second-best highlighting

This module replaces the functionality previously provided by:
- data_loader.py (now removed)
- Common functions from table_generators.py (moved here)
"""

import pandas as pd
import os
import re
from pathlib import Path
from collections import defaultdict


def format_method_name(method):
    """Format method names for LaTeX display."""
    method_mapping = {
        'None': 'Baseline',
        'casteer': 'CASteer',
        'leace': 'LEACE',
        'mean_matching': 'Mean Matching'
    }
    return method_mapping.get(method, method)


def format_value(value, decimals=3):
    """Format numerical values for LaTeX."""
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def format_task_name(task):
    """Format task names for LaTeX display."""
    task_mapping = {
        'horses_to_motorcycles': 'Horses→Motorcycles',
        'dogs_to_cats': 'Dogs→Cats'
    }
    return task_mapping.get(task, task.replace('_to_', '→').replace('_', ' ').title())


def create_latex_table_header(caption, label, col_spec):
    """Create standard LaTeX table header."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline"
    ]
    return lines


def create_latex_table_footer():
    """Create standard LaTeX table footer."""
    lines = [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ]
    return lines


def escape_latex_special_chars(text):
    """Escape special LaTeX characters in text."""
    # Common special characters that might appear in data
    escape_map = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}'
    }
    
    for char, escaped in escape_map.items():
        text = text.replace(char, escaped)
    
    return text


def save_latex_table(latex_lines, filename):
    """Save LaTeX table content to file."""
    with open(filename, 'w') as f:
        f.write('\n'.join(latex_lines))


class LaTeXTableBuilder:
    """Helper class for building LaTeX tables incrementally."""
    
    def __init__(self, caption, label, col_spec):
        self.lines = create_latex_table_header(caption, label, col_spec)
    
    def add_header_row(self, cells):
        """Add a header row to the table."""
        row = " & ".join(cells) + " \\\\"
        self.lines.append(row)
    
    def add_hline(self):
        """Add a horizontal line."""
        self.lines.append("\\hline")
    
    def add_data_row(self, cells):
        """Add a data row to the table."""
        # Format cells and escape special characters
        formatted_cells = []
        for cell in cells:
            if isinstance(cell, str):
                formatted_cells.append(escape_latex_special_chars(cell))
            else:
                formatted_cells.append(str(cell))
        
        row = " & ".join(formatted_cells) + " \\\\"
        self.lines.append(row)
    
    def add_row(self, cells):
        """Add a row to the table (alias for add_data_row for backward compatibility)."""
        return self.add_data_row(cells)
    
    def add_raw_row(self, cells):
        """Add a raw row to the table without escaping LaTeX special characters."""
        row = " & ".join(str(cell) for cell in cells) + " \\\\"
        self.lines.append(row)
    
    def finalize(self):
        """Finalize the table and return LaTeX content."""
        self.lines.extend(create_latex_table_footer())
        return '\n'.join(self.lines)
    
    def save(self, filename):
        """Save the table to a file."""
        content = self.finalize()
        save_latex_table(content.split('\n'), filename)
        return content


# Data loading functions (moved from table_generators.py)

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


def parse_method_and_beta(filename):
    """Extract method name and beta value from filename."""
    name = filename.replace('.json', '')
    
    if name == 'None_0.0':
        return 'None', 0.0
    
    parts = name.split('_')
    if len(parts) >= 2 and parts[-1].replace('.', '').isdigit():
        beta = float(parts[-1])
        method = '_'.join(parts[:-1])
        return method, beta
    
    return name, None


def parse_task_name(task_dir):
    """Parse task directory name to extract source and target concepts."""
    if '_to_' in task_dir:
        parts = task_dir.split('_to_')
        if len(parts) == 2:
            source = parts[0]
            target = parts[1].split('__')[0] if '__' in parts[1] else parts[1]
            return source, target
    return None, None


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
