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


def generate_single_experiment_table(experiment_path=None, output_file=None, enable_highlighting=True, custom_label=None, return_content_only=False, task_filter=None, caption=None, selected_betas=None, experiment_paths=None):
    """
    Generate a single comprehensive table averaging results across filtered tasks.
    
    Args:
        experiment_path: Path to the experiment directory (backward compatibility)
        output_file: Output LaTeX file path
        enable_highlighting: Whether to apply bold/underline highlighting for best/second-best results
        custom_label: Custom LaTeX label for the table
        return_content_only: If True, return content without writing to file
        task_filter: Required list of task names to include in averaging
        caption: Table caption (optional)
        selected_betas: List of beta values to include (None for all)
        experiment_paths: Dict with 'casteer_leace' and 'midsteer' keys for method-specific paths
        
    Returns:
        LaTeX table content as string
    """
    # Determine data loading approach
    use_method_specific = experiment_paths is not None and 'casteer_leace' in experiment_paths and 'midsteer' in experiment_paths
    
    if use_method_specific:
        # Load data from method-specific experiments
        print(f"Using method-specific experiments:")
        print(f"  CASteer/LEACE: {experiment_paths['casteer_leace']}")
        print(f"  MidSteer: {experiment_paths['midsteer']}")
        
        # Load data from both experiments
        df_casteer_leace = load_single_experiment_data(experiment_paths['casteer_leace'])
        df_midsteer = load_single_experiment_data(experiment_paths['midsteer'])
        
        if df_casteer_leace.empty and df_midsteer.empty:
            print("No data found in either experiment")
            return ""
        
        # Filter each dataframe to only include relevant methods
        if not df_casteer_leace.empty:
            df_casteer_leace = df_casteer_leace[df_casteer_leace['Method'].isin(['casteer', 'leace'])]
        
        if not df_midsteer.empty:
            df_midsteer = df_midsteer[df_midsteer['Method'].isin(['mean_matching'])]
        
        # Combine the dataframes
        dfs_to_combine = []
        if not df_casteer_leace.empty:
            dfs_to_combine.append(df_casteer_leace)
        if not df_midsteer.empty:
            dfs_to_combine.append(df_midsteer)
        
        if dfs_to_combine:
            df = pd.concat(dfs_to_combine, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Also include baseline (None) method from either experiment (prefer casteer_leace)
        baseline_data = None
        if not df_casteer_leace.empty:
            # Get baseline from the already loaded casteer_leace data
            casteer_leace_full = load_single_experiment_data(experiment_paths['casteer_leace'])
            baseline_data = casteer_leace_full[casteer_leace_full['Method'] == 'None']
        elif not df_midsteer.empty:
            # Get baseline from the already loaded midsteer data
            midsteer_full = load_single_experiment_data(experiment_paths['midsteer'])
            baseline_data = midsteer_full[midsteer_full['Method'] == 'None']
        
        if baseline_data is not None and not baseline_data.empty:
            df = pd.concat([df, baseline_data], ignore_index=True)
        
    else:
        # Backward compatibility: single experiment path
        if experiment_path is None:
            raise ValueError("Either experiment_path or experiment_paths must be specified")
        
        df = load_single_experiment_data(experiment_path)
        
        if df.empty:
            print(f"No data found in {experiment_path}")
            return ""
    
    # task_filter is now required and must be a list
    if not task_filter or not isinstance(task_filter, list):
        raise ValueError("task_filter must be provided as a non-empty list of task names")
    
    # Get experiment name from path
    if experiment_path is not None:
        exp_name = Path(experiment_path).name
    else:
        # For method-specific experiments, create a descriptive name
        exp_name = "method_specific_experiments"
    # Escape backslashes in experiment name for LaTeX
    exp_name_escaped = exp_name.replace("_", "\\_")
    
    # Filter to only include tasks in task_filter
    available_tasks = set(df['Task'].unique())
    filtered_tasks = [task for task in task_filter if task in available_tasks]
    
    if not filtered_tasks:
        print(f"No tasks from filter {task_filter} found in data. Available tasks: {sorted(available_tasks)}")
        return ""
    
    print(f"Averaging results across tasks: {filtered_tasks}")
    
    # Filter dataframe to only include filtered tasks
    df_filtered = df[df['Task'].isin(filtered_tasks)]
    
    # Filter by selected betas if specified
    if selected_betas is not None:
        available_betas = set(df_filtered['Beta'].unique())
        selected_betas_available = [beta for beta in selected_betas if beta in available_betas]
        
        if not selected_betas_available:
            print(f"No selected betas {selected_betas} found in data. Available betas: {sorted(available_betas)}")
            return ""
        
        print(f"Filtering to selected betas: {selected_betas_available}")
        df_filtered = df_filtered[df_filtered['Beta'].isin(selected_betas_available)]
    
    # Group by Method and Beta, then average across tasks
    averaged_data = []
    for (method, beta), group in df_filtered.groupby(['Method', 'Beta']):
        # Calculate averages across tasks
        avg_row = {
            'Method': method,
            'Beta': beta,
            'Source_Concept_Score': group['Source_Concept_Score'].mean(),
            'Target_Concept_Score': group['Target_Concept_Score'].mean(),
            'Alpaca_BLEU': group['Alpaca_BLEU'].mean(),
            'Alpaca_BertF1': group['Alpaca_BertF1'].mean(),
            'MMLU_BLEU': group['MMLU_BLEU'].mean(),
            'MMLU_BertF1': group['MMLU_BertF1'].mean()
        }
        averaged_data.append(avg_row)
    
    # Convert to DataFrame
    avg_df = pd.DataFrame(averaged_data)
    
    # Create column specification - 6 columns (SCS, TCS, A-BLEU, A-BertF1, M-BLEU, M-BertF1)
    col_spec = "l|l|cccccc|"
    
    # Create task names display
    if len(filtered_tasks) == 1:
        task_display = format_task_name(filtered_tasks[0]).replace("→", "$\\rightarrow$")
    else:
        # Show "Combined Tasks" or list tasks if few enough
        if len(filtered_tasks) <= 3:
            task_names = [format_task_name(task).replace("→", "$\\rightarrow$") for task in filtered_tasks]
            task_display = " + ".join(task_names)
        else:
            task_display = f"Combined {len(filtered_tasks)} Tasks"
    
    # Use custom label if provided, otherwise generate default
    if custom_label:
        table_label = custom_label
    else:
        task_suffix = "_".join([task.replace('_', '') for task in filtered_tasks[:2]])  # Use first 2 tasks for label
        if len(filtered_tasks) > 2:
            task_suffix += f"_plus{len(filtered_tasks)-2}more"
        table_label = f"tab:{exp_name.replace('_', '')}_{task_suffix}"
    
    # Use provided caption or generate default
    if caption is None:
        table_caption = f"Comprehensive Results: {exp_name_escaped} - {task_display} (Averaged)"
    else:
        table_caption = caption
    
    table = LaTeXTableBuilder(
        caption=table_caption,
        label=table_label,
        col_spec=col_spec
    )
    
    # Header row 1 - Task display
    header1_cells = ["Method", "$\\beta$", f"\\multicolumn{{6}}{{c|}}{{{task_display}}}"]
    table.add_header_row(header1_cells)
    
    # Header row 2 - Metric names (SCS first, then TCS)
    header2_cells = ["", "", "SCS", "TCS", "A-BLEU", "A-BertF1", "M-BLEU", "M-BertF1"]
    table.add_header_row(header2_cells)
    table.add_hline()
    
    # Prepare data for ranking - collect all values for each column
    ranking_data = {
        'scs_values': [],
        'tcs_values': [],
        'ab_values': [],
        'af1_values': [],
        'mb_values': [],
        'mf1_values': []
    }
    
    # Collect all values for ranking
    for _, row in avg_df.iterrows():
        scs_val = row['Source_Concept_Score']
        tcs_val = row['Target_Concept_Score']
        ab_val = row['Alpaca_BLEU']
        af1_val = row['Alpaca_BertF1']
        mb_val = row['MMLU_BLEU']
        mf1_val = row['MMLU_BertF1']
        
        # Store raw values and formatted strings
        ranking_data['scs_values'].append((scs_val, format_value(scs_val, 2)))
        ranking_data['tcs_values'].append((tcs_val, format_value(tcs_val, 2)))
        ranking_data['ab_values'].append((ab_val, format_value(ab_val, 3)))
        ranking_data['af1_values'].append((af1_val, format_value(af1_val, 3)))
        ranking_data['mb_values'].append((mb_val, format_value(mb_val, 3)))
        ranking_data['mf1_values'].append((mf1_val, format_value(mf1_val, 3)))
    
    # Apply ranking formatting
    if enable_highlighting:
        ranking_data['scs_formatted'] = format_value_with_ranking(ranking_data['scs_values'])
        ranking_data['tcs_formatted'] = format_value_with_ranking(ranking_data['tcs_values'])
        ranking_data['ab_formatted'] = format_value_with_ranking(ranking_data['ab_values'])
        ranking_data['af1_formatted'] = format_value_with_ranking(ranking_data['af1_values'])
        ranking_data['mb_formatted'] = format_value_with_ranking(ranking_data['mb_values'])
        ranking_data['mf1_formatted'] = format_value_with_ranking(ranking_data['mf1_values'])
    else:
        # No highlighting - just use the formatted values directly
        ranking_data['scs_formatted'] = [orig_str for _, orig_str in ranking_data['scs_values']]
        ranking_data['tcs_formatted'] = [orig_str for _, orig_str in ranking_data['tcs_values']]
        ranking_data['ab_formatted'] = [orig_str for _, orig_str in ranking_data['ab_values']]
        ranking_data['af1_formatted'] = [orig_str for _, orig_str in ranking_data['af1_values']]
        ranking_data['mb_formatted'] = [orig_str for _, orig_str in ranking_data['mb_values']]
        ranking_data['mf1_formatted'] = [orig_str for _, orig_str in ranking_data['mf1_values']]
    
    # Data rows - Group by method first for better visual organization
    row_index = 0
    
    # Get unique methods in desired order
    methods = sorted([m for m in avg_df['Method'].unique() if m != 'None'])
    if 'None' in avg_df['Method'].unique():
        methods = ['None'] + methods
    
    for method_idx, method in enumerate(methods):
        method_df = avg_df[avg_df['Method'] == method]
        
        # Add superscript symbols for method-specific experiments
        if use_method_specific:
            if method in ['casteer', 'leace']:
                method_name = format_method_name(method) + "\\textsuperscript{†}"
            elif method == 'mean_matching':
                method_name = format_method_name(method) + "\\textsuperscript{‡}"
            else:
                method_name = format_method_name(method)
        else:
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
                scs_str = ranking_data['scs_formatted'][row_index]
                tcs_str = ranking_data['tcs_formatted'][row_index]
                ab_str = ranking_data['ab_formatted'][row_index]
                af1_str = ranking_data['af1_formatted'][row_index]
                mb_str = ranking_data['mb_formatted'][row_index]
                mf1_str = ranking_data['mf1_formatted'][row_index]
                
                row_cells = [method_cell, beta_str, scs_str, tcs_str, ab_str, af1_str, mb_str, mf1_str]
                
                # Custom row addition to avoid escaping LaTeX commands
                row = " & ".join(row_cells) + " \\\\"
                table.lines.append(row)
                row_index += 1
        
        # Add line between methods (except after last method)
        if method_idx < len(methods) - 1:
            table.add_hline()
    
    # Get table content and add footnote after table for method-specific experiments
    if return_content_only:
        if use_method_specific:
            # Finalize table without footer, add custom footnote, then add footer
            table.lines.extend(["\\hline", "\\end{tabular}"])
            table.lines.append("\\\\[0.5em]")
            table.lines.append("\\footnotesize")
            table.lines.append("\\textsuperscript{†} CASteer/LEACE: With clipping \\quad \\textsuperscript{‡} MidSteer: Without clipping")
            table.lines.append("\\end{table}")
            table_content = '\n'.join(table.lines)
        else:
            table_content = table.finalize()
    else:
        if use_method_specific:
            # For file output, create custom content with footnote
            table.lines.extend(["\\hline", "\\end{tabular}"])
            table.lines.append("\\\\[0.5em]")
            table.lines.append("\\footnotesize")
            table.lines.append("\\textsuperscript{†} CASteer/LEACE: With clipping \\quad \\textsuperscript{‡} MidSteer: Without clipping")
            table.lines.append("\\end{table}")
            table_content = '\n'.join(table.lines)
            with open(output_file, 'w') as f:
                f.write(table_content)
        else:
            table_content = table.save(output_file)
    
    # Post-process to fix LaTeX formatting
    table_content = table_content.replace("BOLDXSTART", "\\textbf{")
    table_content = table_content.replace("BOLDXEND", "}")
    table_content = table_content.replace("ULXSTART", "\\underline{")
    table_content = table_content.replace("ULXEND", "}")
    
    # Save the corrected content if not return_content_only
    if not return_content_only:
        with open(output_file, 'w') as f:
            f.write(table_content)
        print(f"Single experiment averaged table saved to {output_file}")
    
    return table_content


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
            
            # task_filter must be a list now
            if isinstance(type_config.task_filter, str):
                task_filter = [type_config.task_filter]
            elif isinstance(type_config.task_filter, list):
                task_filter = type_config.task_filter
            else:
                raise ValueError("task_filter must be a string or list of task names")
            
            # Determine experiment paths approach
            if type_config.experiment_path is not None:
                # Backward compatibility: single experiment path
                experiment_path = type_config.experiment_path
                experiment_paths = None
            elif type_config.casteer_leace_experiment is not None and type_config.midsteer_experiment is not None:
                # Method-specific experiment paths
                experiment_path = None
                experiment_paths = {
                    'casteer_leace': type_config.casteer_leace_experiment,
                    'midsteer': type_config.midsteer_experiment
                }
            else:
                raise ValueError("Either experiment_path or both casteer_leace_experiment and midsteer_experiment must be specified")
            
            # Generate the table content using the updated function
            table_content = generate_single_experiment_table(
                experiment_path=experiment_path,
                output_file=str(temp_output),
                enable_highlighting=type_config.enable_highlighting,
                custom_label=self.label,
                return_content_only=True,
                task_filter=task_filter,
                caption=self.config.caption,
                selected_betas=type_config.selected_betas,
                experiment_paths=experiment_paths
            )
            
            return table_content
        
        return _generate()
