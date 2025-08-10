#!/usr/bin/env python3
"""
Demo script showing how to use the new single experiment functions.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latex_utils import (
    load_single_experiment_data, load_concept_scores_single,
    load_alpaca_metrics_single, load_mmlu_metrics_single,
    parse_method_and_beta, parse_task_name
)
from single_experiment_table_generator import generate_single_experiment_table
import pandas as pd

def normalize_concept_name(concept_name):
    """Convert concept name to folder name format."""
    # Replace special characters with underscores
    normalized = concept_name.lower()
    normalized = normalized.replace("'", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    # Remove any double underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized

def determine_base_task(experiment_path):
    """Determine the base task from experiment path."""
    # Look for evaluation folders to determine base tasks
    eval_path = Path(experiment_path) / "evaluation"
    if not eval_path.exists():
        return []
    
    base_tasks = []
    for item in eval_path.iterdir():
        if item.is_dir():
            # Check if it's a base task (no double underscore)
            if "__" not in item.name:
                base_tasks.append(item.name)
    
    return base_tasks

def load_implicit_concept_data(experiment_paths, source_concepts, selected_betas=None):
    """Load data with implicit concept definitions from method-specific experiments."""
    # Parse multiple concepts if comma-separated
    if isinstance(source_concepts, str):
        concept_list = [c.strip() for c in source_concepts.split(',')]
    else:
        concept_list = source_concepts
    
    print(f"Loading data with implicit source concepts: {concept_list}")
    if selected_betas:
        print(f"Will filter scores to beta values: {selected_betas}")
    
    # Normalize all concept names
    normalized_concepts = [normalize_concept_name(concept) for concept in concept_list]
    print(f"Normalized concept names: {normalized_concepts}")
    
    # Handle different experiment path formats
    if isinstance(experiment_paths, dict):
        # Method-specific paths
        casteer_leace_path = experiment_paths.get('casteer_leace')
        midsteer_path = experiment_paths.get('midsteer')
        print(f"Using method-specific experiments:")
        print(f"  CASteer/LEACE: {casteer_leace_path}")
        print(f"  MidSteer: {midsteer_path}")
        
        # Use the first available path to determine base tasks and load initial structure
        primary_path = casteer_leace_path or midsteer_path
    else:
        # Single path (backwards compatibility)
        primary_path = experiment_paths
        casteer_leace_path = midsteer_path = primary_path
        print(f"Using single experiment path: {primary_path}")
    
    # Determine base tasks from primary experiment
    base_tasks = determine_base_task(primary_path)
    print(f"Found base tasks: {base_tasks}")
    
    if not base_tasks:
        raise ValueError(f"No base tasks found in {primary_path}")
    
    # Load standard data from primary experiment first
    df_standard = load_single_experiment_data(primary_path)
    
    if df_standard.empty:
        raise ValueError("No standard experiment data found")
    
    # Create a copy of the dataframe for modification
    df_modified = df_standard.copy()
    
    # Dictionary to collect scores for averaging: {(task, method, beta): [scores]}
    source_scores_to_average = {}
    target_scores_to_average = {}
    concepts_found = {}  # Track which concepts were found for each task
    
    # For each base task and each concept, try to find and load the implicit concept data
    for base_task in base_tasks:
        concepts_found[base_task] = []
        
        # Parse the task to get source and target concepts
        # parse_task_name now imported from latex_utils
        source_concept, target_concept = parse_task_name(base_task)
        if not source_concept or not target_concept:
            print(f"Could not parse task name: {base_task}")
            continue
        print(f"Task {base_task}: source='{source_concept}', target='{target_concept}'")
        
        for i, (concept, normalized_concept) in enumerate(zip(concept_list, normalized_concepts)):
            implicit_folder_name = f"{base_task}__{normalized_concept}"
            
            # Try to load from method-specific experiments
            experiment_sources = []
            if isinstance(experiment_paths, dict):
                if casteer_leace_path:
                    experiment_sources.append(('casteer_leace', casteer_leace_path))
                if midsteer_path:
                    experiment_sources.append(('midsteer', midsteer_path))
            else:
                experiment_sources.append(('single', primary_path))
            
            concept_found = False
            
            for source_name, exp_path in experiment_sources:
                implicit_path = Path(exp_path) / "evaluation" / implicit_folder_name
                
                print(f"Looking for implicit concept folder ({i+1}/{len(concept_list)}) in {source_name}: {implicit_path}")
                
                if implicit_path.exists():
                    print(f"Found implicit concept folder in {source_name}: {implicit_folder_name}")
                    if not concept_found:
                        concepts_found[base_task].append(concept)
                        concept_found = True
                    
                    # Load scores from the implicit concept folder
                    eval_path = implicit_path / "eval"
                    if eval_path.exists():
                        scores_file = eval_path / "scores.tsv"
                        if scores_file.exists():
                            print(f"Loading scores from {source_name}: {scores_file}")
                            
                            # Read the scores file
                            try:
                                scores_df = pd.read_csv(scores_file, sep='\t')
                                print(f"Loaded implicit concept scores from {source_name}: {len(scores_df)} rows")
                                
                                # Filter for both source concept scores (implicit concepts) and target concept scores
                                concept_matches = [
                                    concept.lower(),
                                    normalized_concept,
                                    concept  # try original case too
                                ]
                                
                                # Get scores for implicit concept (source scores)
                                implicit_scores = scores_df[scores_df['concept'].isin(concept_matches)]
                                print(f"Found {len(implicit_scores)} rows for source concept '{concept}' in {source_name}")
                                
                                # Get scores for target concept
                                target_scores = scores_df[scores_df['concept'] == target_concept]
                                print(f"Found {len(target_scores)} rows for target concept '{target_concept}' in {source_name}")
                                
                                # Process both source and target scores
                                for score_type, scores_subset in [('source', implicit_scores), ('target', target_scores)]:
                                    if not scores_subset.empty:
                                        # Parse method and beta from filename and collect scores
                                        for _, score_row in scores_subset.iterrows():
                                            filename = score_row['file']
                                            # Parse method and beta from filename like "casteer_2.0.json"
                                            if filename.endswith('.json'):
                                                filename_base = filename[:-5]  # remove .json
                                                if '_' in filename_base:
                                                    parts = filename_base.rsplit('_', 1)  # split from right, max 1 split
                                                    method = parts[0]
                                                    try:
                                                        beta = float(parts[1])
                                                    except ValueError:
                                                        print(f"Could not parse beta from {filename}")
                                                        continue
                                                else:
                                                    # Handle case like "None_0.0.json"
                                                    if filename_base.startswith('None_'):
                                                        method = 'None'
                                                        try:
                                                            beta = float(filename_base[5:])  # remove "None_"
                                                        except ValueError:
                                                            print(f"Could not parse beta from {filename}")
                                                            continue
                                                    else:
                                                        print(f"Could not parse method/beta from {filename}")
                                                        continue
                                                
                                                # Only use scores from the appropriate experiment for each method
                                                if isinstance(experiment_paths, dict):
                                                    if method in ['casteer', 'leace'] and source_name != 'casteer_leace':
                                                        continue  # Skip CASteer/LEACE scores from midsteer experiment
                                                    if method == 'mean_matching' and source_name != 'midsteer':
                                                        continue  # Skip MidSteer scores from casteer_leace experiment
                                                
                                                # Filter by selected betas if specified
                                                if selected_betas is not None and beta not in selected_betas:
                                                    continue  # Skip this beta value
                                                
                                                # Collect score for averaging
                                                key = (base_task, method, beta)
                                                if score_type == 'source':
                                                    if key not in source_scores_to_average:
                                                        source_scores_to_average[key] = []
                                                    source_scores_to_average[key].append(score_row['avg_score'])
                                                    print(f"Added {method} β={beta} source score {score_row['avg_score']:.2f} from {source_name}")
                                                else:  # target
                                                    if key not in target_scores_to_average:
                                                        target_scores_to_average[key] = []
                                                    target_scores_to_average[key].append(score_row['avg_score'])
                                                    print(f"Added {method} β={beta} target score {score_row['avg_score']:.2f} from {source_name}")
                                            
                                else:
                                    print(f"No scores found for concept '{concept}' with matches: {concept_matches} in {source_name}")
                                    
                            except Exception as e:
                                print(f"Error loading scores from {scores_file}: {e}")
                        else:
                            print(f"No scores.tsv found in {eval_path}")
                    else:
                        print(f"No eval folder found in {implicit_path}")
                else:
                    print(f"Implicit concept folder not found in {source_name}: {implicit_path}")
            
            if not concept_found:
                print(f"Concept '{concept}' not found in any experiment")
    
    # Now average the scores and update the dataframe
    total_updates = 0
    
    # Update source concept scores
    for (base_task, method, beta), score_list in source_scores_to_average.items():
        if score_list:  # Only if we have scores to average
            averaged_score = sum(score_list) / len(score_list)
            
            # Find matching rows in main dataframe
            mask = (
                (df_modified['Task'] == base_task) &
                (df_modified['Method'] == method) &
                (df_modified['Beta'] == beta)
            )
            
            # Update the source concept score
            rows_updated = df_modified.loc[mask, 'Source_Concept_Score'].index
            if len(rows_updated) > 0:
                df_modified.loc[mask, 'Source_Concept_Score'] = averaged_score
                print(f"Updated {len(rows_updated)} rows for {method} beta={beta} with averaged SOURCE score {averaged_score:.2f} (from {len(score_list)} concepts)")
                total_updates += len(rows_updated)
            else:
                print(f"No matching rows found for {method} beta={beta} in task {base_task}")
    
    # Update target concept scores
    for (base_task, method, beta), score_list in target_scores_to_average.items():
        if score_list:  # Only if we have scores to average
            averaged_score = sum(score_list) / len(score_list)
            
            # Find matching rows in main dataframe
            mask = (
                (df_modified['Task'] == base_task) &
                (df_modified['Method'] == method) &
                (df_modified['Beta'] == beta)
            )
            
            # Update the target concept score
            rows_updated = df_modified.loc[mask, 'Target_Concept_Score'].index
            if len(rows_updated) > 0:
                df_modified.loc[mask, 'Target_Concept_Score'] = averaged_score
                print(f"Updated {len(rows_updated)} rows for {method} beta={beta} with averaged TARGET score {averaged_score:.2f} (from {len(score_list)} concepts)")
                total_updates += len(rows_updated)
            else:
                print(f"No matching rows found for {method} beta={beta} in task {base_task}")
    
    # Filter the final dataframe by selected betas if specified
    if selected_betas is not None:
        original_count = len(df_modified)
        df_modified = df_modified[df_modified['Beta'].isin(selected_betas)]
        filtered_count = len(df_modified)
        print(f"Final beta filtering: {original_count} → {filtered_count} rows")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total implicit concepts processed: {len(concept_list)}")
    print(f"Total dataframe rows updated: {total_updates}")
    print(f"Source concept score updates: {len(source_scores_to_average)} method/beta combinations")
    print(f"Target concept score updates: {len(target_scores_to_average)} method/beta combinations")
    if selected_betas:
        print(f"Beta values included: {selected_betas}")
    for task, found_concepts in concepts_found.items():
        print(f"Task {task}: Found {len(found_concepts)}/{len(concept_list)} implicit concepts: {found_concepts}")
    
    # Return both the modified dataframe and which tasks had concepts found
    tasks_with_implicit_concepts = [task for task, concepts in concepts_found.items() if concepts]
    return df_modified, tasks_with_implicit_concepts

def generate_implicit_concept_table(df, source_concepts, enable_highlighting=True, tasks_with_concepts=None):
    """Generate LaTeX table from dataframe with implicit concept scores."""
    
    # Parse multiple concepts if comma-separated
    if isinstance(source_concepts, str):
        concept_list = [c.strip() for c in source_concepts.split(',')]
    else:
        concept_list = source_concepts
    
    # Get unique tasks - filter to only tasks that have implicit concepts if specified
    if tasks_with_concepts is not None:
        tasks = sorted([task for task in df['Task'].unique() if task in tasks_with_concepts])
        if not tasks:
            print("Warning: No tasks found with implicit concept data!")
            return "% No tasks with implicit concept data found"
    else:
        tasks = sorted(df['Task'].unique())
    
    tables = []
    
    for task in tasks:
        task_df = df[df['Task'] == task].copy()
        
        if task_df.empty:
            continue
        
        # Create task-specific table
        task_name_formatted = task.replace('_', ' ').replace(' to ', '$\\rightarrow$').title()
        
        # Create experiment name based on number of concepts
        if len(concept_list) == 1:
            concept_display = concept_list[0]
        else:
            concept_display = f"avg. of {len(concept_list)} concepts"
        
        experiment_name = f"Implicit Concept ({concept_display}) - {task_name_formatted}"
        
        # Start building the table
        table_lines = []
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append(f"\\caption{{Comprehensive Results: {experiment_name}}}")
        
        # Generate label - use first concept name for label consistency
        first_concept_normalized = normalize_concept_name(concept_list[0])
        label = f"tab:implicitconcept_{first_concept_normalized}_{task.replace('_', '')}"
        if len(concept_list) > 1:
            label += f"_avg{len(concept_list)}"
        table_lines.append(f"\\label{{{label}}}")
        
        table_lines.append("\\begin{tabular}{l|l|cccccc|}")
        table_lines.append("\\hline")
        
        # Update header to show averaging if multiple concepts
        if len(concept_list) == 1:
            header_text = f"SCS with {concept_list[0]}"
        else:
            header_text = f"SCS (avg. {len(concept_list)} implicit concepts)"
        
        table_lines.append(f"Method & $\\beta$ & \\multicolumn{{6}}{{c|}}{{{header_text}}} \\\\")
        table_lines.append(" &  & SCS & TCS & A-BLEU & A-BertF1 & M-BLEU & M-BertF1 \\\\")
        table_lines.append("\\hline")
        
        # Group by method
        methods = ['None', 'casteer', 'leace', 'mean_matching']
        method_labels = {
            'None': 'Baseline',
            'casteer': 'CASteer\\textsuperscript{†}',
            'leace': 'LEACE\\textsuperscript{†}', 
            'mean_matching': 'Mean Matching\\textsuperscript{‡}'
        }
        
        # Collect all rows for ranking if highlighting is enabled
        all_rows = []
        for method in methods:
            method_data = task_df[task_df['Method'] == method].sort_values('Beta')
            if not method_data.empty:
                for _, row in method_data.iterrows():
                    all_rows.append(row)
        
        # Apply highlighting if enabled
        if enable_highlighting and len(all_rows) > 1:
            # Collect values for ranking (excluding baseline/None method)
            non_baseline_rows = [row for row in all_rows if row['Method'] != 'None']
            
            if len(non_baseline_rows) > 1:
                # Collect values for each metric
                scs_values = [(row['Source_Concept_Score'], f"{row['Source_Concept_Score']:.2f}") 
                              if pd.notna(row['Source_Concept_Score']) else (None, "—") 
                              for row in non_baseline_rows]
                tcs_values = [(row['Target_Concept_Score'], f"{row['Target_Concept_Score']:.2f}") 
                              if pd.notna(row['Target_Concept_Score']) else (None, "—") 
                              for row in non_baseline_rows]
                ableu_values = [(row['Alpaca_BLEU'], f"{row['Alpaca_BLEU']:.3f}") 
                                if pd.notna(row['Alpaca_BLEU']) else (None, "—") 
                                for row in non_baseline_rows]
                abertf1_values = [(row['Alpaca_BertF1'], f"{row['Alpaca_BertF1']:.3f}") 
                                  if pd.notna(row['Alpaca_BertF1']) else (None, "—") 
                                  for row in non_baseline_rows]
                mbleu_values = [(row['MMLU_BLEU'], f"{row['MMLU_BLEU']:.3f}") 
                                if pd.notna(row['MMLU_BLEU']) else (None, "—") 
                                for row in non_baseline_rows]
                mbertf1_values = [(row['MMLU_BertF1'], f"{row['MMLU_BertF1']:.3f}") 
                                  if pd.notna(row['MMLU_BertF1']) else (None, "—") 
                                  for row in non_baseline_rows]
                
                # Apply ranking (import format_value_with_ranking from table_generators)
                try:
                    from latex_utils import format_value_with_ranking
                    formatted_scs = format_value_with_ranking(scs_values, decimals=2)
                    formatted_tcs = format_value_with_ranking(tcs_values, decimals=2)
                    formatted_ableu = format_value_with_ranking(ableu_values, decimals=3)
                    formatted_abertf1 = format_value_with_ranking(abertf1_values, decimals=3)
                    formatted_mbleu = format_value_with_ranking(mbleu_values, decimals=3)
                    formatted_mbertf1 = format_value_with_ranking(mbertf1_values, decimals=3)
                except ImportError:
                    # Fallback if import fails
                    formatted_scs = [orig_str for _, orig_str in scs_values]
                    formatted_tcs = [orig_str for _, orig_str in tcs_values]
                    formatted_ableu = [orig_str for _, orig_str in ableu_values]
                    formatted_abertf1 = [orig_str for _, orig_str in abertf1_values]
                    formatted_mbleu = [orig_str for _, orig_str in mbleu_values]
                    formatted_mbertf1 = [orig_str for _, orig_str in mbertf1_values]
            else:
                # Not enough rows for ranking
                formatted_scs = formatted_tcs = formatted_ableu = formatted_abertf1 = formatted_mbleu = formatted_mbertf1 = []
        else:
            # No highlighting
            formatted_scs = formatted_tcs = formatted_ableu = formatted_abertf1 = formatted_mbleu = formatted_mbertf1 = []
        
        # Generate table rows
        non_baseline_idx = 0
        for method in methods:
            method_data = task_df[task_df['Method'] == method].sort_values('Beta')
            
            if method_data.empty:
                continue
                
            # Add method rows
            for i, (_, row) in enumerate(method_data.iterrows()):
                method_name = method_labels.get(method, method) if i == 0 else ""
                beta_str = "—" if row['Beta'] == 0.0 else f"{row['Beta']:.1f}"
                
                if method == 'None':
                    # Baseline row - no highlighting, no Alpaca/MMLU scores
                    scs = f"{row['Source_Concept_Score']:.2f}" if pd.notna(row['Source_Concept_Score']) else "—"
                    tcs = f"{row['Target_Concept_Score']:.2f}" if pd.notna(row['Target_Concept_Score']) else "—"
                    line = f"{method_name} & {beta_str} & {scs} & {tcs} & — & — & — & — \\\\"
                else:
                    # Non-baseline row - use formatted values if highlighting is enabled
                    if enable_highlighting and len(formatted_scs) > non_baseline_idx:
                        scs = formatted_scs[non_baseline_idx]
                        tcs = formatted_tcs[non_baseline_idx]
                        ableu = formatted_ableu[non_baseline_idx]
                        abertf1 = formatted_abertf1[non_baseline_idx]
                        mbleu = formatted_mbleu[non_baseline_idx]
                        mbertf1 = formatted_mbertf1[non_baseline_idx]
                    else:
                        # Fallback to simple formatting
                        scs = f"{row['Source_Concept_Score']:.2f}" if pd.notna(row['Source_Concept_Score']) else "—"
                        tcs = f"{row['Target_Concept_Score']:.2f}" if pd.notna(row['Target_Concept_Score']) else "—"
                        ableu = f"{row['Alpaca_BLEU']:.3f}" if pd.notna(row['Alpaca_BLEU']) else "—"
                        abertf1 = f"{row['Alpaca_BertF1']:.3f}" if pd.notna(row['Alpaca_BertF1']) else "—"
                        mbleu = f"{row['MMLU_BLEU']:.3f}" if pd.notna(row['MMLU_BLEU']) else "—"
                        mbertf1 = f"{row['MMLU_BertF1']:.3f}" if pd.notna(row['MMLU_BertF1']) else "—"
                    
                    line = f"{method_name} & {beta_str} & {scs} & {tcs} & {ableu} & {abertf1} & {mbleu} & {mbertf1} \\\\"
                    non_baseline_idx += 1
                
                table_lines.append(line)
            
            # Add horizontal line after each method
            table_lines.append("\\hline")
        
        table_lines.append("\\end{tabular}")
        table_lines.append("\\\\[0.5em]")
        table_lines.append("\\footnotesize")
        table_lines.append("\\textsuperscript{†} With clipping \\quad \\textsuperscript{‡} Without clipping (fully compliant matrix form)")
        table_lines.append("\\end{table}")
        
        tables.append('\n'.join(table_lines))
    
    return '\n\n'.join(tables)


try:
    from .artifacts import TableGenerator
except ImportError:
    from artifacts import TableGenerator

class ImplicitConceptTableGenerator(TableGenerator):
    """Generator for implicit concept result tables."""
    
    def generate(self) -> str:
        """Generate the implicit concept table LaTeX content."""
        def _generate():
            # Get the type-specific config
            type_config = self.get_type_config('implicit_concept_result')
            
            # Create experiment paths dictionary (format expected by load_implicit_concept_data)
            experiment_paths = {
                'casteer_leace': type_config.casteer_leace_experiment,
                'midsteer': type_config.midsteer_experiment
            }
            
            # Convert source concepts list to comma-separated string (format expected by functions)
            source_concepts_str = ",".join(type_config.source_concepts)
            
            print(f"Loading implicit concept data with concepts: {type_config.source_concepts}")
            
            # Load data with implicit concept
            df, tasks_with_concepts = load_implicit_concept_data(
                experiment_paths, 
                source_concepts_str, 
                type_config.selected_betas
            )
            
            if df.empty:
                return "% No implicit concept data found"
            
            # Generate the table content using the existing function
            table_content = generate_implicit_concept_table(
                df, 
                source_concepts_str, 
                type_config.enable_highlighting, 
                tasks_with_concepts
            )
            
            # Post-process to fix LaTeX formatting (same as other table generators)
            table_content = table_content.replace("BOLDXSTART", "\\textbf{")
            table_content = table_content.replace("BOLDXEND", "}")
            table_content = table_content.replace("ULXSTART", "\\underline{")
            table_content = table_content.replace("ULXEND", "}")
            
            # If a custom caption is provided, we need to replace the default caption
            if self.config.caption:
                # Find and replace the caption line in the generated content
                lines = table_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('\\caption{'):
                        lines[i] = f'\\caption{{{self.config.caption}}}'
                        break
                table_content = '\n'.join(lines)
            
            return table_content
        
        return self.safe_generate(_generate)
