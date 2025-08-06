#!/usr/bin/env python3
"""
Demo script showing how to use the new single experiment functions.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_generators import (
    load_single_experiment_data, 
    generate_single_experiment_table
)
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
        from data_loader import parse_task_name
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
        
        for method in methods:
            method_data = task_df[task_df['Method'] == method].sort_values('Beta')
            
            if method_data.empty:
                continue
                
            # Add method rows
            for i, (_, row) in enumerate(method_data.iterrows()):
                method_name = method_labels.get(method, method) if i == 0 else ""
                beta_str = "—" if row['Beta'] == 0.0 else f"{row['Beta']:.1f}"
                
                # Format values
                scs = f"{row['Source_Concept_Score']:.2f}" if pd.notna(row['Source_Concept_Score']) else "—"
                tcs = f"{row['Target_Concept_Score']:.2f}" if pd.notna(row['Target_Concept_Score']) else "—"
                
                if method == 'None':
                    # Baseline row - no Alpaca/MMLU scores
                    line = f"{method_name} & {beta_str} & {scs} & {tcs} & — & — & — & — \\\\"
                else:
                    ableu = f"{row['Alpaca_BLEU']:.3f}" if pd.notna(row['Alpaca_BLEU']) else "—"
                    abertf1 = f"{row['Alpaca_BertF1']:.3f}" if pd.notna(row['Alpaca_BertF1']) else "—"
                    mbleu = f"{row['MMLU_BLEU']:.3f}" if pd.notna(row['MMLU_BLEU']) else "—"
                    mbertf1 = f"{row['MMLU_BertF1']:.3f}" if pd.notna(row['MMLU_BertF1']) else "—"
                    
                    line = f"{method_name} & {beta_str} & {scs} & {tcs} & {ableu} & {abertf1} & {mbleu} & {mbertf1} \\\\"
                
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

def demo_single_experiment(experiment_path, output_file, enable_highlighting=True, selected_betas=None):
    """Demonstrate loading and generating tables for a single experiment."""
    
    print("Loading data from single experiment...")
    print(f"Experiment path: {experiment_path}")
    print(f"Output file: {output_file}")
    print(f"Highlighting enabled: {enable_highlighting}")
    if selected_betas:
        print(f"Beta filtering enabled: {selected_betas}")
    
    try:
        # Load data
        df = load_single_experiment_data(experiment_path)
        
        # Filter by selected betas if specified
        if selected_betas is not None:
            original_count = len(df)
            df = df[df['Beta'].isin(selected_betas)]
            filtered_count = len(df)
            print(f"Beta filtering: {original_count} → {filtered_count} rows (kept {len(df['Beta'].unique())} beta values)")
            
            if df.empty:
                print("Warning: No data remaining after beta filtering!")
                return
        
        if df.empty:
            print("No data found!")
            return
            
        print(f"Loaded {len(df)} rows of data")
        print(f"Methods: {sorted(df['Method'].unique())}")
        print(f"Tasks: {sorted(df['Task'].unique())}")
        print(f"Beta values: {sorted(df['Beta'].unique())}")
        
        # Display a sample of the data
        print("\nSample data:")
        print(df.head())
        
        # Generate comprehensive table
        print("\nGenerating comprehensive table...")
        comprehensive_table = generate_single_experiment_table(
            experiment_path, 
            output_file,
            enable_highlighting=enable_highlighting
        )
        
        print("\nDemo completed successfully!")
        print("Generated files:")
        print(f"- {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_implicit_concept_experiment(experiment_paths, source_concept, output_file, enable_highlighting=True, selected_betas=None):
    """Demonstrate loading and generating tables for implicit concept experiments."""
    
    print("Loading data from implicit concept experiment...")
    if isinstance(experiment_paths, dict):
        print(f"Method-specific experiment paths:")
        for method, path in experiment_paths.items():
            print(f"  {method}: {path}")
    else:
        print(f"Single experiment path: {experiment_paths}")
    print(f"Source concept: {source_concept}")
    print(f"Output file: {output_file}")
    print(f"Highlighting enabled: {enable_highlighting}")
    if selected_betas:
        print(f"Beta filtering enabled: {selected_betas}")
    
    try:
        # Load data with implicit concept
        df, tasks_with_concepts = load_implicit_concept_data(experiment_paths, source_concept, selected_betas)
        
        if df.empty:
            print("No data found!")
            return
            
        print(f"Loaded {len(df)} rows of data")
        print(f"Methods: {sorted(df['Method'].unique())}")
        print(f"Tasks: {sorted(df['Task'].unique())}")
        print(f"Beta values: {sorted(df['Beta'].unique())}")
        
        # Display a sample of the data
        print("\nSample data:")
        print(df.head())
        
        # Parse concept list for better logging
        if isinstance(source_concept, str):
            concept_list = [c.strip() for c in source_concept.split(',')]
        else:
            concept_list = source_concept
        
        # For implicit concepts, we need a custom table generation approach
        # since the standard generate_single_experiment_table won't handle the modified data
        print("\nGenerating comprehensive table with implicit concepts...")
        if len(concept_list) > 1:
            print(f"Using averaged scores from {len(concept_list)} implicit concepts:")
            for i, concept in enumerate(concept_list, 1):
                print(f"  {i}. '{concept}'")
        else:
            print(f"Using single implicit concept: '{concept_list[0]}'")
        
        # Generate table directly from the modified dataframe
        latex_content = generate_implicit_concept_table(df, source_concept, enable_highlighting, tasks_with_concepts)
        
        # Write the LaTeX content to the output file
        with open(output_file, 'w') as f:
            f.write(latex_content)
            
        print(f"Custom table generation completed using implicit concept scores")
        
        print("\nImplicit concept demo completed successfully!")
        print("Generated files:")
        print(f"- {output_file}")
        if len(concept_list) > 1:
            print(f"Note: Source concept scores averaged across {len(concept_list)} implicit concepts:")
            for concept in concept_list:
                print(f"  - '{concept}'")
        else:
            print(f"Note: Source concept scores updated for '{concept_list[0]}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse command line arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive LaTeX tables from experiment directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--no-highlighting",
        action="store_true",
        help="Disable bold/underline highlighting for best/second-best results (highlighting is enabled by default)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for generated LaTeX files (default: current directory)"
    )
    
    parser.add_argument(
        "--betas",
        help="Comma-separated list of beta values to include in tables (e.g., '1.0,2.0,3.0'). Default: include all available betas"
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Operation mode',
        required=True
    )
    
    # Standard mode subparser
    standard_parser = subparsers.add_parser(
        'standard',
        help='Generate tables from a single experiment directory',
        description='Standard mode: Generate comprehensive LaTeX tables from a single experiment directory.\n\n'
                   'Example:\n'
                   '  python parse_single_experiment.py standard --experiment-path /path/to/midsteer_sa_10k_last_renorm_clip',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    standard_parser.add_argument(
        "--experiment-path",
        required=True,
        help="Path to the experiment directory (e.g., /path/to/midsteer_sa_10k_last_renorm_clip)"
    )
    
    standard_parser.add_argument(
        "--output-filename",
        default="single_experiment_comprehensive.tex",
        help="Output LaTeX file name (default: single_experiment_comprehensive.tex)"
    )
    
    # Implicit concept mode subparser
    implicit_parser = subparsers.add_parser(
        'implicit',
        help='Generate tables with implicit concept averaging and method-specific experiments',
        description='Implicit concept mode: Generate tables using multiple implicit concept names and method-specific experiments.\n\n'
                   'Examples:\n'
                   '  # Multiple concepts:\n'
                   '  python parse_single_experiment.py implicit --source-concept "knight\'s riding mammal,large equine" --casteer-leace-experiment /path/to/clip_exp --midsteer-experiment /path/to/no_clip_exp\n\n'
                   '  # Single concept:\n'
                   '  python parse_single_experiment.py implicit --source-concept "knight\'s riding mammal" --casteer-leace-experiment /path/to/clip_exp --midsteer-experiment /path/to/no_clip_exp',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    implicit_parser.add_argument(
        "--source-concept",
        required=True,
        help="Source concept name(s). For multiple concepts, separate with commas: 'knight\'s riding mammal,large equine,domesticated ungulate'"
    )
    
    implicit_parser.add_argument(
        "--casteer-leace-experiment",
        required=True,
        help="Experiment directory for CASteer and LEACE methods (usually with clipping)"
    )
    
    implicit_parser.add_argument(
        "--midsteer-experiment",
        required=True, 
        help="Experiment directory for MidSteer/Mean Matching method (usually no clipping)"
    )
    
    implicit_parser.add_argument(
        "--output-filename",
        help="Output LaTeX file name (default: auto-generated based on concepts)"
    )
    
    args = parser.parse_args()
    
    enable_highlighting = not args.no_highlighting
    
    # Parse beta values if specified
    selected_betas = None
    if args.betas:
        try:
            selected_betas = [float(beta.strip()) for beta in args.betas.split(',')]
            print(f"Filtering to specified beta values: {selected_betas}")
        except ValueError as e:
            print(f"Error parsing beta values '{args.betas}': {e}")
            print("Beta values must be comma-separated numbers (e.g., '1.0,2.0,3.0')")
            sys.exit(1)
    
    if args.mode == 'implicit':
        # Implicit concept mode - validate experiment paths
        casteer_leace_path = Path(args.casteer_leace_experiment)
        if not casteer_leace_path.exists():
            print(f"Error: CASteer/LEACE experiment path does not exist: {casteer_leace_path}")
            sys.exit(1)
        if not casteer_leace_path.is_dir():
            print(f"Error: CASteer/LEACE experiment path is not a directory: {casteer_leace_path}")
            sys.exit(1)
        
        midsteer_path = Path(args.midsteer_experiment)
        if not midsteer_path.exists():
            print(f"Error: MidSteer experiment path does not exist: {midsteer_path}")
            sys.exit(1)
        if not midsteer_path.is_dir():
            print(f"Error: MidSteer experiment path is not a directory: {midsteer_path}")
            sys.exit(1)
        
        # Create experiment paths dictionary
        experiment_paths = {
            'casteer_leace': str(casteer_leace_path),
            'midsteer': str(midsteer_path)
        }
        
        # Create structured output directory for implicit concept experiments
        output_dir = Path(args.output_dir)
        implicit_concept_output_dir = output_dir / "implicit_concept"
        implicit_concept_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with source concept(s)
        concept_list = [c.strip() for c in args.source_concept.split(',')]
        if len(concept_list) == 1:
            concept_safe_name = normalize_concept_name(concept_list[0])
        else:
            # For multiple concepts, use the first concept + count
            first_concept_safe = normalize_concept_name(concept_list[0])
            concept_safe_name = f"{first_concept_safe}_plus{len(concept_list)-1}more"
        
        if args.output_filename:
            # Use user-specified filename
            filename = args.output_filename
        else:
            # Use auto-generated filename based on concept name
            filename = f"implicit_concept_{concept_safe_name}.tex"
        
        output_file = implicit_concept_output_dir / filename
        
        print(f"Output will be saved to: {implicit_concept_output_dir}")
        
        # Run the implicit concept demo
        demo_implicit_concept_experiment(experiment_paths, args.source_concept, output_file, enable_highlighting, selected_betas)
        
    elif args.mode == 'standard':
        # Standard mode
        experiment_path = Path(args.experiment_path)
        if not experiment_path.exists():
            print(f"Error: Experiment path does not exist: {experiment_path}")
            sys.exit(1)
        
        if not experiment_path.is_dir():
            print(f"Error: Experiment path is not a directory: {experiment_path}")
            sys.exit(1)
        
        # Create structured output directory for single experiments
        output_dir = Path(args.output_dir)
        single_experiment_output_dir = output_dir / "single_experiment"
        single_experiment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct full output path
        output_file = single_experiment_output_dir / args.output_filename
        
        print(f"Output will be saved to: {single_experiment_output_dir}")
        
        # Run the standard demo
        demo_single_experiment(str(experiment_path), str(output_file), enable_highlighting, selected_betas)
        
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main() 