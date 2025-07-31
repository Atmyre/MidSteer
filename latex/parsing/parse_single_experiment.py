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

def load_implicit_concept_data(experiment_path, source_concept):
    """Load data with implicit concept definitions."""
    print(f"Loading data with implicit source concept: {source_concept}")
    
    # Normalize the source concept name
    normalized_concept = normalize_concept_name(source_concept)
    print(f"Normalized concept name: {normalized_concept}")
    
    # Determine base tasks
    base_tasks = determine_base_task(experiment_path)
    print(f"Found base tasks: {base_tasks}")
    
    if not base_tasks:
        raise ValueError(f"No base tasks found in {experiment_path}")
    
    # Load standard data first
    df_standard = load_single_experiment_data(experiment_path)
    
    if df_standard.empty:
        raise ValueError("No standard experiment data found")
    
    # Create a copy of the dataframe for modification
    df_modified = df_standard.copy()
    
    # For each base task, try to find and load the implicit concept data
    for base_task in base_tasks:
        implicit_folder_name = f"{base_task}__{normalized_concept}"
        implicit_path = Path(experiment_path) / "evaluation" / implicit_folder_name
        
        print(f"Looking for implicit concept folder: {implicit_path}")
        
        if implicit_path.exists():
            print(f"Found implicit concept folder: {implicit_folder_name}")
            
            # Load scores from the implicit concept folder
            eval_path = implicit_path / "eval"
            if eval_path.exists():
                scores_file = eval_path / "scores.tsv"
                if scores_file.exists():
                    print(f"Loading scores from: {scores_file}")
                    
                    # Read the scores file
                    try:
                        scores_df = pd.read_csv(scores_file, sep='\t')
                        print(f"Loaded implicit concept scores: {len(scores_df)} rows")
                        print(f"Columns: {list(scores_df.columns)}")
                        print(f"Concepts in file: {scores_df['concept'].unique()}")
                        
                        # Filter for the implicit concept scores (should match the normalized concept name)
                        # The concept column might have the original name, so let's check both
                        concept_matches = [
                            source_concept.lower(),
                            normalized_concept,
                            source_concept  # try original case too
                        ]
                        
                        implicit_scores = scores_df[scores_df['concept'].isin(concept_matches)]
                        print(f"Found {len(implicit_scores)} rows for implicit concept")
                        
                        if not implicit_scores.empty:
                            # Parse method and beta from filename
                            for _, score_row in implicit_scores.iterrows():
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
                                    
                                    # Find matching rows in main dataframe
                                    mask = (
                                        (df_modified['Task'] == base_task) &
                                        (df_modified['Method'] == method) &
                                        (df_modified['Beta'] == beta)
                                    )
                                    
                                    # Update the source concept score
                                    avg_score = score_row['avg_score']
                                    rows_updated = df_modified.loc[mask, 'Source_Concept_Score'].index
                                    if len(rows_updated) > 0:
                                        df_modified.loc[mask, 'Source_Concept_Score'] = avg_score
                                        print(f"Updated {len(rows_updated)} rows for {method} beta={beta} with score {avg_score:.2f}")
                                    else:
                                        print(f"No matching rows found for {method} beta={beta} in task {base_task}")
                        else:
                            print(f"No scores found for concepts: {concept_matches}")
                            print(f"Available concepts: {list(scores_df['concept'].unique())}")
                            
                    except Exception as e:
                        print(f"Error loading scores from {scores_file}: {e}")
                else:
                    print(f"No scores.tsv found in {eval_path}")
            else:
                print(f"No eval folder found in {implicit_path}")
        else:
            print(f"Implicit concept folder not found: {implicit_path}")
    
    return df_modified

def generate_implicit_concept_table(df, source_concept, enable_highlighting=True):
    """Generate LaTeX table from dataframe with implicit concept scores."""
    
    # Get unique tasks
    tasks = sorted(df['Task'].unique())
    
    tables = []
    
    for task in tasks:
        task_df = df[df['Task'] == task].copy()
        
        if task_df.empty:
            continue
        
        # Create task-specific table
        task_name_formatted = task.replace('_', ' ').replace(' to ', '$\\rightarrow$').title()
        experiment_name = f"Implicit Concept ({source_concept}) - {task_name_formatted}"
        
        # Start building the table
        table_lines = []
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        table_lines.append(f"\\caption{{Comprehensive Results: {experiment_name}}}")
        
        # Generate label
        label = f"tab:implicitconcept_{normalize_concept_name(source_concept)}_{task.replace('_', '')}"
        table_lines.append(f"\\label{{{label}}}")
        
        table_lines.append("\\begin{tabular}{l|l|cccccc|}")
        table_lines.append("\\hline")
        table_lines.append(f"Method & $\\beta$ & \\multicolumn{{6}}{{c|}}{{SCS with {source_concept}}} \\\\")
        table_lines.append(" &  & SCS & TCS & A-BLEU & A-BertF1 & M-BLEU & M-BertF1 \\\\")
        table_lines.append("\\hline")
        
        # Group by method
        methods = ['None', 'casteer', 'leace', 'mean_matching']
        method_labels = {
            'None': 'Baseline',
            'casteer': 'CASteer',
            'leace': 'LEACE', 
            'mean_matching': 'Mean Matching'
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
        table_lines.append("\\end{table}")
        
        tables.append('\n'.join(table_lines))
    
    return '\n\n'.join(tables)

def demo_single_experiment(experiment_path, output_file, enable_highlighting=True):
    """Demonstrate loading and generating tables for a single experiment."""
    
    print("Loading data from single experiment...")
    print(f"Experiment path: {experiment_path}")
    print(f"Output file: {output_file}")
    print(f"Highlighting enabled: {enable_highlighting}")
    
    try:
        # Load data
        df = load_single_experiment_data(experiment_path)
        
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


def demo_implicit_concept_experiment(base_experiment_path, source_concept, output_file, enable_highlighting=True):
    """Demonstrate loading and generating tables for implicit concept experiments."""
    
    print("Loading data from implicit concept experiment...")
    print(f"Base experiment path: {base_experiment_path}")
    print(f"Source concept: {source_concept}")
    print(f"Output file: {output_file}")
    print(f"Highlighting enabled: {enable_highlighting}")
    
    try:
        # Load data with implicit concept
        df = load_implicit_concept_data(base_experiment_path, source_concept)
        
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
        
        # For implicit concepts, we need a custom table generation approach
        # since the standard generate_single_experiment_table won't handle the modified data
        print("\nGenerating comprehensive table with implicit concepts...")
        
        # Generate table directly from the modified dataframe
        latex_content = generate_implicit_concept_table(df, source_concept, enable_highlighting)
        
        # Write the LaTeX content to the output file
        with open(output_file, 'w') as f:
            f.write(latex_content)
            
        print(f"Custom table generation completed using implicit concept scores")
        
        print("\nImplicit concept demo completed successfully!")
        print("Generated files:")
        print(f"- {output_file}")
        print(f"Note: Source concept scores updated for '{source_concept}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse command line arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive LaTeX tables from a single experiment directory."
    )
    
    parser.add_argument(
        "experiment_path",
        nargs='?',
        help="Path to the experiment directory (e.g., /path/to/midsteer_sa_10k_last_renorm_clip). If not provided with --implicit-concept, will use 50k_all_strengths experiment."
    )
    
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
        "--output-filename",
        default="single_experiment_comprehensive.tex",
        help="Output LaTeX file name (default: single_experiment_comprehensive.tex)"
    )
    
    parser.add_argument(
        "--implicit-concept",
        action="store_true",
        help="Enable implicit concept mode"
    )
    
    parser.add_argument(
        "--source-concept",
        help="Source concept name for implicit concept mode (e.g., 'knight's riding mammal', 'large equine')"
    )
    
    parser.add_argument(
        "--base-dir",
        help="Base directory containing experiment subdirectories (for implicit concept mode)"
    )
    
    args = parser.parse_args()
    
    enable_highlighting = not args.no_highlighting
    
    if args.implicit_concept:
        # Implicit concept mode
        if not args.source_concept:
            print("Error: --source-concept is required when using --implicit-concept")
            sys.exit(1)
        
        # Determine the experiment path
        if args.experiment_path:
            experiment_path = Path(args.experiment_path)
        elif args.base_dir:
            # Use 50k all_strengths experiment
            base_dir = Path(args.base_dir)
            experiment_path = base_dir / "midsteer_sa_50k_last_no_renorm_no_clip_all_strengths"
        else:
            print("Error: Either experiment_path or --base-dir is required for implicit concept mode")
            sys.exit(1)
        
        if not experiment_path.exists():
            print(f"Error: Experiment path does not exist: {experiment_path}")
            sys.exit(1)
        
        if not experiment_path.is_dir():
            print(f"Error: Experiment path is not a directory: {experiment_path}")
            sys.exit(1)
        
        # Create structured output directory for implicit concept experiments
        output_dir = Path(args.output_dir)
        implicit_concept_output_dir = output_dir / "single_experiment_explicit"
        implicit_concept_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with source concept
        concept_safe_name = normalize_concept_name(args.source_concept)
        if args.output_filename == "single_experiment_comprehensive.tex":
            # Use default filename with concept name
            filename = f"implicit_concept_{concept_safe_name}.tex"
        else:
            filename = args.output_filename
        
        output_file = implicit_concept_output_dir / filename
        
        print(f"Output will be saved to: {implicit_concept_output_dir}")
        
        # Run the implicit concept demo
        demo_implicit_concept_experiment(str(experiment_path), args.source_concept, output_file, enable_highlighting)
        
    else:
        # Standard mode
        if not args.experiment_path:
            print("Error: experiment_path is required when not using --implicit-concept")
            sys.exit(1)
        
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
        demo_single_experiment(str(experiment_path), str(output_file), enable_highlighting)


if __name__ == "__main__":
    main() 