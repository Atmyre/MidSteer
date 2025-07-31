#!/usr/bin/env python3
"""
Script to generate Pareto efficiency frontier plots for Target Concept Score vs Alpaca BLEU.
Each point represents a different beta value for each method.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_generators import load_single_experiment_data

def calculate_pareto_frontier(points):
    """
    Calculate the Pareto frontier from a set of points.
    
    Args:
        points: List of (x, y) tuples where x=TCS, y=Alpaca_BLEU
        
    Returns:
        List of (x, y) tuples representing the Pareto frontier, sorted by x
    """
    if not points:
        return []
    
    # Remove any points with NaN values
    valid_points = [(x, y) for x, y in points if not (np.isnan(x) or np.isnan(y))]
    
    if not valid_points:
        return []
    
    # Sort points by TCS (x-axis) in descending order for Pareto frontier calculation
    sorted_points = sorted(valid_points, key=lambda p: p[0], reverse=True)
    
    pareto_frontier = []
    max_bleu = -float('inf')
    
    for tcs, bleu in sorted_points:
        if bleu > max_bleu:
            pareto_frontier.append((tcs, bleu))
            max_bleu = bleu
    
    # Sort by TCS for plotting
    pareto_frontier.sort(key=lambda p: p[0])
    
    return pareto_frontier


def plot_pareto_frontier(experiment_path, output_dir, enable_highlighting=True, mixed_experiment_paths=None):
    """Generate Pareto frontier plots for each task in a single experiment or mixed experiments."""
    
    if mixed_experiment_paths:
        print("Loading data from mixed experiments...")
        print(f"Clip experiment path: {mixed_experiment_paths['clip']}")
        print(f"No-clip experiment path: {mixed_experiment_paths['no_clip']}")
        print(f"Output directory: {output_dir}")
        
        try:
            # Load data from both experiments
            df_clip = load_single_experiment_data(mixed_experiment_paths['clip'])
            df_no_clip = load_single_experiment_data(mixed_experiment_paths['no_clip'])
            
            if df_clip.empty or df_no_clip.empty:
                print("No data found in one or both experiments!")
                return
            
            # Filter methods: take LEACE and CASteer from clip, mean_matching from no_clip
            df_clip_filtered = df_clip[df_clip['Method'].isin(['leace', 'casteer', 'None'])].copy()
            df_no_clip_filtered = df_no_clip[df_no_clip['Method'].isin(['mean_matching'])].copy()
            
            # Combine the datasets
            df = pd.concat([df_clip_filtered, df_no_clip_filtered], ignore_index=True)
            
            print(f"Loaded {len(df_clip)} rows from clip experiment")
            print(f"Loaded {len(df_no_clip)} rows from no-clip experiment")
            print(f"Combined {len(df)} rows total")
            
            # Get experiment name for mixed setup
            exp_name = "mixed_clip_comparison"
            
        except Exception as e:
            print(f"Error loading mixed experiment data: {e}")
            return
    else:
        print("Loading data from single experiment...")
        print(f"Experiment path: {experiment_path}")
        print(f"Output directory: {output_dir}")
        
        try:
            # Load data
            df = load_single_experiment_data(experiment_path)
            
            if df.empty:
                print("No data found!")
                return
                
            print(f"Loaded {len(df)} rows of data")
            
            # Get experiment name from path
            exp_name = Path(experiment_path).name
        except Exception as e:
            print(f"Error loading single experiment data: {e}")
            return
    
    print(f"Methods: {sorted(df['Method'].unique())}")
    print(f"Tasks: {sorted(df['Task'].unique())}")
    print(f"Beta values: {sorted(df['Beta'].unique())}")
    
    try:
        
        # Get unique tasks
        tasks = sorted(df['Task'].unique())
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots for each task
        for task in tasks:
            task_df = df[df['Task'] == task].copy()
            
            # Remove rows with NaN values in TCS or Alpaca_BLEU
            task_df = task_df.dropna(subset=['Target_Concept_Score', 'Alpaca_BLEU'])
            
            if task_df.empty:
                print(f"No valid data for task: {task}")
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Get unique methods (excluding None if it has no valid data)
            methods = []
            method_data = {}
            all_points = []
            
            for method in ['None', 'casteer', 'leace', 'mean_matching']:
                if method in task_df['Method'].values:
                    method_df = task_df[task_df['Method'] == method]
                    method_df = method_df.dropna(subset=['Target_Concept_Score', 'Alpaca_BLEU'])
                    
                    if not method_df.empty:
                        methods.append(method)
                        tcs_values = method_df['Target_Concept_Score'].values
                        bleu_values = method_df['Alpaca_BLEU'].values
                        beta_values = method_df['Beta'].values
                        
                        method_data[method] = {
                            'tcs': tcs_values,
                            'bleu': bleu_values,
                            'beta': beta_values
                        }
                        
                        # Collect all points for overall Pareto frontier
                        all_points.extend(list(zip(tcs_values, bleu_values)))
            
            if not methods:
                print(f"No valid methods found for task: {task}")
                continue
            
            # Define colors and markers for each method
            if mixed_experiment_paths:
                # Mixed experiment: indicate clipping status in labels
                method_styles = {
                    'None': {'color': 'gray', 'marker': 's', 'label': 'Baseline'},
                    'casteer': {'color': 'red', 'marker': 'o', 'label': 'CASteer (with clipping)'},
                    'leace': {'color': 'blue', 'marker': '^', 'label': 'LEACE (with clipping)'},
                    'mean_matching': {'color': 'green', 'marker': 'D', 'label': 'Mean Matching (no clipping)'}
                }
            else:
                # Single experiment: use standard labels
                method_styles = {
                    'None': {'color': 'gray', 'marker': 's', 'label': 'Baseline'},
                    'casteer': {'color': 'red', 'marker': 'o', 'label': 'CASteer'},
                    'leace': {'color': 'blue', 'marker': '^', 'label': 'LEACE'},
                    'mean_matching': {'color': 'green', 'marker': 'D', 'label': 'Mean Matching'}
                }
            
            # Plot each method
            for method in methods:
                data = method_data[method]
                style = method_styles[method]
                
                # Plot points individually to ensure marker consistency
                for i, (tcs, bleu, beta) in enumerate(zip(data['tcs'], data['bleu'], data['beta'])):
                    # Normalize beta for color mapping
                    all_betas = []
                    for m in methods:
                        all_betas.extend(method_data[m]['beta'])
                    
                    if all_betas:
                        beta_norm = (beta - min(all_betas)) / (max(all_betas) - min(all_betas)) if max(all_betas) != min(all_betas) else 0.5
                        color = plt.cm.viridis(beta_norm)
                    else:
                        color = plt.cm.viridis(0.5)
                    
                    # Only add label for the first point of each method to avoid legend duplicates
                    label = style['label'] if i == 0 else None
                    
                    plt.scatter(
                        tcs, 
                        bleu,
                        marker=style['marker'],
                        s=100,
                        alpha=0.7,
                        facecolors=color,
                        edgecolors=style['color'],
                        linewidth=2,
                        label=label
                    )
                    
                    # Add beta value annotations
                    plt.annotate(
                        f'β={beta:.1f}',
                        (tcs, bleu),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8
                    )
            
            # Calculate and plot overall Pareto frontier
            if enable_highlighting and len(all_points) > 1:
                pareto_points = calculate_pareto_frontier(all_points)
                if len(pareto_points) > 1:
                    pareto_x, pareto_y = zip(*pareto_points)
                    plt.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.8, label='Pareto Frontier')
                    
                    # Highlight Pareto frontier points with 'x' markers
                    plt.scatter(pareto_x, pareto_y, s=200, marker='x', 
                              c='black', linewidth=3, alpha=0.8)
            
            # Customize plot
            task_name = task.replace('_', ' ').replace(' to ', ' → ').title()
            plt.title(f'Pareto Frontier: {task_name}\n{exp_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Target Concept Score (TCS)', fontsize=12)
            plt.ylabel('Alpaca BLEU Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower left')
            
            # Add colorbar for beta values
            if len(methods) > 0:
                all_betas = []
                for method in methods:
                    all_betas.extend(method_data[method]['beta'])
                
                if all_betas and max(all_betas) != min(all_betas):
                    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                             norm=plt.Normalize(vmin=min(all_betas), vmax=max(all_betas)))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca())
                    cbar.set_label('Beta (β)', fontsize=12)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save plot
            output_file = output_dir / f"{exp_name}_{task}_pareto_frontier.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_file}")
            
            # Also save as PDF for publication quality
            output_file_pdf = output_dir / f"{exp_name}_{task}_pareto_frontier.pdf"
            plt.savefig(output_file_pdf, bbox_inches='tight')
            print(f"Saved plot: {output_file_pdf}")
            
            plt.close()
        
        print(f"\nAll plots generated successfully in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse command line arguments and generate Pareto frontier plots."""
    parser = argparse.ArgumentParser(
        description="Generate Pareto efficiency frontier plots from a single experiment directory or mixed experiments."
    )
    
    parser.add_argument(
        "experiment_path",
        nargs='?',
        help="Path to the experiment directory (e.g., /path/to/midsteer_sa_10k_last_renorm_clip)"
    )
    
    parser.add_argument(
        "--no-highlighting",
        action="store_true",
        help="Disable Pareto frontier highlighting (highlighting is enabled by default)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for generated plot files (default: current directory)"
    )
    
    parser.add_argument(
        "--mixed-experiments",
        action="store_true",
        help="Generate mixed experiment plots (mean_matching from no_clip, others from clip)"
    )
    
    parser.add_argument(
        "--clip-experiment",
        help="Path to the clipping experiment directory (for mixed experiments)"
    )
    
    parser.add_argument(
        "--no-clip-experiment", 
        help="Path to the no-clipping experiment directory (for mixed experiments)"
    )
    
    args = parser.parse_args()
    
    # Create structured output directory for Pareto plots
    output_dir = Path(args.output_dir)
    pareto_output_dir = output_dir / "pareto_plots"
    pareto_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved to: {pareto_output_dir}")
    
    # Generate plots
    enable_highlighting = not args.no_highlighting
    
    if args.mixed_experiments:
        # Mixed experiments mode
        if not args.clip_experiment or not args.no_clip_experiment:
            print("Error: --mixed-experiments requires both --clip-experiment and --no-clip-experiment")
            sys.exit(1)
            
        # Check if experiment paths exist
        clip_path = Path(args.clip_experiment)
        no_clip_path = Path(args.no_clip_experiment)
        
        if not clip_path.exists():
            print(f"Error: Clip experiment path does not exist: {clip_path}")
            sys.exit(1)
        if not clip_path.is_dir():
            print(f"Error: Clip experiment path is not a directory: {clip_path}")
            sys.exit(1)
            
        if not no_clip_path.exists():
            print(f"Error: No-clip experiment path does not exist: {no_clip_path}")
            sys.exit(1)
        if not no_clip_path.is_dir():
            print(f"Error: No-clip experiment path is not a directory: {no_clip_path}")
            sys.exit(1)
        
        mixed_experiment_paths = {
            'clip': str(clip_path),
            'no_clip': str(no_clip_path)
        }
        
        plot_pareto_frontier(None, pareto_output_dir, enable_highlighting, mixed_experiment_paths)
    else:
        # Single experiment mode
        if not args.experiment_path:
            print("Error: experiment_path is required when not using --mixed-experiments")
            sys.exit(1)
            
        experiment_path = Path(args.experiment_path)
        if not experiment_path.exists():
            print(f"Error: Experiment path does not exist: {experiment_path}")
            sys.exit(1)
        
        if not experiment_path.is_dir():
            print(f"Error: Experiment path is not a directory: {experiment_path}")
            sys.exit(1)
        
        plot_pareto_frontier(str(experiment_path), pareto_output_dir, enable_highlighting)


if __name__ == "__main__":
    main() 