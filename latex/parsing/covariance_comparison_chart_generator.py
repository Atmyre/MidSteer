#!/usr/bin/env python3
"""
Script to generate comparison plots for different numbers of covariances for LEACE and MidSteer.
Analyzes experiments with pattern: midsteer_sa_{num_covariances}_last_no_renorm_clip
"""

import os
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latex_utils import load_single_experiment_data


def find_covariance_experiments(base_dir, pattern_template):
    """
    Find all experiments matching the covariance pattern template.
    
    Args:
        base_dir: Base directory containing experiment subdirectories
        pattern_template: Pattern template with {num} placeholder (e.g., 'midsteer_sa_{num}_last_no_renorm_no_clip')
        
    Returns:
        Dictionary mapping number of covariances to experiment path
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}
    
    experiments = {}
    
    # Convert pattern template to regex
    # Replace {num} with regex pattern for number with optional k suffix
    regex_pattern = pattern_template.replace('{num}', r'(\d+(?:\.\d+)?)(k)?')
    regex_pattern = f'^{regex_pattern}$'
    
    print(f"Using regex pattern: {regex_pattern}")
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            match = re.match(regex_pattern, exp_dir.name)
            if match:
                # Parse the number from the match
                num_str = match.group(1)
                multiplier = match.group(2)
                
                try:
                    num = float(num_str)
                    if multiplier == 'k':
                        num *= 1000
                    num_cov = int(num)
                    experiments[num_cov] = str(exp_dir)
                except ValueError:
                    continue
    
    return experiments


def aggregate_method_performance(df, method, metric='Target_Concept_Score'):
    """
    Aggregate performance for a method across all beta values.
    
    Args:
        df: DataFrame with experiment data
        method: Method name to filter by
        metric: Metric to aggregate ('Target_Concept_Score' or 'Alpaca_BLEU')
        
    Returns:
        Dictionary with aggregated statistics
    """
    method_df = df[df['Method'] == method]
    
    if method_df.empty:
        return None
    
    # Remove NaN values
    values = method_df[metric].dropna()
    
    if values.empty:
        return None
    
    return {
        'mean': values.mean(),
        'std': values.std(),
        'max': values.max(),
        'min': values.min(),
        'median': values.median()
    }


def plot_covariance_comparison(base_dir, output_dir, pattern_template, methods=['leace', 'mean_matching'], 
                              beta_filter=None):
    """
    Generate comparison plots for different numbers of covariances.
    
    Args:
        base_dir: Base directory containing experiment subdirectories
        output_dir: Output directory for plots
        pattern_template: Pattern template with {num} placeholder
        methods: List of methods to compare
        beta_filter: Specific beta value to filter by (e.g., 2.0, 2.5), or None for all betas
    """
    print(f"Searching for covariance experiments in: {base_dir}")
    print(f"Using pattern: {pattern_template}")
    if beta_filter is not None:
        print(f"Filtering by beta = {beta_filter}")
    
    # Find all covariance experiments
    experiments = find_covariance_experiments(base_dir, pattern_template)
    
    if not experiments:
        print("No covariance experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for num_cov, path in sorted(experiments.items()):
        print(f"  {num_cov:,} covariances: {Path(path).name}")
    
    # Load data from all experiments
    all_data = []
    
    for num_cov, exp_path in experiments.items():
        print(f"\nLoading data from {num_cov:,} covariances experiment...")
        
        try:
            df = load_single_experiment_data(exp_path)
            if df.empty:
                print(f"  No data found for {num_cov:,} covariances")
                continue
            
            # Filter by beta if specified
            if beta_filter is not None:
                df = df[df['Beta'] == beta_filter]
                if df.empty:
                    print(f"  No data found for {num_cov:,} covariances with beta={beta_filter}")
                    continue
                
            # Add covariance count column
            df['Num_Covariances'] = num_cov
            all_data.append(df)
            
            print(f"  Loaded {len(df)} rows" + (f" (beta={beta_filter})" if beta_filter is not None else ""))
            
        except Exception as e:
            print(f"  Error loading {num_cov:,} covariances: {e}")
            continue
    
    if not all_data:
        print("No valid data loaded!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} rows")
    print(f"Methods: {sorted(combined_df['Method'].unique())}")
    print(f"Tasks: {sorted(combined_df['Task'].unique())}")
    print(f"Covariance counts: {sorted(combined_df['Num_Covariances'].unique())}")
    
    # Filter to requested methods
    method_df = combined_df[combined_df['Method'].isin(methods)]
    
    if method_df.empty:
        print(f"No data found for methods: {methods}")
        return
    
    # Get unique tasks
    tasks = sorted(method_df['Task'].unique())
    
    # Create plots for each metric, combining both tasks on single chart
    metrics = [
        ('Target_Concept_Score', 'Target Concept Score'),
        ('Alpaca_BLEU', 'Alpaca BLEU Score')
    ]
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Determine clipping status from pattern
    clipping_suffix = ""
    if "no_clip" in pattern_template:
        clipping_suffix = " (no clipping)"
    elif "_clip" in pattern_template:
        clipping_suffix = " (with clipping)"
    
    method_labels = {
        'leace': f'LEACE{clipping_suffix}',
        'mean_matching': f'MidSteer (Mean Matching){clipping_suffix}',
        'casteer': f'CASteer{clipping_suffix}',
        'None': f'Baseline{clipping_suffix}'
    }
    
    # Define method markers (same as in plot_pareto_frontier.py)
    method_markers = {
        'leace': '^',           # triangle
        'mean_matching': 'D',   # diamond
        'casteer': 'o',         # circle
        'None': 's'             # square
    }
    
    # Define task colors (neutral color palette)
    task_colors = {
        'dogs_to_cats': '#1f77b4',         # blue (matplotlib default)
        'horses_to_motorcycles': '#ff7f0e' # orange (matplotlib default)
    }
    
    # Define task label suffixes
    task_labels = {
        'dogs_to_cats': ' (Dogs → Cats)',
        'horses_to_motorcycles': ' (Horses → Motorcycles)'
    }
    
    for metric_col, metric_label in metrics:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Aggregate data by method, task, and covariance count
        plot_data = []
        
        for task in tasks:
            task_df = method_df[method_df['Task'] == task]
            
            for method in methods:
                for num_cov in sorted(task_df['Num_Covariances'].unique()):
                    subset = task_df[(task_df['Method'] == method) & 
                                   (task_df['Num_Covariances'] == num_cov)]
                    
                    if not subset.empty:
                        stats = aggregate_method_performance(subset, method, metric_col)
                        if stats:
                            plot_data.append({
                                'Method': method,
                                'Task': task,
                                'Num_Covariances': num_cov,
                                'Mean': stats['mean'],
                                'Std': stats['std'],
                                'Max': stats['max'],
                                'Min': stats['min'],
                                'Median': stats['median']
                            })
        
        if not plot_data:
            print(f"No plot data for {metric_label}")
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create line plot with different markers for each task
        for method in methods:
            for task in tasks:
                method_task_data = plot_df[(plot_df['Method'] == method) & (plot_df['Task'] == task)]
                if method_task_data.empty:
                    continue
                
                base_label = method_labels.get(method, method)
                method_marker = method_markers.get(method, 'o')
                task_color = task_colors.get(task, 'black')
                task_suffix = task_labels.get(task, '')
                label = base_label + task_suffix
                
                # Plot with or without error bars depending on beta filter
                if beta_filter is not None:
                    # No error bars when filtering by specific beta (single values)
                    plt.plot(
                        method_task_data['Num_Covariances'],
                        method_task_data['Mean'],
                        marker=method_marker,
                        color=task_color,
                        linestyle='--',  # Dashed lines
                        linewidth=2,
                        markersize=8,
                        label=label
                    )
                else:
                    # Plot mean with error bars when aggregating across betas
                    plt.errorbar(
                        method_task_data['Num_Covariances'],
                        method_task_data['Mean'],
                        yerr=method_task_data['Std'],
                        marker=method_marker,
                        color=task_color,
                        linestyle='--',  # Dashed lines
                        linewidth=2,
                        markersize=8,
                        capsize=5,
                        label=label
                    )
        
        # Customize plot
        plt.title(f'{metric_label} vs Number of Covariances\nCombined Tasks Comparison', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Number of Covariances', fontsize=12)
        plt.ylabel(metric_label, fontsize=12)
        
        # Set x-axis to log scale for better visualization
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis labels
        ax = plt.gca()
        x_ticks = sorted(plot_df['Num_Covariances'].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:,}' if x < 1000 else f'{x//1000}k' for x in x_ticks])
        
        plt.tight_layout()
        
        # Save plot
        safe_metric = metric_col.lower().replace('_', '-')
        beta_suffix = f"_beta-{beta_filter}" if beta_filter is not None else ""
        
        # Create a simple pattern identifier from the template
        pattern_id = pattern_template.replace('midsteer_sa_{num}_last_', '').replace('_', '-')
        
        # Save as PDF only
        output_file_pdf = output_dir / f"covariance_comparison_{pattern_id}_combined-tasks_{safe_metric}{beta_suffix}.pdf"
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"Saved plot: {output_file_pdf}")
        
        plt.close()
    
    print(f"\nAll covariance comparison plots generated in: {output_dir}")


try:
    from .artifacts import ChartGenerator
except ImportError:
    from artifacts import ChartGenerator

class CovarianceComparisonChartGenerator(ChartGenerator):
    """Generator for covariance comparison charts."""
    
    def generate(self) -> str:
        """Generate covariance comparison chart and return LaTeX content."""
        def _generate():
            type_config = self.get_type_config('covariance_comparison_chart')
            
            # Create output directory for covariance plots
            covariance_output_dir = self.output_dir / "covariance_plots"
            covariance_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating covariance comparison plots in: {covariance_output_dir}")
            
            # Call the plotting function
            plot_covariance_comparison(
                base_dir=str(self.base_path),
                output_dir=covariance_output_dir,  # Pass Path object directly
                pattern_template=type_config.pattern,
                methods=type_config.methods,
                beta_filter=type_config.beta
            )
            
            # Find generated PDF files
            if not covariance_output_dir.exists():
                return "% Error: covariance_plots directory not created"
            
            # Look for PDF files in the covariance_plots directory
            pdf_files = list(covariance_output_dir.glob("*.pdf"))
            if not pdf_files:
                return "% Error: No PDF files generated in covariance_plots directory"
            
            # Generate LaTeX content for each PDF
            latex_figures = []
            for pdf_file in sorted(pdf_files):
                # Use the provided caption (now required)
                caption = self.config.caption
                
                # Generate label for this figure
                figure_label = self.label
                
                # Create relative path for LaTeX inclusion
                relative_path = f"covariance_plots/{pdf_file.name}"
                
                latex_content = f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={type_config.width} \\linewidth,center]{{{relative_path}}}
    \\caption{{{caption}}}
    \\label{{{figure_label}}}
\\end{{figure}}"""
                
                latex_figures.append(latex_content)
            
            return "\\n\\n".join(latex_figures)
        
        return self.safe_generate(_generate)

