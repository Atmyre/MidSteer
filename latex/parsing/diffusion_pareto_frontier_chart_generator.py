#!/usr/bin/env python3
"""
Script to generate Pareto efficiency frontier plots for diffusion concept erasure experiments.
This plots FID (lower is better) vs Erased Concept CLIP Score (lower is better) for each method.

The script processes concept erasure experiments with the structure:
{erased_concept}_{test_concept}

FID scores are aggregated for unrelated concepts (test_concept != erased_concept).
CLIP scores are computed for concept = erased_concept = test_concept (lower is better for erasure success).
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latex_utils import format_method_name


def parse_method_and_beta_diffusion(method_str):
    """Extract method name and beta value from method string (e.g., 'casteer-2.5')."""
    if method_str == 'orig':
        return 'None', 0.0
    
    parts = method_str.split('-')
    if len(parts) >= 2:
        try:
            beta = float(parts[-1])
            method = '-'.join(parts[:-1])
            return method, beta
        except ValueError:
            return method_str, None
    
    return method_str, None


def load_diffusion_experiment_data(experiment_path):
    """
    Load all experimental data from a diffusion concept erasure experiment directory.
    
    Args:
        experiment_path: Path to the experiment directory 
        
    Returns:
        pandas.DataFrame with columns: Method, Beta, Task, Erased_Concept_CLIP, 
                                      Unrelated_FID_Mean, Unrelated_FID_Std
    """
    
    experiment_path = Path(experiment_path)
    concept_erasure_dir = experiment_path / 'evaluation' / 'concept_erasure'
    
    if not concept_erasure_dir.exists():
        raise FileNotFoundError(f"Concept erasure directory not found: {concept_erasure_dir}")
    
    # Get all task directories
    task_dirs = [d for d in concept_erasure_dir.iterdir() if d.is_dir()]
    
    # Group directories by erased concept
    concept_groups = {}
    for task_dir in task_dirs:
        task_name = task_dir.name
        if '_' in task_name:
            # Split on last underscore to handle multi-word concepts
            parts = task_name.split('_')
            if len(parts) >= 2:
                # Try to find the split point - assume last part is test_concept
                test_concept = parts[-1]
                erased_concept = '_'.join(parts[:-1])
                
                if erased_concept not in concept_groups:
                    concept_groups[erased_concept] = {}
                concept_groups[erased_concept][test_concept] = task_dir
    
    all_data = []
    
    for erased_concept, test_dirs in concept_groups.items():
        # Find the directory for erased concept CLIP scores (test_concept = erased_concept)
        erased_concept_clip_data = None
        if erased_concept in test_dirs:
            clip_file = test_dirs[erased_concept] / 'clip_score.tsv'
            if clip_file.exists():
                try:
                    df_clip = pd.read_csv(clip_file, sep='\t')
                    # Filter for erased concept only (concept = erased_concept)
                    erased_concept_clip_data = df_clip[df_clip['concept'] == erased_concept].copy()
                    
                    # Check if there are any non-'orig' methods
                    if not erased_concept_clip_data.empty:
                        non_orig_methods = erased_concept_clip_data[erased_concept_clip_data['method'] != 'orig']
                        if non_orig_methods.empty:
                            print(f"Warning: Only 'orig' method found in erased concept CLIP data for {erased_concept}, skipping")
                            continue
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    print(f"Warning: Empty or corrupted CLIP file for {erased_concept}, skipping")
                    continue
        
        if erased_concept_clip_data is None or erased_concept_clip_data.empty:
            print(f"Warning: No erased concept CLIP data found for {erased_concept}")
            continue
        
        # Collect FID data from unrelated concepts
        unrelated_fid_data = []
        for test_concept, test_dir in test_dirs.items():
            # Skip if test_concept is the erased concept
            if test_concept == erased_concept:
                continue
                
            fid_file = test_dir / 'fid.tsv'
            if fid_file.exists():
                try:
                    df_fid = pd.read_csv(fid_file, sep='\t')
                    
                    # Check if FID file is empty or has no data rows
                    if df_fid.empty or len(df_fid) == 0:
                        print(f"Warning: Empty FID file for {erased_concept}_{test_concept}, skipping this test concept")
                        continue
                    
                    # Check if there are any non-'orig' methods in FID data
                    if 'method' in df_fid.columns:
                        non_orig_fid_methods = df_fid[df_fid['method'] != 'orig']
                        if non_orig_fid_methods.empty:
                            print(f"Warning: Only 'orig' method in FID data for {erased_concept}_{test_concept}, skipping this test concept")
                            continue
                    
                    df_fid['test_concept'] = test_concept
                    unrelated_fid_data.append(df_fid)
                    
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    print(f"Warning: Empty or corrupted FID file for {erased_concept}_{test_concept}, skipping this test concept")
                    continue
        
        if not unrelated_fid_data:
            print(f"Warning: No unrelated FID data found for {erased_concept}")
            continue
        
        # Combine all unrelated FID data
        combined_fid_df = pd.concat(unrelated_fid_data, ignore_index=True)
        
        # Aggregate FID scores by method (mean and std across unrelated concepts)
        fid_stats = combined_fid_df.groupby('method')['fid'].agg(['mean', 'std']).reset_index()
        fid_stats.columns = ['method', 'fid_mean', 'fid_std']
        
        # Get unique methods from both CLIP and FID data
        clip_methods = set(erased_concept_clip_data['method'].unique())
        fid_methods = set(fid_stats['method'].unique())
        all_methods = clip_methods.intersection(fid_methods)
        
        # Combine CLIP and FID data
        for method in all_methods:
            # Skip 'orig' method as it's not a steering method
            if method == 'orig':
                continue
                
            method_name, beta = parse_method_and_beta_diffusion(method)
            
            # Get erased concept CLIP score for this method
            clip_row = erased_concept_clip_data[erased_concept_clip_data['method'] == method]
            if clip_row.empty:
                continue
            clip_score = clip_row['clip_score'].iloc[0]
            
            # Get FID stats for this method
            fid_row = fid_stats[fid_stats['method'] == method]
            if fid_row.empty:
                continue
            fid_mean = fid_row['fid_mean'].iloc[0]
            fid_std = fid_row['fid_std'].iloc[0]
            
            all_data.append({
                'Method': method_name,
                'Beta': beta,
                'Task': erased_concept,  # Task is now the erased concept
                'Erased_Concept_CLIP': clip_score,
                'Unrelated_FID_Mean': fid_mean,
                'Unrelated_FID_Std': fid_std
            })
    
    return pd.DataFrame(all_data)


def calculate_pareto_frontier_diffusion(points):
    """
    Calculate the Pareto frontier from a set of points for diffusion concept erasure experiments.
    
    Args:
        points: List of (fid, clip_score) tuples where lower FID is better, lower CLIP is better (for erasure)
        
    Returns:
        List of (fid, clip_score) tuples representing the Pareto frontier, sorted by FID
    """
    if not points:
        return []
    
    # Remove any points with NaN values
    valid_points = [(fid, clip) for fid, clip in points if not (np.isnan(fid) or np.isnan(clip))]
    
    if not valid_points:
        return []
    
    # Sort points by FID (x-axis) in ascending order
    sorted_points = sorted(valid_points, key=lambda p: p[0])
    
    pareto_frontier = []
    min_clip = float('inf')
    
    # For each FID level, we want the minimum CLIP score (lower is better for erasure)
    for fid, clip in sorted_points:
        if clip < min_clip:
            pareto_frontier.append((fid, clip))
            min_clip = clip
    
    return pareto_frontier


def plot_diffusion_pareto_frontier(experiment_path, output_dir, enable_highlighting=True, mixed_experiment_paths=None):
    """Generate Pareto frontier plots for each task in a diffusion experiment."""
    
    if mixed_experiment_paths:
        print("Mixed experiments not yet supported for diffusion plots")
        return
    
    print("Loading data from diffusion experiment...")
    print(f"Experiment path: {experiment_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load data
        df = load_diffusion_experiment_data(experiment_path)
        
        if df.empty:
            print("No data found!")
            return
            
        print(f"Loaded {len(df)} rows of data")
        
        # Get experiment name from path
        exp_name = Path(experiment_path).name
    except Exception as e:
        print(f"Error loading diffusion experiment data: {e}")
        import traceback
        traceback.print_exc()
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
            
            # Remove rows with NaN values in CLIP or FID
            task_df = task_df.dropna(subset=['Erased_Concept_CLIP', 'Unrelated_FID_Mean'])
            
            if task_df.empty:
                print(f"No valid data for task: {task}")
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Get unique methods
            methods = []
            method_data = {}
            all_points = []
            
            for method in ['casteer', 'leace', 'mean_matching']:
                if method in task_df['Method'].values:
                    method_df = task_df[task_df['Method'] == method]
                    method_df = method_df.dropna(subset=['Erased_Concept_CLIP', 'Unrelated_FID_Mean'])
                    
                    if not method_df.empty:
                        methods.append(method)
                        clip_values = method_df['Erased_Concept_CLIP'].values
                        fid_values = method_df['Unrelated_FID_Mean'].values
                        fid_std_values = method_df['Unrelated_FID_Std'].values
                        beta_values = method_df['Beta'].values
                        
                        method_data[method] = {
                            'clip': clip_values,
                            'fid': fid_values,
                            'fid_std': fid_std_values,
                            'beta': beta_values
                        }
                        
                        # Collect all points for overall Pareto frontier (FID, CLIP)
                        all_points.extend(list(zip(fid_values, clip_values)))
            
            if not methods:
                print(f"No valid methods found for task: {task}")
                continue
            
            # Define colors and markers for each method
            method_styles = {
                'casteer': {'color': 'red', 'marker': 'o', 'label': 'CASteer'},
                'leace': {'color': 'blue', 'marker': '^', 'label': 'LEACE'},
                'mean_matching': {'color': 'green', 'marker': 'D', 'label': 'Mean Matching'}
            }
            
            # Plot each method
            for method in methods:
                data = method_data[method]
                style = method_styles[method]
                
                # Plot points individually to ensure marker consistency
                for i, (clip, fid, fid_std, beta) in enumerate(zip(data['clip'], data['fid'], data['fid_std'], data['beta'])):
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
                    
                    # Plot point without error bars (swapped axes: FID on x, CLIP on y)
                    plt.scatter(
                        fid,  # X-axis: FID
                        clip,  # Y-axis: CLIP score
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
                        (fid, clip),  # Swapped coordinates
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8
                    )
            
            # Calculate and plot overall Pareto frontier
            if enable_highlighting and len(all_points) > 1:
                pareto_points = calculate_pareto_frontier_diffusion(all_points)
                if len(pareto_points) > 1:
                    pareto_x, pareto_y = zip(*pareto_points)  # pareto_x=FID, pareto_y=CLIP
                    plt.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.8, label='Pareto Frontier')
                    
                    # Highlight Pareto frontier points with 'x' markers
                    plt.scatter(pareto_x, pareto_y, s=200, marker='x', 
                              c='black', linewidth=3, alpha=0.8)
            
            # Customize plot
            task_name = task.replace('_', ' ').replace(' to ', ' → ').title()
            plt.title(f'Diffusion Pareto Frontier: {task_name}\n{exp_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Unrelated Concepts FID (lower is better)', fontsize=12)
            plt.ylabel('Erased Concept CLIP Score (lower is better)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Invert y-axis so lower CLIP scores (better erasure) appear at the top
            plt.gca().invert_yaxis()
            
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
            
            # Save as PDF only for publication quality
            output_file_pdf = output_dir / f"{exp_name}_{task}_diffusion_pareto_frontier.pdf"
            plt.savefig(output_file_pdf, bbox_inches='tight')
            print(f"Saved plot: {output_file_pdf}")
            
            plt.close()
        
        print(f"\nAll diffusion plots generated successfully in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


try:
    from .artifacts import ChartGenerator
except ImportError:
    from artifacts import ChartGenerator


class DiffusionParetoFrontierChartGenerator(ChartGenerator):
    """Generator for diffusion Pareto frontier charts."""
    
    def generate(self) -> str:
        """Generate the diffusion Pareto frontier chart and LaTeX content."""
        def _generate():
            # Get the type-specific config
            type_config = self.get_type_config('diffusion_pareto_frontier_chart')
            
            # Create output directory for diffusion pareto plots
            pareto_output_dir = self.output_dir / "diffusion_pareto_plots"
            pareto_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating diffusion Pareto frontier plots in: {pareto_output_dir}")
            
            # Call the plot_diffusion_pareto_frontier function directly
            plot_diffusion_pareto_frontier(
                experiment_path=type_config.experiment_path, 
                output_dir=pareto_output_dir, 
                enable_highlighting=type_config.enable_highlighting
            )
            
            exp_name = Path(type_config.experiment_path).name
            
            # Find generated PDF files to include in LaTeX
            if not pareto_output_dir.exists():
                return "% Error: diffusion_pareto_plots directory not created"
            
            # Look for PDF files that match this artifact's experiment configuration and task
            experiment_name = Path(type_config.experiment_path).name
            if "chihuahua_muffin" in self.artifact_key:
                pdf_files = list(pareto_output_dir.glob(f"{experiment_name}_chihuahua_to_muffin_diffusion_pareto_frontier.pdf"))
            elif "horse_motorcycle" in self.artifact_key:
                pdf_files = list(pareto_output_dir.glob(f"{experiment_name}_horse_to_motorcycle_diffusion_pareto_frontier.pdf"))
            elif "snoopy_mickey" in self.artifact_key:
                pdf_files = list(pareto_output_dir.glob(f"{experiment_name}_snoopy_to_mickey_diffusion_pareto_frontier.pdf"))
            else:
                pdf_files = list(pareto_output_dir.glob(f"{experiment_name}_*_diffusion_pareto_frontier.pdf"))
            
            if not pdf_files:
                return f"% Error: No diffusion PDF files found for artifact '{self.artifact_key}'"
            
            # Generate LaTeX content for each PDF
            latex_figures = []
            for pdf_file in sorted(pdf_files):
                # Use the provided caption (now required)
                caption = self.config.caption
                
                unique_label = self.label
                
                # Use relative path from output directory
                relative_pdf_path = f"diffusion_pareto_plots/{pdf_file.name}"
                
                figure_latex = f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={type_config.width}\\linewidth,center]{{{relative_pdf_path}}}
    \\caption{{{caption}}}
    \\label{{{unique_label}}}
\\end{{figure}}"""
                
                latex_figures.append(figure_latex)
            
            return '\\n\\n'.join(latex_figures)
        
        return self.safe_generate(_generate)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diffusion_pareto_frontier_chart_generator.py <experiment_path> <output_dir>")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_diffusion_pareto_frontier(experiment_path, output_dir)
