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


def main():
    """Main function to parse command line arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive LaTeX tables from a single experiment directory."
    )
    
    parser.add_argument(
        "experiment_path",
        help="Path to the experiment directory (e.g., /path/to/midsteer_sa_10k_last_renorm_clip)"
    )
    
    parser.add_argument(
        "--no-highlighting",
        action="store_true",
        help="Disable bold/underline highlighting for best/second-best results (highlighting is enabled by default)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default="single_experiment_comprehensive.tex",
        help="Output LaTeX file name (default: single_experiment_comprehensive.tex)"
    )
    
    args = parser.parse_args()
    
    # Check if experiment path exists
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"Error: Experiment path does not exist: {experiment_path}")
        sys.exit(1)
    
    if not experiment_path.is_dir():
        print(f"Error: Experiment path is not a directory: {experiment_path}")
        sys.exit(1)
    
    # Run the demo
    enable_highlighting = not args.no_highlighting
    demo_single_experiment(str(experiment_path), args.output, enable_highlighting)


if __name__ == "__main__":
    main() 