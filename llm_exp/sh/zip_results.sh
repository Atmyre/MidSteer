#!/usr/bin/env bash

# Script to zip scores.tsv files from llm_exp/results directory
# while preserving directory structure
#
# Usage:
#   ./zip_results.sh                           # zip all scores.tsv files
#   ./zip_results.sh experiment1 experiment2   # zip specific experiments
#   ./zip_results.sh "*midsteer*"              # zip experiments matching pattern

# Function to show usage
show_usage() {
    echo "Usage: $0 [experiment_name_or_pattern...]"
    echo "  If no arguments provided, zips all scores.tsv files"
    echo "  Otherwise, zips scores.tsv files from matching experiment directories"
    echo ""
    echo "Examples:"
    echo "  $0                           # zip all experiments"
    echo "  $0 experiment1 experiment2   # zip specific experiments"
    echo "  $0 '*midsteer*'              # zip experiments matching pattern"
}

# Remove existing scores.zip if it exists
[ -f "scores.zip" ] && rm scores.zip

# If no arguments provided, zip everything
if [ $# -eq 0 ]; then
    echo "No arguments provided. Zipping all scores.tsv files..."
    find llm_exp/results -name "scores.tsv" -type f -print0 | xargs -0 zip scores.zip
else
    # Process each argument as experiment name or pattern
    temp_file_list=$(mktemp)
    
    for pattern in "$@"; do
        echo "Processing pattern: $pattern"
        
        # Find matching directories and then look for scores.tsv files in them
        find llm_exp/results -maxdepth 2 -type d -name "*${pattern}*" | while read -r dir; do
            find "$dir" -name "scores.tsv" -type f >> "$temp_file_list"
        done
    done
    
    # Check if any files were found
    if [ -s "$temp_file_list" ]; then
        # Remove duplicates and create zip
        sort -u "$temp_file_list" | tr '\n' '\0' | xargs -0 zip scores.zip
        echo "Created scores.zip with $(wc -l < "$temp_file_list") files"
    else
        echo "No scores.tsv files found matching the specified patterns"
        echo "Available experiments:"
        find llm_exp/results -type d -mindepth 2 -maxdepth 2 | sed 's|llm_exp/results/[^/]*/||' | sort -u
        rm "$temp_file_list"
        exit 1
    fi
    
    rm "$temp_file_list"
fi

