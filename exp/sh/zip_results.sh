#!/usr/bin/env bash

# Script to archive scores.tsv files from exp/results directory
# while preserving directory structure using tar.gz
#
# Usage:
#   ./zip_results.sh                           # archive all scores.tsv files
#   ./zip_results.sh experiment1 experiment2   # archive specific experiments
#   ./zip_results.sh "*midsteer*"              # archive experiments matching pattern

# Function to show usage
show_usage() {
    echo "Usage: $0 [experiment_name_or_pattern...]"
    echo "  If no arguments provided, archives all scores.tsv files"
    echo "  Otherwise, archives scores.tsv files from matching experiment directories"
    echo ""
    echo "Examples:"
    echo "  $0                           # archive all experiments"
    echo "  $0 experiment1 experiment2   # archive specific experiments"
    echo "  $0 '*midsteer*'              # archive experiments matching pattern"
}

# Remove existing scores.tar.gz if it exists
[ -f "scores.tar.gz" ] && rm scores.tar.gz

# If no arguments provided, archive everything
if [ $# -eq 0 ]; then
    echo "No arguments provided. Archiving all scores.tsv files..."
    find exp/results/ -name "*.tsv" -type f | tar -czf scores.tar.gz -T -
else
    # Process each argument as experiment name or pattern
    temp_file_list=$(mktemp)
    
    for pattern in "$@"; do
        echo "Processing pattern: $pattern"
        
        # Find matching directories and then look for scores.tsv files in them
        find exp/results/ -maxdepth 2 -type d -name "*${pattern}*" | while read -r dir; do
            find "$dir" -name "*.tsv" -type f >> "$temp_file_list"
        done
    done
    
    # Check if any files were found
    if [ -s "$temp_file_list" ]; then
        # Remove duplicates and create tar.gz archive
        sort -u "$temp_file_list" | tar -czf scores.tar.gz -T -
        echo "Created scores.tar.gz with $(wc -l < "$temp_file_list") files"
    else
        echo "No scores.tsv files found matching the specified patterns"
        echo "Available experiments:"
        find exp/results -type d -mindepth 2 -maxdepth 2 | sed 's|exp/results/[^/]*/||' | sort -u
        rm "$temp_file_list"
        exit 1
    fi
    
    rm "$temp_file_list"
fi

