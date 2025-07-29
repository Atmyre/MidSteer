#!/usr/bin/env bash

# Script to zip all scores.tsv files from llm_exp/results directory
# while preserving directory structure

# Remove existing scores.zip if it exists
[ -f "scores.zip" ] && rm scores.zip

# Create zip file with all scores.tsv files, preserving directory structure
find llm_exp/results -name "scores.tsv" -type f -print0 | xargs -0 zip scores.zip

