#!/bin/bash
# Test script for the configuration-based ablation generation approach
# This script does exactly the same thing as test_config_approach.py

# Change to the project root directory (where the templates and results are)
cd "$(dirname "$0")/../.."

# Run the test using the virtual environment
/Users/astepanov/repos/mmsteer/.venv/bin/python3 latex/parsing/config_based_generator.py latex/parsing/ablation_config.yaml --test
