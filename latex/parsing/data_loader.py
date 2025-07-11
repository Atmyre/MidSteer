#!/usr/bin/env python3
"""
Data loading and parsing utilities for LLM experimental results.
"""

import os
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict


def parse_method_and_beta(filename):
    """Extract method name and beta value from filename."""
    name = filename.replace('.json', '')
    
    if name == 'None_0.0':
        return 'None', 0.0
    
    parts = name.split('_')
    if len(parts) >= 2 and parts[-1].replace('.', '').isdigit():
        beta = float(parts[-1])
        method = '_'.join(parts[:-1])
        return method, beta
    
    return name, None


def parse_task_name(task_dir):
    """Parse task directory name to extract source and target concepts."""
    if '_to_' in task_dir:
        parts = task_dir.split('_to_')
        if len(parts) == 2:
            source = parts[0]
            target = parts[1].split('__')[0] if '__' in parts[1] else parts[1]
            return source, target
    return None, None
