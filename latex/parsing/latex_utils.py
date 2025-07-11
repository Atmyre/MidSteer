#!/usr/bin/env python3
"""
LaTeX formatting utilities for generating publication-ready tables.
"""

import pandas as pd


def format_method_name(method):
    """Format method names for LaTeX display."""
    method_mapping = {
        'None': 'Baseline',
        'casteer': 'CASteer',
        'leace': 'LEACE',
        'mean_matching': 'Mean Matching'
    }
    return method_mapping.get(method, method)


def format_value(value, decimals=3):
    """Format numerical values for LaTeX."""
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def format_task_name(task):
    """Format task names for LaTeX display."""
    task_mapping = {
        'horses_to_motorcycles': 'Horses→Motorcycles',
        'dogs_to_cats': 'Dogs→Cats'
    }
    return task_mapping.get(task, task.replace('_to_', '→').replace('_', ' ').title())


def create_latex_table_header(caption, label, col_spec):
    """Create standard LaTeX table header."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline"
    ]
    return lines


def create_latex_table_footer():
    """Create standard LaTeX table footer."""
    lines = [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ]
    return lines


def escape_latex_special_chars(text):
    """Escape special LaTeX characters in text."""
    # Common special characters that might appear in data
    escape_map = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}'
    }
    
    for char, escaped in escape_map.items():
        text = text.replace(char, escaped)
    
    return text


def save_latex_table(latex_lines, filename):
    """Save LaTeX table content to file."""
    with open(filename, 'w') as f:
        f.write('\n'.join(latex_lines))


class LaTeXTableBuilder:
    """Helper class for building LaTeX tables incrementally."""
    
    def __init__(self, caption, label, col_spec):
        self.lines = create_latex_table_header(caption, label, col_spec)
    
    def add_header_row(self, cells):
        """Add a header row to the table."""
        row = " & ".join(cells) + " \\\\"
        self.lines.append(row)
    
    def add_hline(self):
        """Add a horizontal line."""
        self.lines.append("\\hline")
    
    def add_data_row(self, cells):
        """Add a data row to the table."""
        # Format cells and escape special characters
        formatted_cells = []
        for cell in cells:
            if isinstance(cell, str):
                formatted_cells.append(escape_latex_special_chars(cell))
            else:
                formatted_cells.append(str(cell))
        
        row = " & ".join(formatted_cells) + " \\\\"
        self.lines.append(row)
    
    def finalize(self):
        """Finalize the table and return LaTeX content."""
        self.lines.extend(create_latex_table_footer())
        return '\n'.join(self.lines)
    
    def save(self, filename):
        """Save the table to a file."""
        content = self.finalize()
        save_latex_table(content.split('\n'), filename)
        return content
