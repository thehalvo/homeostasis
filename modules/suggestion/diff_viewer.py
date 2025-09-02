"""
Diff Viewer for fix suggestions.

Provides utilities for creating and visualizing code diffs for fix suggestions.
"""

import difflib
import html
import logging

logger = logging.getLogger(__name__)


def create_diff(original_code: str, suggested_code: str) -> str:
    """Create a unified diff between original and suggested code.
    
    Args:
        original_code: Original code
        suggested_code: Suggested code
        
    Returns:
        str: Unified diff
    """
    # Split code into lines
    original_lines = original_code.splitlines(keepends=True)
    suggested_lines = suggested_code.splitlines(keepends=True)
    
    # Create unified diff
    diff = difflib.unified_diff(
        original_lines,
        suggested_lines,
        fromfile='Original',
        tofile='Suggested',
        n=3  # Context lines
    )
    
    return ''.join(diff)


def highlight_diff(original_code: str, suggested_code: str, 
                  context_lines: int = 3) -> str:
    """Create an HTML-highlighted diff between original and suggested code.
    
    Args:
        original_code: Original code
        suggested_code: Suggested code
        context_lines: Number of context lines to show
        
    Returns:
        str: HTML-highlighted diff
    """
    # Split code into lines
    original_lines = original_code.splitlines()
    suggested_lines = suggested_code.splitlines()
    
    # Create line-by-line diff
    diff = difflib.Differ().compare(original_lines, suggested_lines)
    
    # Process the diff to highlight changes and add context
    result = []
    changed_line_numbers = set()
    
    # First pass: identify changed lines
    line_num = 0
    for line in diff:
        if line.startswith('- '):
            line_num += 1
            changed_line_numbers.add(line_num)
        elif line.startswith('+ '):
            # Added lines don't increment the original line counter
            pass
        elif line.startswith('  '):
            line_num += 1
            
    # Second pass: create highlighted diff with context
    line_num = 0
    include_line = False
    context_counter = 0
    
    for line in diff:
        if line.startswith('- '):
            line_num += 1
            include_line = True
            context_counter = context_lines
            result.append(f'<span class="diff-removed">{html.escape(line)}</span>')
        elif line.startswith('+ '):
            include_line = True
            context_counter = context_lines
            result.append(f'<span class="diff-added">{html.escape(line)}</span>')
        elif line.startswith('  '):
            line_num += 1
            if context_counter > 0:
                include_line = True
                context_counter -= 1
            elif line_num in set(n + j for n in changed_line_numbers for j in range(-context_lines, 1)):
                include_line = True
            else:
                include_line = False
                
            if include_line:
                result.append(f'<span class="diff-context">{html.escape(line)}</span>')
            elif result and not result[-1].startswith('<span class="diff-ellipsis">'):
                result.append('<span class="diff-ellipsis">...</span>')
                
    # Join results
    return '<br>'.join(result)


def create_side_by_side_diff(original_code: str, suggested_code: str) -> str:
    """Create a side-by-side HTML diff of original and suggested code.
    
    Args:
        original_code: Original code
        suggested_code: Suggested code
        
    Returns:
        str: HTML side-by-side diff
    """
    # Create a HtmlDiff instance
    diff = difflib.HtmlDiff()
    
    # Split code into lines
    original_lines = original_code.splitlines()
    suggested_lines = suggested_code.splitlines()
    
    # Create side-by-side diff
    html_diff = diff.make_file(
        original_lines,
        suggested_lines,
        fromdesc='Original',
        todesc='Suggested',
        context=True,
        numlines=3
    )
    
    return html_diff


def summarize_changes(original_code: str, suggested_code: str) -> str:
    """Summarize changes between original and suggested code.
    
    Args:
        original_code: Original code
        suggested_code: Suggested code
        
    Returns:
        str: Summary of changes
    """
    # Split code into lines
    original_lines = original_code.splitlines()
    suggested_lines = suggested_code.splitlines()
    
    # Create a unified diff
    diff = difflib.unified_diff(
        original_lines,
        suggested_lines,
        fromfile='Original',
        tofile='Suggested',
        n=0  # No context
    )
    
    # Process diff to count changes
    added_lines = 0
    removed_lines = 0
    
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            added_lines += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines += 1
            
    # Generate summary
    summary = f"Changed {added_lines + removed_lines} line(s): "
    if removed_lines > 0:
        summary += f"removed {removed_lines} line(s)"
    if removed_lines > 0 and added_lines > 0:
        summary += ", "
    if added_lines > 0:
        summary += f"added {added_lines} line(s)"
        
    return summary