"""
Utilities for code diff generation and handling multi-line patches.
"""
import os
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

def generate_diff(original_code: str, patched_code: str, 
                  filename: str = "code.py", context_lines: int = 3) -> str:
    """
    Generate a unified diff between original code and patched code.
    
    Args:
        original_code: The original source code as a string
        patched_code: The patched source code as a string
        filename: The filename to use in the diff header
        context_lines: Number of context lines to include
        
    Returns:
        A string containing the unified diff
    """
    # Split both code strings into lines
    original_lines = original_code.splitlines(keepends=True)
    patched_lines = patched_code.splitlines(keepends=True)
    
    # Generate the unified diff
    diff = difflib.unified_diff(
        original_lines, 
        patched_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context_lines,
        lineterm="\n"
    )
    
    # Join the diff lines into a single string
    return "".join(diff)

def parse_diff(diff_content: str) -> Dict[str, Any]:
    """
    Parse a unified diff into a structured format.
    
    Args:
        diff_content: The unified diff as a string
        
    Returns:
        Dictionary containing the parsed diff information
    """
    result = {
        "filename": None,
        "hunks": []
    }
    
    lines = diff_content.splitlines()
    
    # Check if we have a valid diff
    if len(lines) < 2 or not lines[0].startswith('--- ') or not lines[1].startswith('+++ '):
        return result
    
    # Extract filename from the diff header
    from_file = lines[0][4:].strip()
    if from_file.startswith('a/'):
        result["filename"] = from_file[2:]
    
    current_hunk = None
    
    # Process each line in the diff
    for line in lines[2:]:
        # Check for hunk header
        if line.startswith('@@'):
            # Parse the hunk range
            match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
            if match:
                current_hunk = {
                    "original_start": int(match.group(1)),
                    "original_count": int(match.group(2)),
                    "new_start": int(match.group(3)),
                    "new_count": int(match.group(4)),
                    "content": line,
                    "removed": [],
                    "added": []
                }
                result["hunks"].append(current_hunk)
        elif current_hunk is not None:
            # Add the line to the appropriate list
            if line.startswith('-'):
                current_hunk["removed"].append(line[1:])
            elif line.startswith('+'):
                current_hunk["added"].append(line[1:])
            else:
                # Context line, add to both lists
                current_hunk["content"] += "\n" + line
    
    return result

def apply_diff_to_file(file_path: Path, diff_content: str, 
                       reverse: bool = False) -> bool:
    """
    Apply a diff to a file directly.
    
    Args:
        file_path: Path to the file to patch
        diff_content: The unified diff to apply
        reverse: If True, apply the diff in reverse (undo the changes)
        
    Returns:
        True if the patch was applied successfully, False otherwise
    """
    try:
        # Read the file content
        with open(file_path, "r") as f:
            file_content = f.read()
        
        # Split into lines
        lines = file_content.splitlines(True)
        
        # Parse the diff
        parsed_diff = parse_diff(diff_content)
        
        # Apply each hunk
        for hunk in parsed_diff["hunks"]:
            original_start = hunk["original_start"] - 1  # Convert to 0-based index
            original_count = hunk["original_count"]
            new_start = hunk["new_start"] - 1  # Convert to 0-based index
            new_count = hunk["new_count"]
            
            if reverse:
                # For reverse patching, we swap removed and added
                removed_lines = hunk["added"]
                added_lines = hunk["removed"]
            else:
                removed_lines = hunk["removed"]
                added_lines = hunk["added"]
            
            # Check if the lines to be removed match the ones in the file
            original_slice = lines[original_start:original_start + original_count]
            original_matches = True
            
            for i, line in enumerate(removed_lines):
                if i >= len(original_slice) or original_slice[i].rstrip('\n') != line:
                    original_matches = False
                    break
            
            if not original_matches:
                return False
            
            # Apply the changes
            lines[original_start:original_start + original_count] = added_lines
        
        # Write the modified content back to the file
        with open(file_path, "w") as f:
            f.writelines(lines)
        
        return True
    
    except Exception as e:
        print(f"Error applying diff: {e}")
        return False

def identify_code_block(code: str, line_number: int) -> Tuple[int, int]:
    """
    Identify the start and end of a code block containing the specified line.
    
    Args:
        code: The source code as a string
        line_number: The line number (1-based) within the code block to identify
        
    Returns:
        Tuple of (start_line, end_line) for the identified block (1-based line numbers)
    """
    lines = code.splitlines()
    
    # Ensure line_number is within bounds
    if line_number < 1 or line_number > len(lines):
        return (line_number, line_number)
    
    # Adjust to 0-based indexing
    line_idx = line_number - 1
    
    # Get indentation of the target line
    target_line = lines[line_idx]
    target_indent = len(target_line) - len(target_line.lstrip())
    
    # Find the block start
    start_idx = line_idx
    for i in range(line_idx - 1, -1, -1):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Get indentation of the current line
        indent = len(line) - len(line.lstrip())
        
        # If indentation is less than or equal to target line,
        # we've found the start of the block (or a parent block)
        if indent <= target_indent:
            start_idx = i + 1
            break
            
        # Also check for block starters like if/def/class at the same level
        if indent == target_indent and re.match(r'^\s*(if|for|while|def|class|try|with)', line):
            start_idx = i
            break
    
    # Find the block end
    end_idx = line_idx
    for i in range(line_idx + 1, len(lines)):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Get indentation of the current line
        indent = len(line) - len(line.lstrip())
        
        # If indentation is less than or equal to target line,
        # we've found the end of the block
        if indent <= target_indent:
            end_idx = i - 1
            break
    
    # Convert back to 1-based line numbers
    return (start_idx + 1, end_idx + 1)

def extract_code_block(file_path: Path, line_range: Tuple[int, int]) -> str:
    """
    Extract a code block from a file.
    
    Args:
        file_path: Path to the file
        line_range: Tuple of (start_line, end_line) (1-based line numbers)
        
    Returns:
        The extracted code block as a string
    """
    # Read the file
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract the specified lines
    start_line, end_line = line_range
    code_block = lines[start_line - 1:end_line]
    
    return "".join(code_block)

def get_code_context(file_path: Path, focus_line: int, 
                     context_before: int = 5, context_after: int = 5) -> str:
    """
    Get a code snippet with context around a specific line.
    
    Args:
        file_path: Path to the file
        focus_line: The line to focus on (1-based)
        context_before: Number of lines of context before the focus line
        context_after: Number of lines of context after the focus line
        
    Returns:
        A string containing the code snippet with context
    """
    # Read the file
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Calculate context range
    start_line = max(1, focus_line - context_before)
    end_line = min(len(lines), focus_line + context_after)
    
    # Extract the context
    context_lines = lines[start_line - 1:end_line]
    
    # Format with line numbers
    result = []
    for i, line in enumerate(context_lines):
        line_num = start_line + i
        highlight = " > " if line_num == focus_line else "   "
        result.append(f"{line_num:4d}{highlight}{line}")
    
    return "".join(result)