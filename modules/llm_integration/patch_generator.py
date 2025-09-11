"""
Patch Generator Module

This module provides functionality for generating code patches using LLM integration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PatchData:
    """Represents data for a code patch."""

    patch_id: str
    original_code: str
    patched_code: str
    description: str
    confidence: float
    error_type: Optional[str] = None
    fix_type: Optional[str] = None
    line_numbers: Optional[List[int]] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patch_id": self.patch_id,
            "original_code": self.original_code,
            "patched_code": self.patched_code,
            "description": self.description,
            "confidence": self.confidence,
            "error_type": self.error_type,
            "fix_type": self.fix_type,
            "line_numbers": self.line_numbers,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    def get_diff(self) -> str:
        """Get a simple diff representation."""
        lines = []
        lines.append("--- Original")
        lines.append("+++ Patched")
        lines.append(f"@@ Description: {self.description} @@")

        # Simple line-by-line diff
        original_lines = self.original_code.splitlines()
        patched_lines = self.patched_code.splitlines()

        for i, (orig, patch) in enumerate(zip(original_lines, patched_lines)):
            if orig != patch:
                lines.append(f"- {orig}")
                lines.append(f"+ {patch}")
            else:
                lines.append(f"  {orig}")

        # Handle extra lines
        if len(original_lines) > len(patched_lines):
            for line in original_lines[len(patched_lines):]:
                lines.append(f"- {line}")
        elif len(patched_lines) > len(original_lines):
            for line in patched_lines[len(original_lines)]:
                lines.append(f"+ {line}")

        return "\n".join(lines)


class PatchGenerator:
    """Generates code patches using LLM integration."""

    def __init__(self, llm_provider=None):
        """
        Initialize the patch generator.

        Args:
            llm_provider: LLM provider instance for generating patches
        """
        self.llm_provider = llm_provider
        self.patch_history = []

    def generate_patch(
        self,
        error_data: Dict[str, Any],
        source_code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PatchData]:
        """
        Generate a patch for the given error.

        Args:
            error_data: Error information
            source_code: Original source code
            context: Additional context for patch generation

        Returns:
            PatchData if successful, None otherwise
        """
        # This is a placeholder implementation
        # In a real implementation, this would use the LLM provider
        # to generate appropriate patches

        patch_id = f"patch_{len(self.patch_history) + 1}"

        # Simple example: if it's a syntax error, try to fix it
        if error_data.get("error_type") == "SyntaxError":
            patched_code = self._fix_syntax_error(source_code, error_data)
            if patched_code and patched_code != source_code:
                patch = PatchData(
                    patch_id=patch_id,
                    original_code=source_code,
                    patched_code=patched_code,
                    description="Fixed syntax error",
                    confidence=0.8,
                    error_type="SyntaxError",
                    fix_type="syntax_fix",
                )
                self.patch_history.append(patch)
                return patch

        return None

    def _fix_syntax_error(self, code: str, error_data: Dict[str, Any]) -> Optional[str]:
        """Simple syntax error fixer (placeholder)."""
        # This is a very basic implementation
        # Real implementation would use LLM or more sophisticated analysis

        error_msg = error_data.get("message", "").lower()

        # Fix missing colons
        if 'expected ":"' in error_msg or "missing colon" in error_msg:
            lines = code.splitlines()
            for i, line in enumerate(lines):
                if any(
                    keyword in line
                    for keyword in ["if ", "for ", "while ", "def ", "class "]
                ):
                    if not line.rstrip().endswith(":"):
                        lines[i] = line.rstrip() + ":"
            return "\n".join(lines)

        # Fix unclosed brackets
        if "unmatched" in error_msg or "unclosed" in error_msg:
            # Count brackets
            open_parens = code.count("(")
            close_parens = code.count(")")
            if open_parens > close_parens:
                return code + ")" * (open_parens - close_parens)

            open_brackets = code.count("[")
            close_brackets = code.count("]")
            if open_brackets > close_brackets:
                return code + "]" * (open_brackets - close_brackets)

            open_braces = code.count("{")
            close_braces = code.count("}")
            if open_braces > close_braces:
                return code + "}" * (open_braces - close_braces)

        return None
