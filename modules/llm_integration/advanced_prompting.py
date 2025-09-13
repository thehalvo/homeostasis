#!/usr/bin/env python3
"""
Advanced Prompting Module

This module implements advanced prompting strategies for Phase 13.3,
including chain-of-thought prompting, few-shot learning, and
context-aware prompt construction.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptingStrategy(Enum):
    """Different prompting strategies."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    LEAST_TO_MOST = "least_to_most"
    DECOMPOSITION = "decomposition"


@dataclass
class PromptExample:
    """Represents a few-shot example."""

    context: str
    problem: str
    solution: str
    explanation: str
    tags: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.0


@dataclass
class PromptTemplate:
    """Advanced prompt template with strategies."""

    name: str
    strategy: PromptingStrategy
    base_template: str
    examples: List[PromptExample] = field(default_factory=list)
    chain_of_thought_steps: List[str] = field(default_factory=list)
    decomposition_hints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


class AdvancedPromptManager:
    """
    Manages advanced prompting strategies for code generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced prompt manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.templates: Dict[str, PromptTemplate] = {}
        self.example_cache: Dict[str, Any] = {}
        self.prompt_history: List[Dict[str, Any]] = []

        # Configuration
        self.max_examples = self.config.get("max_examples", 5)
        self.enable_caching = self.config.get("enable_caching", True)
        self.adaptive_strategy = self.config.get("adaptive_strategy", True)

        # Initialize built-in templates
        self._initialize_templates()

        # Load learned examples
        self._load_learned_examples()

        logger.info("Initialized Advanced Prompt Manager")

    def create_prompt(
        self,
        task_type: str,
        context: Dict[str, Any],
        strategy: Optional[PromptingStrategy] = None,
    ) -> str:
        """
        Create an advanced prompt for the given task.

        Args:
            task_type: Type of task (e.g., 'bug_fix', 'refactoring')
            context: Task context including error info, code, etc.
            strategy: Specific strategy to use (auto-selected if None)

        Returns:
            Constructed prompt
        """
        # Auto-select strategy if not provided
        if strategy is None:
            strategy = self._select_strategy(task_type, context)

        # Get appropriate template
        template = self._get_template(task_type, strategy)

        # Build prompt using selected strategy
        if strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            prompt = self._build_chain_of_thought_prompt(template, context)
        elif strategy == PromptingStrategy.FEW_SHOT:
            prompt = self._build_few_shot_prompt(template, context)
        elif strategy == PromptingStrategy.TREE_OF_THOUGHT:
            prompt = self._build_tree_of_thought_prompt(template, context)
        elif strategy == PromptingStrategy.SELF_CONSISTENCY:
            prompt = self._build_self_consistency_prompt(template, context)
        elif strategy == PromptingStrategy.LEAST_TO_MOST:
            prompt = self._build_least_to_most_prompt(template, context)
        elif strategy == PromptingStrategy.DECOMPOSITION:
            prompt = self._build_decomposition_prompt(template, context)
        else:
            prompt = self._build_basic_prompt(template, context)

        # Add success criteria
        prompt = self._add_success_criteria(prompt, template)

        # Cache prompt for learning
        self._cache_prompt(task_type, strategy, context, prompt)

        return prompt

    def add_learned_example(
        self, task_type: str, example: PromptExample, effectiveness_score: float
    ):
        """
        Add a learned example from successful code generation.

        Args:
            task_type: Type of task the example is for
            example: The example to add
            effectiveness_score: How effective this example was (0-1)
        """
        example.effectiveness_score = effectiveness_score

        # Add to appropriate templates
        for template in self.templates.values():
            if task_type in template.name:
                template.examples.append(example)

                # Keep only the best examples
                template.examples.sort(
                    key=lambda x: x.effectiveness_score, reverse=True
                )
                template.examples = template.examples[: self.max_examples * 2]

        # Save to persistent storage
        self._save_learned_examples()

    def _initialize_templates(self):
        """Initialize built-in prompt templates."""
        # Bug fix template with chain-of-thought
        self.templates["bug_fix_cot"] = PromptTemplate(
            name="bug_fix",
            strategy=PromptingStrategy.CHAIN_OF_THOUGHT,
            base_template="""You are an expert software engineer tasked with fixing a bug.

## Problem Context:
{context}

## Your Task:
Fix the bug by following these steps:
{chain_of_thought}

## Requirements:
{requirements}""",
            chain_of_thought_steps=[
                "1. Understand the error: What exactly is going wrong?",
                "2. Identify root cause: Why is this error occurring?",
                "3. Consider side effects: What else might this error affect?",
                "4. Design the fix: What's the minimal change needed?",
                "5. Verify correctness: Will this fix solve the problem without breaking anything?",
                "6. Check edge cases: Are there any special cases to consider?",
            ],
            success_criteria=[
                "The error is completely resolved",
                "No new errors are introduced",
                "Code style is preserved",
                "The fix is minimal and focused",
            ],
        )

        # Refactoring template with decomposition
        self.templates["refactoring_decomp"] = PromptTemplate(
            name="refactoring",
            strategy=PromptingStrategy.DECOMPOSITION,
            base_template="""You are refactoring code to improve its quality.

## Code to Refactor:
{code}

## Refactoring Goals:
{goals}

## Decomposition Approach:
{decomposition}

## Constraints:
{constraints}""",
            decomposition_hints=[
                "Break down the refactoring into atomic changes",
                "Identify independent improvements",
                "Order changes by dependency",
                "Group related changes together",
                "Consider testability at each step",
            ],
            success_criteria=[
                "Functionality is preserved",
                "Code quality metrics improve",
                "Tests continue to pass",
                "Performance is maintained or improved",
            ],
        )

        # Feature addition with few-shot
        self.templates["feature_addition_few_shot"] = PromptTemplate(
            name="feature_addition",
            strategy=PromptingStrategy.FEW_SHOT,
            base_template="""Add a new feature to the codebase.

## Feature Request:
{feature_description}

## Current Code:
{current_code}

## Examples of Similar Features:
{examples}

## Your Implementation:
Based on the examples above, implement the requested feature following the same patterns and style.""",
            examples=[],  # Will be populated with learned examples
            success_criteria=[
                "Feature works as specified",
                "Integrates well with existing code",
                "Follows established patterns",
                "Includes appropriate error handling",
            ],
        )

        # Multi-file coordination with tree-of-thought
        self.templates["multi_file_tot"] = PromptTemplate(
            name="multi_file_coordination",
            strategy=PromptingStrategy.TREE_OF_THOUGHT,
            base_template="""Coordinate changes across multiple files.

## Change Overview:
{change_description}

## Affected Files:
{file_list}

## Exploration Tree:
Consider multiple approaches and their implications:
{tree_of_thought}

## Coordination Requirements:
{coordination_requirements}""",
            chain_of_thought_steps=[
                "Map dependencies between files",
                "Identify shared interfaces",
                "Plan change order",
                "Consider backward compatibility",
                "Verify consistency across files",
            ],
            success_criteria=[
                "All files remain consistent",
                "Interfaces are properly updated",
                "No circular dependencies introduced",
                "Changes are atomic and complete",
            ],
        )

    def _select_strategy(
        self, task_type: str, context: Dict[str, Any]
    ) -> PromptingStrategy:
        """Auto-select the best prompting strategy."""
        if not self.adaptive_strategy:
            # Use default strategies
            default_strategies = {
                "bug_fix": PromptingStrategy.CHAIN_OF_THOUGHT,
                "refactoring": PromptingStrategy.DECOMPOSITION,
                "feature_addition": PromptingStrategy.FEW_SHOT,
                "multi_file": PromptingStrategy.TREE_OF_THOUGHT,
                "optimization": PromptingStrategy.SELF_CONSISTENCY,
                "complex_logic": PromptingStrategy.LEAST_TO_MOST,
            }
            return default_strategies.get(task_type, PromptingStrategy.CHAIN_OF_THOUGHT)

        # Adaptive selection based on context
        complexity_score = self._assess_complexity(context)

        if complexity_score > 0.8:
            # Very complex - use tree of thought
            return PromptingStrategy.TREE_OF_THOUGHT
        elif complexity_score > 0.6:
            # Complex - use decomposition or least-to-most
            if "multiple_steps" in context:
                return PromptingStrategy.LEAST_TO_MOST
            else:
                return PromptingStrategy.DECOMPOSITION
        elif self._has_good_examples(task_type):
            # Have good examples - use few-shot
            return PromptingStrategy.FEW_SHOT
        elif "logical_reasoning" in context:
            # Needs reasoning - use chain of thought
            return PromptingStrategy.CHAIN_OF_THOUGHT
        else:
            # Default to self-consistency for reliability
            return PromptingStrategy.SELF_CONSISTENCY

    def _assess_complexity(self, context: Dict[str, Any]) -> float:
        """Assess the complexity of the task."""
        complexity_factors = 0.0

        # File count
        if context.get("file_count", 1) > 1:
            complexity_factors += 0.2

        # Line count
        lines = context.get("affected_lines", 0)
        if lines > 100:
            complexity_factors += 0.2
        elif lines > 50:
            complexity_factors += 0.1

        # Error complexity
        if "multiple_errors" in context:
            complexity_factors += 0.2

        # Dependency count
        deps = len(context.get("dependencies", []))
        if deps > 5:
            complexity_factors += 0.2
        elif deps > 2:
            complexity_factors += 0.1

        # Cross-cutting concerns
        if context.get("cross_cutting", False):
            complexity_factors += 0.2

        return min(1.0, complexity_factors)

    def _has_good_examples(self, task_type: str) -> bool:
        """Check if we have good examples for few-shot learning."""
        for template in self.templates.values():
            if task_type in template.name:
                good_examples = [
                    e for e in template.examples if e.effectiveness_score > 0.7
                ]
                return len(good_examples) >= 3
        return False

    def _get_template(
        self, task_type: str, strategy: PromptingStrategy
    ) -> PromptTemplate:
        """Get the appropriate template."""
        # Look for exact match
        template_key = f"{task_type}_{strategy.value}"
        if template_key in self.templates:
            return self.templates[template_key]

        # Look for task type match
        for key, template in self.templates.items():
            if task_type in key and template.strategy == strategy:
                return template

        # Look for strategy match
        for template in self.templates.values():
            if template.strategy == strategy:
                return template

        # Create default template
        return PromptTemplate(
            name=task_type,
            strategy=strategy,
            base_template="Perform the following task:\n{context}\n\nRequirements:\n{requirements}",
            success_criteria=[
                "Complete the task successfully",
                "Maintain code quality",
            ],
        )

    def _build_chain_of_thought_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a chain-of-thought prompt."""
        # Format the chain of thought steps
        cot_steps = "\n".join(template.chain_of_thought_steps)

        # Build the prompt
        prompt = template.base_template.format(
            context=self._format_context(context),
            chain_of_thought=cot_steps,
            requirements=self._format_requirements(context),
        )

        # Add reasoning instructions
        prompt += """

## Step-by-Step Reasoning:
For each step above, provide your reasoning before moving to the next step.
Show your work and explain your thought process.

## Final Solution:
After completing all reasoning steps, provide your final solution with confidence level."""

        return prompt

    def _build_few_shot_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a few-shot learning prompt."""
        # Select best examples
        examples = self._select_best_examples(template, context)

        # Format examples
        formatted_examples = self._format_examples(examples)

        # Build the prompt
        prompt = template.base_template.format(
            feature_description=context.get("description", ""),
            current_code=context.get("code", ""),
            examples=formatted_examples,
        )

        # Add pattern recognition guidance
        prompt += """

## Pattern Analysis:
Based on the examples above, identify:
1. Common patterns and conventions used
2. Error handling approaches
3. Code organization principles
4. Testing strategies

Apply these patterns to your solution."""

        return prompt

    def _build_tree_of_thought_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a tree-of-thought prompt."""
        # Create exploration branches
        branches = [
            "Approach A: Minimal change strategy",
            "Approach B: Refactor and improve",
            "Approach C: Complete redesign",
        ]

        tree_structure = """
1. Explore each approach:
   {}
   
2. For each approach, consider:
   - Pros and cons
   - Implementation complexity
   - Risk assessment
   - Long-term maintainability
   
3. Compare approaches and select the best one

4. Implement the selected approach
""".format(
            "\n   ".join(branches)
        )

        # Build the prompt
        prompt = template.base_template.format(
            change_description=context.get("description", ""),
            file_list=self._format_file_list(context.get("files", [])),
            tree_of_thought=tree_structure,
            coordination_requirements=context.get("requirements", ""),
        )

        return prompt

    def _build_self_consistency_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a self-consistency prompt."""
        prompt = self._build_basic_prompt(template, context)

        # Add self-consistency instructions
        prompt += """

## Self-Consistency Check:
After providing your solution:
1. Verify it solves the stated problem
2. Check for any contradictions
3. Ensure all requirements are met
4. Validate edge cases are handled

If any issues are found, revise your solution."""

        return prompt

    def _build_least_to_most_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a least-to-most prompt."""
        # Decompose problem into sub-problems
        sub_problems = self._decompose_problem(context)

        prompt = f"""Solve this problem using a least-to-most approach.

## Main Problem:
{context.get('description', '')}

## Sub-problems (solve in order):
"""

        for i, sub_problem in enumerate(sub_problems, 1):
            prompt += f"\n{i}. {sub_problem}"

        prompt += """

## Instructions:
1. Solve each sub-problem in order
2. Use the solution of earlier sub-problems to solve later ones
3. Combine all solutions to solve the main problem

## Final Solution:
Provide the complete solution that addresses the main problem."""

        return prompt

    def _build_decomposition_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a decomposition prompt."""
        # Format decomposition hints
        decomp_hints = "\n".join(f"- {hint}" for hint in template.decomposition_hints)

        # Build the prompt
        prompt = template.base_template.format(
            code=context.get("code", ""),
            goals=context.get("goals", ""),
            decomposition=decomp_hints,
            constraints=context.get("constraints", ""),
        )

        # Add decomposition instructions
        prompt += """

## Decomposition Process:
1. Identify all atomic changes needed
2. Group related changes
3. Order by dependencies
4. Implement each group separately
5. Verify integration

## Output Format:
Provide each change as a separate, clearly labeled section."""

        return prompt

    def _build_basic_prompt(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> str:
        """Build a basic prompt as fallback."""
        # Use template with simple substitution
        prompt_vars = {
            "context": self._format_context(context),
            "requirements": self._format_requirements(context),
            "code": context.get("code", ""),
            "description": context.get("description", ""),
            "constraints": context.get("constraints", ""),
        }

        # Safe format - only replace variables that exist in template
        prompt = template.base_template
        for key, value in prompt_vars.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))

        return prompt

    def _add_success_criteria(self, prompt: str, template: PromptTemplate) -> str:
        """Add success criteria to the prompt."""
        if not template.success_criteria:
            return prompt

        criteria_text = "\n\n## Success Criteria:\n"
        for criterion in template.success_criteria:
            criteria_text += f"✓ {criterion}\n"

        return prompt + criteria_text

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for inclusion in prompt."""
        formatted_parts = []

        if "error_type" in context:
            formatted_parts.append(f"Error Type: {context['error_type']}")

        if "error_message" in context:
            formatted_parts.append(f"Error Message: {context['error_message']}")

        if "file_path" in context:
            formatted_parts.append(f"File: {context['file_path']}")

        if "line_number" in context:
            formatted_parts.append(f"Line: {context['line_number']}")

        if "description" in context:
            formatted_parts.append(f"Description: {context['description']}")

        return "\n".join(formatted_parts)

    def _format_requirements(self, context: Dict[str, Any]) -> str:
        """Format requirements for inclusion in prompt."""
        requirements = context.get("requirements", [])

        if isinstance(requirements, list):
            return "\n".join(f"- {req}" for req in requirements)
        else:
            return str(requirements)

    def _format_file_list(self, files: List[str]) -> str:
        """Format file list for inclusion in prompt."""
        if not files:
            return "No files specified"

        return "\n".join(f"- {file}" for file in files)

    def _select_best_examples(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> List[PromptExample]:
        """Select the best examples for few-shot learning."""
        if not template.examples:
            return []

        # Score examples by relevance
        scored_examples = []
        for example in template.examples:
            score = self._score_example_relevance(example, context)
            scored_examples.append((score, example))

        # Sort by score and effectiveness
        scored_examples.sort(
            key=lambda x: (x[0], x[1].effectiveness_score), reverse=True
        )

        # Return top examples
        return [ex[1] for ex in scored_examples[: self.max_examples]]

    def _score_example_relevance(
        self, example: PromptExample, context: Dict[str, Any]
    ) -> float:
        """Score how relevant an example is to the current context."""
        score = 0.0

        # Check tag overlap
        context_tags = set(context.get("tags", []))
        example_tags = set(example.tags)
        if context_tags and example_tags:
            overlap = len(context_tags & example_tags) / len(
                context_tags | example_tags
            )
            score += overlap * 0.3

        # Check problem similarity (simple text similarity)
        if "error_type" in context and context["error_type"] in example.problem:
            score += 0.2

        if "language" in context and context["language"] in example.tags:
            score += 0.2

        # Check solution complexity similarity
        context_complexity = self._assess_complexity(context)
        example_complexity = len(example.solution) / 1000.0  # Simple proxy
        complexity_diff = abs(context_complexity - min(1.0, example_complexity))
        score += (1.0 - complexity_diff) * 0.3

        return score

    def _format_examples(self, examples: List[PromptExample]) -> str:
        """Format examples for inclusion in prompt."""
        if not examples:
            return "No examples available."

        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(
                f"""
### Example {i}:
**Context**: {example.context}
**Problem**: {example.problem}
**Solution**:
```
{example.solution}
```
**Explanation**: {example.explanation}
"""
            )

        return "\n".join(formatted)

    def _decompose_problem(self, context: Dict[str, Any]) -> List[str]:
        """Decompose a problem into sub-problems."""
        # This is a simplified decomposition
        sub_problems = []

        # Based on error type
        error_type = context.get("error_type", "")

        if "NameError" in error_type:
            sub_problems.extend(
                [
                    "Identify where the undefined name is used",
                    "Determine what the name should refer to",
                    "Add the necessary import or definition",
                ]
            )
        elif "TypeError" in error_type:
            sub_problems.extend(
                [
                    "Identify the types involved in the error",
                    "Determine the expected types",
                    "Add type conversion or change the logic",
                ]
            )
        elif "AttributeError" in error_type:
            sub_problems.extend(
                [
                    "Identify the object and attribute",
                    "Check if the attribute exists",
                    "Add the attribute or use an alternative",
                ]
            )
        else:
            # Generic decomposition
            sub_problems.extend(
                [
                    "Understand the root cause",
                    "Identify affected components",
                    "Design the minimal fix",
                    "Verify the fix",
                ]
            )

        return sub_problems

    def _cache_prompt(
        self,
        task_type: str,
        strategy: PromptingStrategy,
        context: Dict[str, Any],
        prompt: str,
    ):
        """Cache prompt for learning and analysis."""
        if not self.enable_caching:
            return

        # Create cache key
        context_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()[:16]

        cache_entry = {
            "task_type": task_type,
            "strategy": strategy.value,
            "context_hash": context_hash,
            "prompt": prompt,
            "timestamp": Path(__file__).stat().st_mtime,
        }

        self.prompt_history.append(cache_entry)

        # Limit history size
        if len(self.prompt_history) > 1000:
            self.prompt_history = self.prompt_history[-1000:]

    def _load_learned_examples(self):
        """Load learned examples from storage."""
        # This would load from a persistent store
        # For now, we'll just initialize with some hardcoded examples

        # Python bug fix examples
        python_examples = [
            PromptExample(
                context="Python web application",
                problem="NameError: name 'request' is not defined in Flask route",
                solution="from flask import request",
                explanation="Import the request object from Flask when using it in route handlers",
                tags=["python", "flask", "import", "web"],
                effectiveness_score=0.9,
            ),
            PromptExample(
                context="Data processing script",
                problem="TypeError: unsupported operand type(s) for +: 'int' and 'str'",
                solution="total = int(value1) + int(value2)  # Convert strings to int before addition",
                explanation="Ensure compatible types for arithmetic operations",
                tags=["python", "type_error", "conversion"],
                effectiveness_score=0.85,
            ),
        ]

        # Add to templates
        for template in self.templates.values():
            if "bug_fix" in template.name or "feature" in template.name:
                template.examples.extend(python_examples)

    def _save_learned_examples(self):
        """Save learned examples to persistent storage."""
        # This would save to a database or file
        # For now, just log
        total_examples = sum(len(t.examples) for t in self.templates.values())
        logger.info(f"Saved {total_examples} learned examples")


def create_advanced_prompt_manager(
    config: Optional[Dict[str, Any]] = None,
) -> AdvancedPromptManager:
    """
    Factory function to create an advanced prompt manager.

    Args:
        config: Configuration dictionary

    Returns:
        Configured prompt manager
    """
    return AdvancedPromptManager(config)
