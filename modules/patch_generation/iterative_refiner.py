#!/usr/bin/env python3
"""
Iterative Patch Refiner

This module implements iterative refinement of code patches based on
validation feedback, test results, and progressive improvement strategies.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile

from ..llm_integration.provider_abstraction import (
    LLMManager, LLMRequest, LLMMessage
)
from ..analysis.llm_context_manager import LLMContextManager
from .code_style_analyzer import CodeStyleAnalyzer

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Different strategies for patch refinement."""
    TEST_DRIVEN = "test_driven"
    ERROR_FOCUSED = "error_focused"
    PERFORMANCE_ORIENTED = "performance_oriented"
    STYLE_COMPLIANCE = "style_compliance"
    SEMANTIC_PRESERVATION = "semantic_preservation"


@dataclass
class RefinementIteration:
    """Represents a single refinement iteration."""
    iteration_number: int
    strategy: RefinementStrategy
    changes_made: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    improvement_metrics: Dict[str, float]
    feedback_incorporated: List[str]
    success: bool
    
    
@dataclass
class RefinementHistory:
    """Tracks the history of refinement iterations."""
    original_patch: Dict[str, Any]
    iterations: List[RefinementIteration] = field(default_factory=list)
    final_patch: Optional[Dict[str, Any]] = None
    total_time: float = 0.0
    convergence_achieved: bool = False
    best_iteration: Optional[int] = None
    

@dataclass
class ValidationFeedback:
    """Feedback from validation processes."""
    test_results: Dict[str, Any]
    static_analysis: Dict[str, Any]
    style_violations: List[Dict[str, Any]]
    semantic_issues: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    suggestions: List[str]
    

class IterativeRefiner:
    """
    Implements iterative refinement of code patches with learning capabilities.
    """
    
    def __init__(self,
                 llm_manager: LLMManager,
                 context_manager: LLMContextManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the iterative refiner.
        
        Args:
            llm_manager: LLM manager for refinement generation
            context_manager: Context manager for tracking iterations
            config: Configuration dictionary
        """
        self.llm_manager = llm_manager
        self.context_manager = context_manager
        self.style_analyzer = CodeStyleAnalyzer()
        
        # Configuration
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 5)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.95)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.enable_parallel_strategies = self.config.get('enable_parallel_strategies', False)
        self.test_timeout = self.config.get('test_timeout', 300)  # 5 minutes
        
        # Strategy weights (can be learned over time)
        self.strategy_weights = {
            RefinementStrategy.TEST_DRIVEN: 0.3,
            RefinementStrategy.ERROR_FOCUSED: 0.3,
            RefinementStrategy.PERFORMANCE_ORIENTED: 0.1,
            RefinementStrategy.STYLE_COMPLIANCE: 0.2,
            RefinementStrategy.SEMANTIC_PRESERVATION: 0.1
        }
        
        # Refinement patterns learned from successful iterations
        self.learned_patterns = []
        
        logger.info("Initialized Iterative Refiner")
    
    def refine_patch(self,
                    initial_patch: Dict[str, Any],
                    validation_feedback: ValidationFeedback,
                    target_file: str,
                    original_content: str,
                    test_command: Optional[str] = None) -> RefinementHistory:
        """
        Iteratively refine a patch based on validation feedback.
        
        Args:
            initial_patch: Initial patch to refine
            validation_feedback: Initial validation feedback
            target_file: Target file path
            original_content: Original file content
            test_command: Command to run tests
            
        Returns:
            Refinement history with final patch
        """
        history = RefinementHistory(original_patch=initial_patch)
        current_patch = initial_patch.copy()
        current_feedback = validation_feedback
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            logger.info(f"Starting refinement iteration {iteration + 1}")
            
            # Select refinement strategy
            strategy = self._select_refinement_strategy(
                current_feedback, history
            )
            
            # Generate refinement
            refined_patch = self._generate_refinement(
                current_patch, current_feedback, strategy, history
            )
            
            # Apply and validate refinement
            validation_results = self._validate_refinement(
                refined_patch, target_file, original_content, test_command
            )
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                current_feedback, validation_results
            )
            
            # Create iteration record
            iteration_record = RefinementIteration(
                iteration_number=iteration + 1,
                strategy=strategy,
                changes_made=self._extract_changes(current_patch, refined_patch),
                validation_results=validation_results,
                improvement_metrics=improvement_metrics,
                feedback_incorporated=self._extract_feedback_items(current_feedback),
                success=self._is_validation_successful(validation_results)
            )
            
            history.iterations.append(iteration_record)
            
            # Check for convergence
            if self._has_converged(improvement_metrics):
                logger.info(f"Convergence achieved at iteration {iteration + 1}")
                history.convergence_achieved = True
                history.final_patch = refined_patch
                break
            
            # Update for next iteration
            current_patch = refined_patch
            current_feedback = self._create_feedback_from_validation(validation_results)
            
            # Update strategy weights based on results
            self._update_strategy_weights(strategy, improvement_metrics)
        
        # Select best iteration if no convergence
        if not history.convergence_achieved:
            history.best_iteration = self._select_best_iteration(history)
            if history.best_iteration is not None:
                history.final_patch = self._reconstruct_patch_at_iteration(
                    initial_patch, history, history.best_iteration
                )
            else:
                history.final_patch = current_patch
        
        history.total_time = time.time() - start_time
        
        # Learn from this refinement session
        self._learn_from_refinement(history)
        
        return history
    
    def _select_refinement_strategy(self,
                                  feedback: ValidationFeedback,
                                  history: RefinementHistory) -> RefinementStrategy:
        """Select the best refinement strategy based on feedback and history."""
        scores = {}
        
        # Score each strategy based on current needs
        if feedback.test_results.get('failed_tests', 0) > 0:
            scores[RefinementStrategy.TEST_DRIVEN] = (
                self.strategy_weights[RefinementStrategy.TEST_DRIVEN] * 2.0
            )
        
        if feedback.semantic_issues:
            scores[RefinementStrategy.SEMANTIC_PRESERVATION] = (
                self.strategy_weights[RefinementStrategy.SEMANTIC_PRESERVATION] * 1.5
            )
        
        if feedback.style_violations:
            scores[RefinementStrategy.STYLE_COMPLIANCE] = (
                self.strategy_weights[RefinementStrategy.STYLE_COMPLIANCE] * 1.3
            )
        
        if feedback.test_results.get('errors', []):
            scores[RefinementStrategy.ERROR_FOCUSED] = (
                self.strategy_weights[RefinementStrategy.ERROR_FOCUSED] * 1.8
            )
        
        if feedback.performance_metrics.get('regression', False):
            scores[RefinementStrategy.PERFORMANCE_ORIENTED] = (
                self.strategy_weights[RefinementStrategy.PERFORMANCE_ORIENTED] * 1.5
            )
        
        # Add default scores for strategies not yet scored
        for strategy in RefinementStrategy:
            if strategy not in scores:
                scores[strategy] = self.strategy_weights[strategy]
        
        # Avoid repeating the same strategy too often
        if history.iterations:
            last_strategy = history.iterations[-1].strategy
            scores[last_strategy] *= 0.7
        
        # Select strategy with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _generate_refinement(self,
                           current_patch: Dict[str, Any],
                           feedback: ValidationFeedback,
                           strategy: RefinementStrategy,
                           history: RefinementHistory) -> Dict[str, Any]:
        """Generate a refined patch using the selected strategy."""
        # Build refinement prompt
        prompt = self._build_refinement_prompt(
            current_patch, feedback, strategy, history
        )
        
        # Add learned patterns to prompt
        if self.learned_patterns:
            prompt += self._format_learned_patterns(strategy)
        
        # Generate refinement
        request = LLMRequest(
            messages=[LLMMessage(role='user', content=prompt)],
            temperature=0.2,  # Low temperature for consistency
            max_tokens=3000
        )
        
        try:
            response = self.llm_manager.complete(request)
            refined_patch = self._parse_refinement_response(
                response.content, current_patch
            )
            
            # Apply strategy-specific post-processing
            refined_patch = self._apply_strategy_specific_processing(
                refined_patch, strategy, feedback
            )
            
            return refined_patch
            
        except Exception as e:
            logger.error(f"Error generating refinement: {e}")
            # Return current patch if refinement fails
            return current_patch
    
    def _build_refinement_prompt(self,
                                current_patch: Dict[str, Any],
                                feedback: ValidationFeedback,
                                strategy: RefinementStrategy,
                                history: RefinementHistory) -> str:
        """Build a prompt for patch refinement."""
        prompt_parts = []
        
        # Strategy-specific instructions
        strategy_instructions = {
            RefinementStrategy.TEST_DRIVEN: """
Focus on making all tests pass. Analyze test failures and adjust the patch to fix them.
Priority: Fix failing tests while maintaining existing functionality.""",
            
            RefinementStrategy.ERROR_FOCUSED: """
Focus on resolving all errors and exceptions. Ensure the patch doesn't introduce new errors.
Priority: Error-free execution.""",
            
            RefinementStrategy.PERFORMANCE_ORIENTED: """
Focus on performance optimization. Ensure the patch doesn't degrade performance.
Priority: Maintain or improve execution speed and resource usage.""",
            
            RefinementStrategy.STYLE_COMPLIANCE: """
Focus on code style compliance. Fix all style violations while preserving functionality.
Priority: Clean, consistent, idiomatic code.""",
            
            RefinementStrategy.SEMANTIC_PRESERVATION: """
Focus on preserving semantic correctness. Ensure the patch maintains the intended behavior.
Priority: Correct program semantics and logic preservation."""
        }
        
        prompt_parts.append(f"Refine the following patch using a {strategy.value} strategy.")
        prompt_parts.append(strategy_instructions[strategy])
        
        # Current patch
        prompt_parts.append(f"""
CURRENT PATCH:
{json.dumps(current_patch, indent=2)}""")
        
        # Feedback details
        prompt_parts.append(f"""
VALIDATION FEEDBACK:
Test Results: {json.dumps(feedback.test_results, indent=2)}
Style Violations: {json.dumps(feedback.style_violations, indent=2)}
Semantic Issues: {json.dumps(feedback.semantic_issues, indent=2)}
Performance Metrics: {json.dumps(feedback.performance_metrics, indent=2)}

SPECIFIC ISSUES TO ADDRESS:""")
        
        # Add specific issues based on strategy
        if strategy == RefinementStrategy.TEST_DRIVEN:
            failed_tests = feedback.test_results.get('failed_tests', [])
            for test in failed_tests[:5]:  # Limit to 5 tests
                prompt_parts.append(f"- Test '{test['name']}' failed: {test['error']}")
        
        elif strategy == RefinementStrategy.ERROR_FOCUSED:
            errors = feedback.test_results.get('errors', [])
            for error in errors[:5]:
                prompt_parts.append(f"- Error: {error}")
        
        elif strategy == RefinementStrategy.STYLE_COMPLIANCE:
            for violation in feedback.style_violations[:5]:
                prompt_parts.append(f"- Style: {violation['message']} at line {violation['line']}")
        
        # History context
        if history.iterations:
            prompt_parts.append(f"""
REFINEMENT HISTORY:
- Iterations completed: {len(history.iterations)}
- Strategies tried: {[it.strategy.value for it in history.iterations]}
- Best improvement: {max(it.improvement_metrics.get('overall', 0) for it in history.iterations):.2f}""")
        
        # Suggestions from feedback
        if feedback.suggestions:
            prompt_parts.append("\nSUGGESTIONS:")
            for suggestion in feedback.suggestions[:3]:
                prompt_parts.append(f"- {suggestion}")
        
        # Output format
        prompt_parts.append("""
Generate a refined patch that addresses the issues while maintaining correctness.

OUTPUT FORMAT:
```json
{
    "changes": [
        {
            "line_start": <number>,
            "line_end": <number>,
            "original_code": "exact original code",
            "new_code": "refined replacement code",
            "refinement_reason": "specific reason for this refinement"
        }
    ],
    "refinement_summary": "summary of refinements made",
    "expected_improvements": ["list of expected improvements"],
    "confidence": 0.0-1.0
}
```""")
        
        return "\n".join(prompt_parts)
    
    def _validate_refinement(self,
                           patch: Dict[str, Any],
                           target_file: str,
                           original_content: str,
                           test_command: Optional[str]) -> Dict[str, Any]:
        """Validate a refined patch."""
        validation_results = {
            'tests_passed': False,
            'style_compliant': False,
            'no_errors': False,
            'performance_acceptable': True,
            'semantic_preserved': True,
            'details': {}
        }
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix=Path(target_file).suffix, delete=False) as tmp:
            # Apply patch to content
            patched_content = self._apply_patch_to_content(
                original_content, patch.get('changes', [])
            )
            tmp.write(patched_content)
            tmp_path = tmp.name
        
        try:
            # Run tests if command provided
            if test_command:
                test_results = self._run_tests(test_command, tmp_path, target_file)
                validation_results['tests_passed'] = test_results['success']
                validation_results['details']['test_results'] = test_results
            
            # Check style compliance
            style_results = self._check_style_compliance(tmp_path, patched_content)
            validation_results['style_compliant'] = len(style_results) == 0
            validation_results['details']['style_violations'] = style_results
            
            # Check for syntax errors
            syntax_results = self._check_syntax(tmp_path, target_file)
            validation_results['no_errors'] = syntax_results['valid']
            validation_results['details']['syntax_errors'] = syntax_results.get('errors', [])
            
            # Simple semantic check (could be enhanced)
            semantic_results = self._check_semantic_preservation(
                original_content, patched_content, patch
            )
            validation_results['semantic_preserved'] = semantic_results['preserved']
            validation_results['details']['semantic_analysis'] = semantic_results
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
        
        return validation_results
    
    def _apply_patch_to_content(self, content: str, changes: List[Dict[str, Any]]) -> str:
        """Apply patch changes to content."""
        lines = content.split('\n')
        
        # Sort changes by line number in reverse to avoid offset issues
        sorted_changes = sorted(changes, key=lambda x: x.get('line_start', 0), reverse=True)
        
        for change in sorted_changes:
            start = change.get('line_start', 1) - 1
            end = change.get('line_end', start + 1) - 1
            new_lines = change.get('new_code', '').split('\n')
            
            # Apply change
            lines[start:end + 1] = new_lines
        
        return '\n'.join(lines)
    
    def _run_tests(self, test_command: str, patched_file: str, original_file: str) -> Dict[str, Any]:
        """Run tests with the patched file."""
        # Replace original file path with patched file in command
        modified_command = test_command.replace(original_file, patched_file)
        
        try:
            result = subprocess.run(
                modified_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.test_timeout
            )
            
            # Parse test output (simplified)
            failed_tests = []
            errors = []
            
            if result.returncode != 0:
                # Extract failed test information
                output = result.stdout + result.stderr
                
                # Common test failure patterns
                test_patterns = [
                    r'FAILED.*?(\w+::\w+)',  # pytest
                    r'FAIL:\s*(\w+)',  # unittest
                    r'✗\s+(\w+)',  # jest
                ]
                
                for pattern in test_patterns:
                    import re
                    matches = re.findall(pattern, output)
                    for match in matches:
                        failed_tests.append({
                            'name': match,
                            'error': 'Test failed'  # Could extract more details
                        })
                
                # Extract errors
                if 'Error' in output or 'Exception' in output:
                    error_lines = [line for line in output.split('\n') if 'Error' in line or 'Exception' in line]
                    errors.extend(error_lines[:5])  # Limit to 5 errors
            
            return {
                'success': result.returncode == 0,
                'failed_tests': failed_tests,
                'errors': errors,
                'output': result.stdout,
                'error_output': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'failed_tests': [],
                'errors': ['Test execution timed out'],
                'output': '',
                'error_output': 'Timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'failed_tests': [],
                'errors': [str(e)],
                'output': '',
                'error_output': str(e)
            }
    
    def _check_style_compliance(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Check style compliance of the patched code."""
        violations = []
        
        # Use the style analyzer
        language = self._detect_language(file_path)
        style_conventions = self.style_analyzer.analyze_file_style(
            file_path, language, content
        )
        
        # Check against conventions (simplified)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Line length
            if len(line) > style_conventions.line_length:
                violations.append({
                    'line': i + 1,
                    'message': f'Line too long ({len(line)} > {style_conventions.line_length})',
                    'severity': 'warning'
                })
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                violations.append({
                    'line': i + 1,
                    'message': 'Trailing whitespace',
                    'severity': 'warning'
                })
        
        return violations
    
    def _check_syntax(self, file_path: str, original_path: str) -> Dict[str, Any]:
        """Check syntax validity of the patched code."""
        language = self._detect_language(original_path)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if language == 'python':
                import ast
                ast.parse(content)
                return {'valid': True}
            
            elif language in ['javascript', 'typescript']:
                # Could use a proper parser, but for now just check basic syntax
                # by looking for common syntax errors
                if content.count('{') != content.count('}'):
                    return {'valid': False, 'errors': ['Mismatched braces']}
                if content.count('(') != content.count(')'):
                    return {'valid': False, 'errors': ['Mismatched parentheses']}
                return {'valid': True}
            
            else:
                # For other languages, assume valid if no obvious issues
                return {'valid': True}
                
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [f'Syntax error at line {e.lineno}: {e.msg}']
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Error checking syntax: {str(e)}']
            }
    
    def _check_semantic_preservation(self,
                                   original: str,
                                   patched: str,
                                   patch: Dict[str, Any]) -> Dict[str, Any]:
        """Check if semantic meaning is preserved."""
        # This is a simplified check - in production, use more sophisticated analysis
        
        # Check if critical structures are preserved
        preserved = True
        issues = []
        
        # Check function signatures haven't changed unexpectedly
        import re
        original_funcs = re.findall(r'def\s+(\w+)\s*\([^)]*\)', original)
        patched_funcs = re.findall(r'def\s+(\w+)\s*\([^)]*\)', patched)
        
        removed_funcs = set(original_funcs) - set(patched_funcs)
        if removed_funcs:
            preserved = False
            issues.append(f"Functions removed: {removed_funcs}")
        
        # Check class definitions
        original_classes = re.findall(r'class\s+(\w+)', original)
        patched_classes = re.findall(r'class\s+(\w+)', patched)
        
        removed_classes = set(original_classes) - set(patched_classes)
        if removed_classes:
            preserved = False
            issues.append(f"Classes removed: {removed_classes}")
        
        return {
            'preserved': preserved,
            'issues': issues,
            'confidence': 0.8 if preserved else 0.3
        }
    
    def _calculate_improvement_metrics(self,
                                     old_feedback: ValidationFeedback,
                                     new_validation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate improvement metrics between iterations."""
        metrics = {}
        
        # Test improvement
        old_failed = len(old_feedback.test_results.get('failed_tests', []))
        new_failed = len(new_validation.get('details', {}).get('test_results', {}).get('failed_tests', []))
        
        if old_failed > 0:
            metrics['test_improvement'] = (old_failed - new_failed) / old_failed
        else:
            metrics['test_improvement'] = 1.0 if new_failed == 0 else 0.0
        
        # Style improvement
        old_violations = len(old_feedback.style_violations)
        new_violations = len(new_validation.get('details', {}).get('style_violations', []))
        
        if old_violations > 0:
            metrics['style_improvement'] = (old_violations - new_violations) / old_violations
        else:
            metrics['style_improvement'] = 1.0 if new_violations == 0 else 0.0
        
        # Error improvement
        old_errors = len(old_feedback.test_results.get('errors', []))
        new_errors = len(new_validation.get('details', {}).get('syntax_errors', []))
        
        if old_errors > 0:
            metrics['error_improvement'] = (old_errors - new_errors) / old_errors
        else:
            metrics['error_improvement'] = 1.0 if new_errors == 0 else 0.0
        
        # Overall improvement
        weights = {
            'test_improvement': 0.5,
            'style_improvement': 0.2,
            'error_improvement': 0.3
        }
        
        metrics['overall'] = sum(
            metrics.get(key, 0) * weight
            for key, weight in weights.items()
        )
        
        return metrics
    
    def _is_validation_successful(self, validation_results: Dict[str, Any]) -> bool:
        """Check if validation is successful."""
        return (
            validation_results.get('tests_passed', False) and
            validation_results.get('no_errors', False) and
            validation_results.get('semantic_preserved', True)
        )
    
    def _has_converged(self, improvement_metrics: Dict[str, float]) -> bool:
        """Check if refinement has converged."""
        overall_improvement = improvement_metrics.get('overall', 0)
        
        # Converged if overall improvement is above threshold
        # or if we've achieved perfect scores
        return (
            overall_improvement >= self.convergence_threshold or
            all(
                improvement_metrics.get(key, 0) >= 0.99
                for key in ['test_improvement', 'error_improvement']
            )
        )
    
    def _create_feedback_from_validation(self, validation_results: Dict[str, Any]) -> ValidationFeedback:
        """Create feedback object from validation results."""
        details = validation_results.get('details', {})
        
        return ValidationFeedback(
            test_results=details.get('test_results', {}),
            static_analysis={},  # Could add static analysis here
            style_violations=details.get('style_violations', []),
            semantic_issues=details.get('semantic_analysis', {}).get('issues', []),
            performance_metrics={},  # Could add performance metrics
            suggestions=self._generate_suggestions_from_validation(validation_results)
        )
    
    def _generate_suggestions_from_validation(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on validation results."""
        suggestions = []
        
        if not validation_results.get('tests_passed'):
            suggestions.append("Focus on fixing failing tests")
        
        if not validation_results.get('style_compliant'):
            suggestions.append("Address style violations for better code quality")
        
        if not validation_results.get('no_errors'):
            suggestions.append("Fix syntax errors before other improvements")
        
        if not validation_results.get('semantic_preserved'):
            suggestions.append("Ensure the fix preserves original functionality")
        
        return suggestions
    
    def _extract_changes(self, old_patch: Dict[str, Any], new_patch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the changes between two patches."""
        changes = []
        
        old_changes = {
            (c['line_start'], c['line_end']): c
            for c in old_patch.get('changes', [])
        }
        
        new_changes = {
            (c['line_start'], c['line_end']): c
            for c in new_patch.get('changes', [])
        }
        
        # Find modified changes
        for key, new_change in new_changes.items():
            if key in old_changes:
                old_change = old_changes[key]
                if old_change['new_code'] != new_change['new_code']:
                    changes.append({
                        'type': 'modified',
                        'location': key,
                        'old': old_change['new_code'],
                        'new': new_change['new_code'],
                        'reason': new_change.get('refinement_reason', 'Refined')
                    })
            else:
                changes.append({
                    'type': 'added',
                    'location': key,
                    'change': new_change,
                    'reason': new_change.get('refinement_reason', 'Added')
                })
        
        # Find removed changes
        for key, old_change in old_changes.items():
            if key not in new_changes:
                changes.append({
                    'type': 'removed',
                    'location': key,
                    'change': old_change,
                    'reason': 'Removed as unnecessary'
                })
        
        return changes
    
    def _extract_feedback_items(self, feedback: ValidationFeedback) -> List[str]:
        """Extract actionable feedback items."""
        items = []
        
        # From test results
        for test in feedback.test_results.get('failed_tests', [])[:3]:
            items.append(f"Fix test: {test['name']}")
        
        # From style violations
        for violation in feedback.style_violations[:3]:
            items.append(f"Style: {violation['message']}")
        
        # From semantic issues
        for issue in feedback.semantic_issues[:3]:
            items.append(f"Semantic: {issue}")
        
        # From suggestions
        items.extend(feedback.suggestions[:2])
        
        return items
    
    def _update_strategy_weights(self, strategy: RefinementStrategy, improvement_metrics: Dict[str, float]):
        """Update strategy weights based on performance."""
        overall_improvement = improvement_metrics.get('overall', 0)
        
        # Increase weight for successful strategies
        if overall_improvement > 0.5:
            self.strategy_weights[strategy] = min(
                1.0,
                self.strategy_weights[strategy] + self.learning_rate * overall_improvement
            )
        elif overall_improvement < 0:
            # Decrease weight for unsuccessful strategies
            self.strategy_weights[strategy] = max(
                0.1,
                self.strategy_weights[strategy] - self.learning_rate * abs(overall_improvement)
            )
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for s in self.strategy_weights:
            self.strategy_weights[s] /= total_weight
    
    def _select_best_iteration(self, history: RefinementHistory) -> Optional[int]:
        """Select the best iteration from history."""
        if not history.iterations:
            return None
        
        # Score each iteration
        scores = []
        for i, iteration in enumerate(history.iterations):
            score = 0.0
            
            # Success is most important
            if iteration.success:
                score += 1.0
            
            # Then overall improvement
            score += iteration.improvement_metrics.get('overall', 0) * 0.5
            
            # Prefer earlier iterations if scores are similar (simpler is better)
            score -= i * 0.01
            
            scores.append((i, score))
        
        # Return iteration with highest score
        best_iteration, _ = max(scores, key=lambda x: x[1])
        return best_iteration
    
    def _reconstruct_patch_at_iteration(self,
                                      initial_patch: Dict[str, Any],
                                      history: RefinementHistory,
                                      iteration_index: int) -> Dict[str, Any]:
        """Reconstruct the patch state at a specific iteration."""
        patch = initial_patch.copy()
        
        # Apply changes up to the specified iteration
        for i in range(iteration_index + 1):
            iteration = history.iterations[i]
            # This is simplified - in reality, we'd need to track the actual patches
            # For now, we'll just note that this iteration was the best
            patch['refinement_iteration'] = i + 1
            patch['refinement_strategy'] = iteration.strategy.value
        
        return patch
    
    def _learn_from_refinement(self, history: RefinementHistory):
        """Learn patterns from successful refinements."""
        if history.convergence_achieved or (history.best_iteration is not None):
            # Extract successful patterns
            successful_iterations = [
                it for it in history.iterations
                if it.success or it.improvement_metrics.get('overall', 0) > 0.7
            ]
            
            for iteration in successful_iterations:
                pattern = {
                    'strategy': iteration.strategy,
                    'feedback_type': self._categorize_feedback(iteration.feedback_incorporated),
                    'improvement': iteration.improvement_metrics.get('overall', 0),
                    'changes': len(iteration.changes_made)
                }
                
                # Add to learned patterns (with a limit)
                self.learned_patterns.append(pattern)
                if len(self.learned_patterns) > 100:
                    # Remove oldest patterns
                    self.learned_patterns = self.learned_patterns[-100:]
    
    def _categorize_feedback(self, feedback_items: List[str]) -> str:
        """Categorize feedback items for pattern learning."""
        categories = {
            'test_focused': any('test' in item.lower() for item in feedback_items),
            'style_focused': any('style' in item.lower() for item in feedback_items),
            'error_focused': any('error' in item.lower() or 'exception' in item.lower() for item in feedback_items),
            'semantic_focused': any('semantic' in item.lower() or 'logic' in item.lower() for item in feedback_items)
        }
        
        # Return the primary category
        for category, present in categories.items():
            if present:
                return category
        
        return 'general'
    
    def _format_learned_patterns(self, strategy: RefinementStrategy) -> str:
        """Format learned patterns for inclusion in prompts."""
        relevant_patterns = [
            p for p in self.learned_patterns
            if p['strategy'] == strategy and p['improvement'] > 0.5
        ]
        
        if not relevant_patterns:
            return ""
        
        # Sort by improvement
        relevant_patterns.sort(key=lambda x: x['improvement'], reverse=True)
        
        pattern_text = "\n\nLEARNED PATTERNS FOR THIS STRATEGY:"
        for pattern in relevant_patterns[:3]:  # Top 3 patterns
            pattern_text += f"\n- When facing {pattern['feedback_type']} issues, "
            pattern_text += f"making {pattern['changes']} changes "
            pattern_text += f"achieved {pattern['improvement']:.0%} improvement"
        
        return pattern_text
    
    def _parse_refinement_response(self, response: str, current_patch: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the refinement response from LLM."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                refined_data = json.loads(json_match.group(1))
                
                # Merge with current patch structure
                refined_patch = current_patch.copy()
                refined_patch['changes'] = refined_data.get('changes', current_patch.get('changes', []))
                refined_patch['refinement_summary'] = refined_data.get('refinement_summary', '')
                refined_patch['expected_improvements'] = refined_data.get('expected_improvements', [])
                refined_patch['confidence'] = refined_data.get('confidence', 0.5)
                
                return refined_patch
            else:
                # Try direct JSON parsing
                refined_data = json.loads(response)
                refined_patch = current_patch.copy()
                refined_patch.update(refined_data)
                return refined_patch
                
        except Exception as e:
            logger.error(f"Error parsing refinement response: {e}")
            # Return current patch unchanged
            return current_patch
    
    def _apply_strategy_specific_processing(self,
                                          patch: Dict[str, Any],
                                          strategy: RefinementStrategy,
                                          feedback: ValidationFeedback) -> Dict[str, Any]:
        """Apply strategy-specific post-processing to the patch."""
        if strategy == RefinementStrategy.STYLE_COMPLIANCE:
            # Ensure proper formatting
            for change in patch.get('changes', []):
                if 'new_code' in change:
                    # Could apply automatic formatting here
                    change['new_code'] = change['new_code'].rstrip()
        
        elif strategy == RefinementStrategy.TEST_DRIVEN:
            # Add test-specific modifications
            patch['test_focused'] = True
        
        elif strategy == RefinementStrategy.PERFORMANCE_ORIENTED:
            # Mark performance-critical sections
            patch['performance_optimized'] = True
        
        return patch
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        suffix = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust'
        }
        return language_map.get(suffix, 'unknown')


def create_iterative_refiner(llm_manager: LLMManager,
                           context_manager: LLMContextManager,
                           config: Optional[Dict[str, Any]] = None) -> IterativeRefiner:
    """
    Factory function to create an iterative refiner.
    
    Args:
        llm_manager: LLM manager instance
        context_manager: Context manager instance
        config: Configuration dictionary
        
    Returns:
        Configured iterative refiner
    """
    return IterativeRefiner(llm_manager, context_manager, config)