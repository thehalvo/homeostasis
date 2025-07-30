"""Automated Rule Extraction from Successful Fixes.

This module analyzes successful patches to identify patterns and
automatically generate new detection and fix rules.
"""

import ast
import json
import logging
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import difflib
import hashlib

from ..llm_integration.patch_generator import PatchData
from ..monitoring.healing_metrics import HealingMetrics
from ..primary_languages.rule_engine import Rule, RuleEngine

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPattern:
    """Represents a pattern extracted from successful fixes."""
    pattern_id: str
    pattern_type: str  # 'error_detection', 'fix_template', 'code_transform'
    description: str
    source_examples: List[str]  # IDs of fixes this was extracted from
    confidence: float
    frequency: int
    
    # Pattern specifics
    error_patterns: Optional[List[str]] = None  # Regex patterns for errors
    code_patterns: Optional[Dict[str, str]] = None  # Before/after code patterns
    fix_template: Optional[str] = None  # Template for generating fixes
    conditions: Optional[Dict[str, Any]] = None  # Conditions for applying
    
    def to_rule(self) -> Optional[Rule]:
        """Convert pattern to a Rule object."""
        if self.pattern_type == 'error_detection':
            return Rule(
                rule_id=f"auto_{self.pattern_id}",
                name=f"Auto-extracted: {self.description}",
                error_patterns=self.error_patterns,
                confidence_score=self.confidence,
                auto_generated=True,
                metadata={
                    'source_examples': self.source_examples,
                    'frequency': self.frequency
                }
            )
        elif self.pattern_type == 'fix_template':
            return Rule(
                rule_id=f"auto_{self.pattern_id}",
                name=f"Auto-extracted: {self.description}",
                fix_template=self.fix_template,
                conditions=self.conditions,
                confidence_score=self.confidence,
                auto_generated=True,
                metadata={
                    'source_examples': self.source_examples,
                    'frequency': self.frequency
                }
            )
        return None


class RuleExtractor:
    """Extracts rules and patterns from successful fixes."""
    
    def __init__(
        self,
        storage_dir: Path = Path("data/extracted_rules"),
        min_examples: int = 3,
        min_confidence: float = 0.7
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.min_examples = min_examples
        self.min_confidence = min_confidence
        
        self.patterns = {}
        self.fix_history = []
        self.pattern_candidates = defaultdict(list)
        
        # Load existing patterns
        self._load_patterns()
    
    def analyze_successful_fix(
        self,
        error_data: Dict[str, Any],
        patch_data: PatchData,
        fix_metadata: Dict[str, Any]
    ) -> List[ExtractedPattern]:
        """Analyze a successful fix to extract patterns."""
        fix_id = fix_metadata.get('fix_id', str(hashlib.md5(
            f"{error_data}{patch_data}".encode()).hexdigest()))
        
        # Store fix history
        self.fix_history.append({
            'fix_id': fix_id,
            'error_data': error_data,
            'patch_data': patch_data,
            'metadata': fix_metadata
        })
        
        extracted_patterns = []
        
        # Extract error detection patterns
        error_patterns = self._extract_error_patterns(error_data)
        if error_patterns:
            extracted_patterns.extend(error_patterns)
        
        # Extract code transformation patterns
        code_patterns = self._extract_code_patterns(patch_data)
        if code_patterns:
            extracted_patterns.extend(code_patterns)
        
        # Extract fix templates
        fix_templates = self._extract_fix_templates(error_data, patch_data)
        if fix_templates:
            extracted_patterns.extend(fix_templates)
        
        # Update pattern candidates
        for pattern in extracted_patterns:
            self._update_pattern_candidates(pattern, fix_id)
        
        # Check if any candidates are ready to become rules
        new_rules = self._promote_candidates_to_rules()
        
        return new_rules
    
    def _extract_error_patterns(self, error_data: Dict[str, Any]) -> List[ExtractedPattern]:
        """Extract patterns from error messages and stack traces."""
        patterns = []
        
        error_message = error_data.get('error_message', '')
        stack_trace = error_data.get('stack_trace', '')
        
        # Extract regex patterns from error message
        if error_message:
            # Look for common error patterns
            error_pattern = self._generalize_error_message(error_message)
            if error_pattern:
                pattern = ExtractedPattern(
                    pattern_id=hashlib.md5(error_pattern.encode()).hexdigest()[:8],
                    pattern_type='error_detection',
                    description=f"Error pattern: {error_pattern[:50]}...",
                    source_examples=[],
                    confidence=0.8,
                    frequency=1,
                    error_patterns=[error_pattern]
                )
                patterns.append(pattern)
        
        # Extract patterns from stack trace
        if stack_trace:
            trace_patterns = self._extract_stack_trace_patterns(stack_trace)
            for tp in trace_patterns:
                pattern = ExtractedPattern(
                    pattern_id=hashlib.md5(tp.encode()).hexdigest()[:8],
                    pattern_type='error_detection',
                    description=f"Stack trace pattern: {tp[:50]}...",
                    source_examples=[],
                    confidence=0.7,
                    frequency=1,
                    error_patterns=[tp]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_code_patterns(self, patch_data: PatchData) -> List[ExtractedPattern]:
        """Extract code transformation patterns from patches."""
        patterns = []
        
        # Get the diff
        diff_lines = patch_data.get_diff().split('\n')
        
        # Extract before/after code blocks
        before_code = []
        after_code = []
        
        for line in diff_lines:
            if line.startswith('-') and not line.startswith('---'):
                before_code.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                after_code.append(line[1:])
        
        if before_code and after_code:
            # Try to identify the transformation pattern
            transform = self._identify_code_transform(before_code, after_code)
            if transform:
                pattern = ExtractedPattern(
                    pattern_id=hashlib.md5(str(transform).encode()).hexdigest()[:8],
                    pattern_type='code_transform',
                    description=transform['description'],
                    source_examples=[],
                    confidence=transform['confidence'],
                    frequency=1,
                    code_patterns={
                        'before': transform['before_pattern'],
                        'after': transform['after_pattern']
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_fix_templates(
        self,
        error_data: Dict[str, Any],
        patch_data: PatchData
    ) -> List[ExtractedPattern]:
        """Extract fix templates from successful patches."""
        patterns = []
        
        # Analyze the fix structure
        fix_structure = self._analyze_fix_structure(patch_data)
        
        if fix_structure:
            # Create a template from the fix
            template = self._create_fix_template(fix_structure, error_data)
            
            if template:
                pattern = ExtractedPattern(
                    pattern_id=hashlib.md5(template.encode()).hexdigest()[:8],
                    pattern_type='fix_template',
                    description=fix_structure['description'],
                    source_examples=[],
                    confidence=0.75,
                    frequency=1,
                    fix_template=template,
                    conditions=fix_structure.get('conditions', {})
                )
                patterns.append(pattern)
        
        return patterns
    
    def _generalize_error_message(self, error_message: str) -> Optional[str]:
        """Convert specific error message to general regex pattern."""
        # Replace specific values with regex patterns
        generalized = error_message
        
        # Replace file paths
        generalized = re.sub(r'[/\\][\w/\\.-]+\.\w+', r'[/\\\\][\\w/\\\\.-]+\\.\\w+', generalized)
        
        # Replace line numbers
        generalized = re.sub(r'line \d+', r'line \\d+', generalized)
        
        # Replace variable names (simple heuristic)
        generalized = re.sub(r"'[a-zA-Z_]\w*'", r"'[a-zA-Z_]\\w*'", generalized)
        
        # Replace numbers
        generalized = re.sub(r'\b\d+\b', r'\\d+', generalized)
        
        # Only return if meaningfully different from original
        if generalized != error_message and len(generalized) < len(error_message) * 2:
            return generalized
        
        return None
    
    def _extract_stack_trace_patterns(self, stack_trace: str) -> List[str]:
        """Extract patterns from stack traces."""
        patterns = []
        
        # Look for common stack trace patterns
        lines = stack_trace.split('\n')
        
        # Extract method call patterns
        for line in lines:
            if 'at ' in line or 'File ' in line:
                # Generalize the line
                pattern = re.sub(r'\d+', r'\\d+', line)
                pattern = re.sub(r'0x[0-9a-fA-F]+', r'0x[0-9a-fA-F]+', pattern)
                patterns.append(pattern)
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _identify_code_transform(
        self,
        before_code: List[str],
        after_code: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Identify the transformation pattern between before and after code."""
        before_text = '\n'.join(before_code)
        after_text = '\n'.join(after_code)
        
        # Common transformation patterns
        transformations = [
            {
                'name': 'null_check_addition',
                'before_pattern': r'(\w+)\.(\w+)',
                'after_pattern': r'if \(\1 != null\) \{\n    \1\.\2\n\}',
                'description': 'Add null check before method call',
                'confidence': 0.9
            },
            {
                'name': 'try_catch_wrapper',
                'before_pattern': r'(.+)',
                'after_pattern': r'try \{\n    \1\n\} catch \(Exception e\) \{\n    .+\n\}',
                'description': 'Wrap code in try-catch block',
                'confidence': 0.85
            },
            {
                'name': 'bounds_check',
                'before_pattern': r'(\w+)\[(\w+)\]',
                'after_pattern': r'if \(\2 >= 0 && \2 < \1\.length\) \{\n    \1\[\2\]\n\}',
                'description': 'Add array bounds check',
                'confidence': 0.9
            },
            {
                'name': 'resource_cleanup',
                'before_pattern': r'(\w+) = new (\w+)\(\)',
                'after_pattern': r'try \(\1 = new \2\(\)\) \{\n    .+\n\}',
                'description': 'Add try-with-resources for cleanup',
                'confidence': 0.8
            }
        ]
        
        # Check each transformation pattern
        for transform in transformations:
            if (re.search(transform['before_pattern'], before_text) and
                re.search(transform['after_pattern'], after_text)):
                return transform
        
        # Generic transformation if no specific pattern matches
        if before_text != after_text:
            return {
                'name': 'generic_transform',
                'before_pattern': before_text[:50],
                'after_pattern': after_text[:50],
                'description': 'Generic code transformation',
                'confidence': 0.6
            }
        
        return None
    
    def _analyze_fix_structure(self, patch_data: PatchData) -> Optional[Dict[str, Any]]:
        """Analyze the structure of a fix."""
        diff = patch_data.get_diff()
        
        # Count added/removed lines
        added_lines = len([l for l in diff.split('\n') if l.startswith('+')])
        removed_lines = len([l for l in diff.split('\n') if l.startswith('-')])
        
        structure = {
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'description': ''
        }
        
        # Categorize fix type
        if added_lines > removed_lines * 2:
            structure['type'] = 'addition'
            structure['description'] = 'Add new code block'
        elif removed_lines > added_lines * 2:
            structure['type'] = 'removal'
            structure['description'] = 'Remove problematic code'
        else:
            structure['type'] = 'modification'
            structure['description'] = 'Modify existing code'
        
        # Look for specific patterns
        if 'if' in diff and 'null' in diff:
            structure['pattern'] = 'null_check'
        elif 'try' in diff and 'catch' in diff:
            structure['pattern'] = 'exception_handling'
        elif 'finally' in diff or 'close()' in diff:
            structure['pattern'] = 'resource_cleanup'
        
        return structure
    
    def _create_fix_template(
        self,
        fix_structure: Dict[str, Any],
        error_data: Dict[str, Any]
    ) -> Optional[str]:
        """Create a fix template from the structure analysis."""
        template_parts = []
        
        # Add conditional based on error type
        if error_data.get('error_type'):
            template_parts.append(f"if error_type == '{error_data['error_type']}':")
        
        # Add fix pattern
        if fix_structure.get('pattern') == 'null_check':
            template_parts.append("  add_null_check(variable)")
        elif fix_structure.get('pattern') == 'exception_handling':
            template_parts.append("  wrap_in_try_catch(code_block)")
        elif fix_structure.get('pattern') == 'resource_cleanup':
            template_parts.append("  add_resource_cleanup(resource)")
        else:
            template_parts.append(f"  apply_{fix_structure['type']}_fix()")
        
        return '\n'.join(template_parts) if template_parts else None
    
    def _update_pattern_candidates(self, pattern: ExtractedPattern, fix_id: str) -> None:
        """Update pattern candidates with new example."""
        pattern_key = f"{pattern.pattern_type}:{pattern.pattern_id}"
        
        # Find similar patterns
        similar_found = False
        for key, candidates in self.pattern_candidates.items():
            if self._patterns_similar(pattern, candidates[0]):
                candidates.append((pattern, fix_id))
                similar_found = True
                break
        
        if not similar_found:
            self.pattern_candidates[pattern_key].append((pattern, fix_id))
    
    def _patterns_similar(self, p1: ExtractedPattern, p2: Tuple[ExtractedPattern, str]) -> bool:
        """Check if two patterns are similar enough to merge."""
        pattern2 = p2[0] if isinstance(p2, tuple) else p2
        
        if p1.pattern_type != pattern2.pattern_type:
            return False
        
        # Type-specific similarity checks
        if p1.pattern_type == 'error_detection':
            # Check if error patterns overlap
            if p1.error_patterns and pattern2.error_patterns:
                return any(ep1 == ep2 for ep1 in p1.error_patterns 
                          for ep2 in pattern2.error_patterns)
        
        elif p1.pattern_type == 'code_transform':
            # Check if transformations are similar
            if p1.code_patterns and pattern2.code_patterns:
                return (p1.code_patterns.get('before') == pattern2.code_patterns.get('before') or
                       p1.code_patterns.get('after') == pattern2.code_patterns.get('after'))
        
        return False
    
    def _promote_candidates_to_rules(self) -> List[ExtractedPattern]:
        """Promote pattern candidates to rules if they meet criteria."""
        promoted = []
        
        for pattern_key, candidates in list(self.pattern_candidates.items()):
            if len(candidates) >= self.min_examples:
                # Merge patterns and create rule
                merged_pattern = self._merge_patterns(candidates)
                
                if merged_pattern.confidence >= self.min_confidence:
                    self.patterns[pattern_key] = merged_pattern
                    promoted.append(merged_pattern)
                    
                    # Save the new pattern
                    self._save_pattern(merged_pattern)
                    
                    # Clear candidates
                    del self.pattern_candidates[pattern_key]
        
        return promoted
    
    def _merge_patterns(self, candidates: List[Tuple[ExtractedPattern, str]]) -> ExtractedPattern:
        """Merge multiple pattern candidates into one."""
        base_pattern = candidates[0][0]
        
        # Collect all source examples
        source_examples = [fix_id for _, fix_id in candidates]
        
        # Calculate combined confidence
        confidences = [p[0].confidence for p in candidates]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Create merged pattern
        merged = ExtractedPattern(
            pattern_id=base_pattern.pattern_id,
            pattern_type=base_pattern.pattern_type,
            description=base_pattern.description,
            source_examples=source_examples,
            confidence=avg_confidence,
            frequency=len(candidates),
            error_patterns=base_pattern.error_patterns,
            code_patterns=base_pattern.code_patterns,
            fix_template=base_pattern.fix_template,
            conditions=base_pattern.conditions
        )
        
        return merged
    
    def get_rules(self, min_frequency: int = 3) -> List[Rule]:
        """Get all extracted rules above frequency threshold."""
        rules = []
        
        for pattern in self.patterns.values():
            if pattern.frequency >= min_frequency:
                rule = pattern.to_rule()
                if rule:
                    rules.append(rule)
        
        return rules
    
    def _save_pattern(self, pattern: ExtractedPattern) -> None:
        """Save pattern to disk."""
        pattern_file = self.storage_dir / f"{pattern.pattern_type}_{pattern.pattern_id}.json"
        
        with open(pattern_file, 'w') as f:
            json.dump(asdict(pattern), f, indent=2)
    
    def _load_patterns(self) -> None:
        """Load existing patterns from disk."""
        if self.storage_dir.exists():
            for pattern_file in self.storage_dir.glob("*.json"):
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    pattern = ExtractedPattern(**data)
                    pattern_key = f"{pattern.pattern_type}:{pattern.pattern_id}"
                    self.patterns[pattern_key] = pattern


class PatternAnalyzer:
    """Analyzes patterns across multiple fixes to identify trends."""
    
    def __init__(self, rule_extractor: RuleExtractor):
        self.rule_extractor = rule_extractor
        self.pattern_stats = defaultdict(lambda: {
            'occurrences': 0,
            'success_rate': 0.0,
            'avg_fix_time': 0.0,
            'languages': set(),
            'error_types': set()
        })
    
    def analyze_pattern_effectiveness(
        self,
        pattern: ExtractedPattern,
        healing_metrics: HealingMetrics
    ) -> Dict[str, Any]:
        """Analyze how effective a pattern has been."""
        effectiveness = {
            'pattern_id': pattern.pattern_id,
            'total_applications': len(pattern.source_examples),
            'success_rate': 0.0,
            'avg_fix_time': 0.0,
            'error_coverage': []
        }
        
        # Get metrics for each application
        successes = 0
        total_time = 0
        error_types = set()
        
        for fix_id in pattern.source_examples:
            metrics = healing_metrics.get_fix_metrics(fix_id)
            if metrics:
                if metrics.get('success'):
                    successes += 1
                total_time += metrics.get('fix_time', 0)
                error_types.add(metrics.get('error_type'))
        
        if pattern.source_examples:
            effectiveness['success_rate'] = successes / len(pattern.source_examples)
            effectiveness['avg_fix_time'] = total_time / len(pattern.source_examples)
            effectiveness['error_coverage'] = list(error_types)
        
        return effectiveness
    
    def find_pattern_correlations(self) -> List[Tuple[str, str, float]]:
        """Find patterns that often appear together."""
        correlations = []
        
        # Build co-occurrence matrix
        pattern_pairs = defaultdict(int)
        
        for fix in self.rule_extractor.fix_history:
            fix_patterns = []
            # Extract all patterns from this fix
            # (This would need actual implementation)
            
            # Count co-occurrences
            for i, p1 in enumerate(fix_patterns):
                for p2 in fix_patterns[i+1:]:
                    key = tuple(sorted([p1.pattern_id, p2.pattern_id]))
                    pattern_pairs[key] += 1
        
        # Calculate correlations
        for (p1_id, p2_id), count in pattern_pairs.items():
            if count >= 3:  # Minimum co-occurrences
                # Simple correlation score
                correlation = count / len(self.rule_extractor.fix_history)
                correlations.append((p1_id, p2_id, correlation))
        
        return sorted(correlations, key=lambda x: x[2], reverse=True)


class AutomatedRuleGenerator:
    """Generates new rules based on extracted patterns."""
    
    def __init__(
        self,
        rule_engine: RuleEngine,
        rule_extractor: RuleExtractor,
        pattern_analyzer: PatternAnalyzer
    ):
        self.rule_engine = rule_engine
        self.rule_extractor = rule_extractor
        self.pattern_analyzer = pattern_analyzer
    
    def generate_and_add_rules(
        self,
        min_frequency: int = 5,
        min_effectiveness: float = 0.8
    ) -> List[Rule]:
        """Generate new rules and add them to the rule engine."""
        # Get candidate rules
        candidate_rules = self.rule_extractor.get_rules(min_frequency)
        
        added_rules = []
        
        for rule in candidate_rules:
            # Check effectiveness
            pattern = self._get_pattern_for_rule(rule)
            if pattern:
                effectiveness = self.pattern_analyzer.analyze_pattern_effectiveness(
                    pattern,
                    HealingMetrics()  # Would use actual metrics instance
                )
                
                if effectiveness['success_rate'] >= min_effectiveness:
                    # Add to rule engine
                    self.rule_engine.add_rule(rule)
                    added_rules.append(rule)
                    
                    logger.info(
                        f"Added auto-generated rule {rule.rule_id} with "
                        f"{effectiveness['success_rate']:.2%} success rate"
                    )
        
        return added_rules
    
    def create_composite_rules(self) -> List[Rule]:
        """Create composite rules from correlated patterns."""
        composite_rules = []
        
        # Find correlated patterns
        correlations = self.pattern_analyzer.find_pattern_correlations()
        
        for p1_id, p2_id, correlation in correlations[:10]:  # Top 10
            if correlation > 0.5:
                # Create composite rule
                rule = self._create_composite_rule(p1_id, p2_id, correlation)
                if rule:
                    composite_rules.append(rule)
        
        return composite_rules
    
    def _get_pattern_for_rule(self, rule: Rule) -> Optional[ExtractedPattern]:
        """Get the pattern that generated a rule."""
        # Extract pattern ID from rule ID (assuming format "auto_<pattern_id>")
        if rule.rule_id.startswith("auto_"):
            pattern_id = rule.rule_id[5:]
            for pattern in self.rule_extractor.patterns.values():
                if pattern.pattern_id == pattern_id:
                    return pattern
        return None
    
    def _create_composite_rule(
        self,
        pattern1_id: str,
        pattern2_id: str,
        correlation: float
    ) -> Optional[Rule]:
        """Create a composite rule from two correlated patterns."""
        # Get patterns
        pattern1 = None
        pattern2 = None
        
        for pattern in self.rule_extractor.patterns.values():
            if pattern.pattern_id == pattern1_id:
                pattern1 = pattern
            elif pattern.pattern_id == pattern2_id:
                pattern2 = pattern
        
        if pattern1 and pattern2:
            # Create composite rule
            return Rule(
                rule_id=f"composite_{pattern1_id}_{pattern2_id}",
                name=f"Composite: {pattern1.description} + {pattern2.description}",
                confidence_score=min(pattern1.confidence, pattern2.confidence) * correlation,
                auto_generated=True,
                metadata={
                    'correlation': correlation,
                    'component_patterns': [pattern1_id, pattern2_id]
                }
            )
        
        return None