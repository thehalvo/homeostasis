#!/usr/bin/env python3
"""
Enhanced LLM Patch Generator with Phase 13.3 Capabilities

This module extends the basic LLM patch generator with advanced features:
- Integration with advanced code generator for context-aware generation
- Integration with contextual analyzer for deep code understanding
- Integration with iterative refiner for progressive improvement
- Integration with advanced prompting for better LLM interactions
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .llm_patch_generator import (
    LLMPatchGenerator, PatchValidationConfig, TestResult, ValidationResult
)
from .advanced_code_generator import (
    AdvancedCodeGenerator, CodeGenerationMode, GenerationResult,
    create_advanced_code_generator
)
from .contextual_analyzer import (
    ContextualAnalyzer, ImpactAnalysis, create_contextual_analyzer
)
from .iterative_refiner import (
    IterativeRefiner, ValidationFeedback, RefinementHistory,
    create_iterative_refiner
)
from ..llm_integration.advanced_prompting import (
    AdvancedPromptManager, PromptingStrategy, PromptExample,
    create_advanced_prompt_manager
)
from ..llm_integration.provider_abstraction import LLMManager
from ..llm_integration.api_key_manager import APIKeyManager
from ..analysis.llm_context_manager import LLMContextManager

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPatchResult:
    """Result of enhanced patch generation."""
    success: bool
    patch: Dict[str, Any]
    generation_mode: CodeGenerationMode
    refinement_history: Optional[RefinementHistory] = None
    impact_analysis: Optional[ImpactAnalysis] = None
    multi_file_patches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedLLMPatchGenerator(LLMPatchGenerator):
    """
    Enhanced LLM patch generator with Phase 13.3 capabilities.
    
    This class extends the basic LLM patch generator with:
    - Advanced code generation with semantic understanding
    - Contextual analysis for multi-file awareness
    - Iterative refinement based on validation feedback
    - Advanced prompting strategies for better results
    """
    
    def __init__(self,
                 api_key_manager: Optional[APIKeyManager] = None,
                 context_manager: Optional[LLMContextManager] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced LLM patch generator.
        
        Args:
            api_key_manager: API key manager for LLM providers
            context_manager: LLM context manager for storing contexts
            config: Configuration dictionary
        """
        # Initialize base class
        super().__init__(api_key_manager, context_manager, config)
        
        # Initialize Phase 13.3 components
        self.advanced_generator = create_advanced_code_generator(
            self.llm_manager, self.context_manager, config
        )
        self.contextual_analyzer = create_contextual_analyzer(config)
        self.iterative_refiner = create_iterative_refiner(
            self.llm_manager, self.context_manager, config
        )
        self.prompt_manager = create_advanced_prompt_manager(config)
        
        # Enhanced configuration
        self.enable_advanced_features = config.get('enable_advanced_features', True)
        self.enable_iterative_refinement = config.get('enable_iterative_refinement', True)
        self.enable_impact_analysis = config.get('enable_impact_analysis', True)
        self.semantic_analysis_depth = config.get('semantic_analysis_depth', 3)
        
        logger.info("Initialized Enhanced LLM Patch Generator with Phase 13.3 capabilities")
    
    def generate_enhanced_patch(self,
                              error_context: Dict[str, Any],
                              source_code: Optional[str] = None,
                              additional_context: Optional[Dict[str, Any]] = None,
                              mode: Optional[CodeGenerationMode] = None) -> EnhancedPatchResult:
        """
        Generate an enhanced patch with all Phase 13.3 capabilities.
        
        Args:
            error_context: Error context information
            source_code: Source code containing the error
            additional_context: Additional context information
            mode: Code generation mode (auto-detected if None)
            
        Returns:
            Enhanced patch result with all analysis and refinements
        """
        try:
            # Determine generation mode
            if mode is None:
                mode = self._determine_generation_mode(error_context, additional_context)
            
            # Perform codebase context analysis if enabled
            codebase_context = None
            if self.enable_impact_analysis:
                codebase_context = self.contextual_analyzer.analyze_codebase_context(
                    error_context.get('file_path', ''),
                    additional_context.get('root_dir')
                )
            
            # Use advanced code generator for initial patch
            generation_result = self.advanced_generator.generate_with_context(
                error_context, source_code or '', mode
            )
            
            if not generation_result.success:
                return EnhancedPatchResult(
                    success=False,
                    patch={},
                    generation_mode=mode,
                    reasoning=generation_result.reasoning,
                    warnings=generation_result.warnings
                )
            
            # Perform impact analysis
            impact_analysis = None
            if self.enable_impact_analysis and generation_result.changes:
                impact_analysis = self._perform_impact_analysis(
                    error_context, generation_result, codebase_context
                )
            
            # Create initial patch structure
            initial_patch = self._create_patch_from_generation(
                generation_result, error_context, mode
            )
            
            # Perform iterative refinement if enabled
            refinement_history = None
            final_patch = initial_patch
            
            if self.enable_iterative_refinement and self.validation_config.enable_test_validation:
                refinement_history = self._perform_iterative_refinement(
                    initial_patch, error_context, source_code, generation_result
                )
                
                if refinement_history.final_patch:
                    final_patch = refinement_history.final_patch
            
            # Handle multi-file patches
            multi_file_patches = {}
            if generation_result.multi_file_changes:
                multi_file_patches = self._create_multi_file_patches(
                    generation_result.multi_file_changes, error_context
                )
            
            # Create enhanced result
            return EnhancedPatchResult(
                success=True,
                patch=final_patch,
                generation_mode=mode,
                refinement_history=refinement_history,
                impact_analysis=impact_analysis,
                multi_file_patches=multi_file_patches,
                confidence=self._calculate_overall_confidence(
                    generation_result, refinement_history, impact_analysis
                ),
                reasoning=self._create_comprehensive_reasoning(
                    generation_result, refinement_history, impact_analysis
                ),
                warnings=self._collect_all_warnings(
                    generation_result, refinement_history, impact_analysis
                ),
                metadata={
                    'semantic_analysis': generation_result.semantic_analysis,
                    'codebase_context': codebase_context,
                    'generation_time': generation_result.__dict__.get('generation_time', 0),
                    'refinement_time': refinement_history.total_time if refinement_history else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced patch generation: {e}")
            return EnhancedPatchResult(
                success=False,
                patch={},
                generation_mode=mode or CodeGenerationMode.BUG_FIX,
                reasoning=f"Generation failed: {str(e)}",
                warnings=[str(e)]
            )
    
    def generate_patch_from_error_context(self,
                                        error_context: Dict[str, Any],
                                        source_code: Optional[str] = None,
                                        additional_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Override base method to use enhanced generation when enabled.
        
        Args:
            error_context: Error context information
            source_code: Source code containing the error
            additional_context: Additional context
            
        Returns:
            Generated patch or None
        """
        if not self.enable_advanced_features:
            # Fall back to base implementation
            return super().generate_patch_from_error_context(
                error_context, source_code, additional_context
            )
        
        # Use enhanced generation
        result = self.generate_enhanced_patch(
            error_context, source_code, additional_context
        )
        
        if result.success:
            return result.patch
        return None
    
    def _enhance_prompt_with_source_code(self,
                                       prompt_data: Dict[str, Any],
                                       source_code: str,
                                       error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override to use advanced prompting strategies.
        
        Args:
            prompt_data: Original prompt data
            source_code: Source code
            error_context: Error context
            
        Returns:
            Enhanced prompt data
        """
        # First apply base enhancement
        prompt_data = super()._enhance_prompt_with_source_code(
            prompt_data, source_code, error_context
        )
        
        if self.enable_advanced_features:
            # Use advanced prompt manager
            task_type = self._determine_task_type(error_context)
            
            # Create advanced prompt
            advanced_prompt = self.prompt_manager.create_prompt(
                task_type,
                {
                    **error_context,
                    'code': source_code,
                    'requirements': prompt_data.get('requirements', [])
                }
            )
            
            # Replace user prompt with advanced version
            prompt_data['user_prompt'] = advanced_prompt
        
        return prompt_data
    
    def _determine_generation_mode(self,
                                 error_context: Dict[str, Any],
                                 additional_context: Optional[Dict[str, Any]]) -> CodeGenerationMode:
        """Determine the appropriate code generation mode."""
        # Check for multi-file indicators
        if additional_context and additional_context.get('multi_file_required'):
            return CodeGenerationMode.MULTI_FILE
        
        # Check for refactoring indicators
        if error_context.get('task_type') == 'refactoring':
            return CodeGenerationMode.REFACTORING
        
        # Check for feature addition
        if error_context.get('task_type') == 'feature_addition':
            return CodeGenerationMode.FEATURE_ADDITION
        
        # Default to bug fix
        return CodeGenerationMode.BUG_FIX
    
    def _determine_task_type(self, error_context: Dict[str, Any]) -> str:
        """Determine task type for prompt selection."""
        if error_context.get('task_type'):
            return error_context['task_type']
        
        error_type = error_context.get('error_type', '')
        
        if 'performance' in error_type.lower():
            return 'optimization'
        elif 'refactor' in error_type.lower():
            return 'refactoring'
        elif any(x in error_type.lower() for x in ['add', 'feature', 'implement']):
            return 'feature_addition'
        else:
            return 'bug_fix'
    
    def _perform_impact_analysis(self,
                               error_context: Dict[str, Any],
                               generation_result: GenerationResult,
                               codebase_context: Optional[Dict[str, Any]]) -> ImpactAnalysis:
        """Perform impact analysis on the generated changes."""
        # Determine what entity is being changed
        file_path = error_context.get('file_path', '')
        
        # Analyze first change to determine entity
        if generation_result.changes:
            first_change = generation_result.changes[0]
            change_type = 'function'  # Default
            changed_entity = 'unknown'
            
            # Try to identify what's being changed
            original_code = first_change.get('original_code', '')
            if 'class ' in original_code:
                change_type = 'class'
                # Extract class name
                import re
                match = re.search(r'class\s+(\w+)', original_code)
                if match:
                    changed_entity = match.group(1)
            elif 'def ' in original_code:
                change_type = 'function'
                # Extract function name
                match = re.search(r'def\s+(\w+)', original_code)
                if match:
                    changed_entity = match.group(1)
            
            # Perform impact analysis
            return self.contextual_analyzer.analyze_change_impact(
                file_path,
                change_type,
                changed_entity,
                {
                    'changes': generation_result.changes,
                    'multi_file': bool(generation_result.multi_file_changes),
                    'semantic_confidence': generation_result.confidence
                }
            )
        
        # Return empty analysis if no changes
        return ImpactAnalysis(
            direct_impacts=[],
            indirect_impacts=[],
            affected_tests=[],
            risk_level='low',
            breaking_changes=[],
            suggested_validations=[]
        )
    
    def _create_patch_from_generation(self,
                                    generation_result: GenerationResult,
                                    error_context: Dict[str, Any],
                                    mode: CodeGenerationMode) -> Dict[str, Any]:
        """Create a patch structure from generation result."""
        return {
            'patch_id': str(Path(__file__).stat().st_mtime),  # Use timestamp as ID
            'patch_type': f'enhanced_{mode.value}',
            'error_context': error_context,
            'llm_analysis': generation_result.reasoning,
            'fix_type': mode.value,
            'changes': generation_result.changes,
            'test_suggestions': generation_result.test_suggestions,
            'confidence': generation_result.confidence,
            'generated_by': 'enhanced_llm',
            'file_path': error_context.get('file_path', ''),
            'language': error_context.get('language', 'unknown'),
            'metadata': {
                'semantic_analysis': generation_result.semantic_analysis,
                'dependency_updates': generation_result.dependency_updates,
                'generation_mode': mode.value
            }
        }
    
    def _perform_iterative_refinement(self,
                                    initial_patch: Dict[str, Any],
                                    error_context: Dict[str, Any],
                                    source_code: Optional[str],
                                    generation_result: GenerationResult) -> RefinementHistory:
        """Perform iterative refinement on the initial patch."""
        # Create initial validation feedback
        target_file = error_context.get('file_path', '')
        
        # Run initial validation
        initial_validation = self._run_test_validation(
            target_file,
            str(Path(target_file).parent) if target_file else None
        )
        
        # Create feedback from validation
        initial_feedback = ValidationFeedback(
            test_results={
                'success': initial_validation.success,
                'failed_tests': [],
                'errors': [initial_validation.error_output] if not initial_validation.success else []
            },
            static_analysis={},
            style_violations=[],
            semantic_issues=[],
            performance_metrics={},
            suggestions=generation_result.test_suggestions
        )
        
        # Get test command
        test_command = self._get_test_command(
            target_file,
            str(Path(target_file).parent) if target_file else '.'
        )
        
        # Perform refinement
        return self.iterative_refiner.refine_patch(
            initial_patch,
            initial_feedback,
            target_file,
            source_code or '',
            test_command
        )
    
    def _create_multi_file_patches(self,
                                 multi_file_changes: Dict[str, List[Dict[str, Any]]],
                                 error_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create patch structures for multi-file changes."""
        patches = {}
        
        for file_path, changes in multi_file_changes.items():
            patches[file_path] = {
                'patch_id': f"{error_context.get('patch_id', '')}_{Path(file_path).stem}",
                'patch_type': 'enhanced_multi_file',
                'error_context': error_context,
                'file_path': file_path,
                'changes': changes,
                'confidence': 0.8,  # Slightly lower confidence for coordinated changes
                'generated_by': 'enhanced_llm'
            }
        
        return patches
    
    def _calculate_overall_confidence(self,
                                    generation_result: GenerationResult,
                                    refinement_history: Optional[RefinementHistory],
                                    impact_analysis: Optional[ImpactAnalysis]) -> float:
        """Calculate overall confidence score."""
        confidence_factors = [generation_result.confidence]
        
        # Factor in refinement success
        if refinement_history:
            if refinement_history.convergence_achieved:
                confidence_factors.append(0.9)
            elif refinement_history.best_iteration is not None:
                # Get best iteration's improvement
                best_iter = refinement_history.iterations[refinement_history.best_iteration]
                confidence_factors.append(
                    best_iter.improvement_metrics.get('overall', 0.5)
                )
            else:
                confidence_factors.append(0.3)
        
        # Factor in impact analysis risk
        if impact_analysis:
            risk_confidence = {
                'low': 0.9,
                'medium': 0.7,
                'high': 0.5
            }
            confidence_factors.append(
                risk_confidence.get(impact_analysis.risk_level, 0.5)
            )
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def _create_comprehensive_reasoning(self,
                                      generation_result: GenerationResult,
                                      refinement_history: Optional[RefinementHistory],
                                      impact_analysis: Optional[ImpactAnalysis]) -> str:
        """Create comprehensive reasoning explanation."""
        reasoning_parts = []
        
        # Generation reasoning
        reasoning_parts.append(f"Initial Analysis: {generation_result.reasoning}")
        
        # Semantic understanding
        if generation_result.semantic_analysis:
            sem_conf = generation_result.semantic_analysis.get('semantic_confidence', 0)
            reasoning_parts.append(
                f"Semantic Understanding: {sem_conf:.0%} confidence in code comprehension"
            )
        
        # Impact analysis
        if impact_analysis:
            reasoning_parts.append(
                f"Impact Assessment: {impact_analysis.risk_level} risk level with "
                f"{len(impact_analysis.direct_impacts)} direct and "
                f"{len(impact_analysis.indirect_impacts)} indirect dependencies affected"
            )
        
        # Refinement summary
        if refinement_history:
            if refinement_history.convergence_achieved:
                reasoning_parts.append(
                    f"Refinement: Achieved convergence after "
                    f"{len(refinement_history.iterations)} iterations"
                )
            else:
                reasoning_parts.append(
                    f"Refinement: Completed {len(refinement_history.iterations)} "
                    f"improvement iterations"
                )
        
        return "\n\n".join(reasoning_parts)
    
    def _collect_all_warnings(self,
                            generation_result: GenerationResult,
                            refinement_history: Optional[RefinementHistory],
                            impact_analysis: Optional[ImpactAnalysis]) -> List[str]:
        """Collect all warnings from various analyses."""
        warnings = list(generation_result.warnings)
        
        # Add impact analysis warnings
        if impact_analysis:
            for breaking_change in impact_analysis.breaking_changes:
                warnings.append(
                    f"Breaking change: {breaking_change['description']}"
                )
        
        # Add refinement warnings
        if refinement_history:
            for iteration in refinement_history.iterations:
                if not iteration.success:
                    warnings.append(
                        f"Refinement iteration {iteration.iteration_number} "
                        f"had issues: {iteration.validation_results}"
                    )
        
        return warnings
    
    def learn_from_success(self,
                         patch_result: EnhancedPatchResult,
                         success_metrics: Dict[str, Any]):
        """
        Learn from successful patch applications.
        
        Args:
            patch_result: The successful patch result
            success_metrics: Metrics about the success
        """
        if not patch_result.success:
            return
        
        # Extract task type and create example
        task_type = self._determine_task_type(patch_result.patch.get('error_context', {}))
        
        # Create prompt example from successful patch
        example = PromptExample(
            context=patch_result.patch.get('error_context', {}).get('language', 'unknown'),
            problem=patch_result.patch.get('error_context', {}).get('error_message', ''),
            solution=str(patch_result.patch.get('changes', [])),
            explanation=patch_result.reasoning,
            tags=[
                patch_result.patch.get('language', 'unknown'),
                patch_result.generation_mode.value,
                f"confidence_{patch_result.confidence:.1f}"
            ]
        )
        
        # Add to prompt manager
        effectiveness = success_metrics.get('effectiveness', patch_result.confidence)
        self.prompt_manager.add_learned_example(task_type, example, effectiveness)
        
        # Log learning
        logger.info(
            f"Learned from successful {task_type} patch with "
            f"{effectiveness:.0%} effectiveness"
        )


def create_enhanced_llm_patch_generator(config: Optional[Dict[str, Any]] = None) -> EnhancedLLMPatchGenerator:
    """
    Factory function to create an enhanced LLM patch generator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured enhanced LLM patch generator
    """
    api_key_manager = APIKeyManager()
    context_manager = LLMContextManager()
    
    return EnhancedLLMPatchGenerator(
        api_key_manager=api_key_manager,
        context_manager=context_manager,
        config=config
    )