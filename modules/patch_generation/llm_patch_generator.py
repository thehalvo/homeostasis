#!/usr/bin/env python3
"""
LLM-Powered Universal Patch Generator

This module provides LLM-based patch generation that can handle virtually any
type of defect by leveraging the language understanding capabilities of LLMs
rather than relying solely on predefined templates.
"""

import ast
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from ..llm_integration.provider_abstraction import (
    LLMManager, LLMRequest, LLMMessage, LLMError
)
from ..llm_integration.api_key_manager import APIKeyManager
from ..analysis.llm_context_manager import LLMContextManager, LLMContext
from .multi_language_framework_detector import (
    MultiLanguageFrameworkDetector, LanguageType, create_multi_language_detector
)
from .code_style_analyzer import CodeStyleAnalyzer, create_code_style_analyzer


logger = logging.getLogger(__name__)


class LLMPatchGenerator:
    """
    Universal patch generator powered by Large Language Models.
    
    This generator can handle any type of code defect by:
    1. Analyzing the error context using LLM understanding
    2. Generating appropriate fixes based on the specific error and code context
    3. Supporting multiple programming languages and frameworks
    4. Preserving code style and structure
    """

    def __init__(self, 
                 api_key_manager: Optional[APIKeyManager] = None,
                 context_manager: Optional[LLMContextManager] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM patch generator.
        
        Args:
            api_key_manager: API key manager for LLM providers
            context_manager: LLM context manager for storing error contexts
            config: Configuration for patch generation
        """
        self.api_key_manager = api_key_manager or APIKeyManager()
        self.llm_manager = LLMManager(self.api_key_manager)
        self.context_manager = context_manager or LLMContextManager()
        self.language_detector = create_multi_language_detector()
        self.style_analyzer = create_code_style_analyzer()
        
        # Configuration
        self.config = config or {}
        self.max_context_length = self.config.get('max_context_length', 8000)
        self.temperature = self.config.get('temperature', 0.1)  # Low temperature for consistent code
        self.max_tokens = self.config.get('max_tokens', 2000)
        
        # Language-specific configurations
        self.language_configs = {
            'python': {
                'file_extensions': ['.py'],
                'comment_prefix': '#',
                'indent_size': 4,
                'common_imports': ['import os', 'import sys', 'from typing import'],
            },
            'javascript': {
                'file_extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'comment_prefix': '//',
                'indent_size': 2,
                'common_imports': ['const', 'import', 'require'],
            },
            'java': {
                'file_extensions': ['.java'],
                'comment_prefix': '//',
                'indent_size': 4,
                'common_imports': ['import', 'package'],
            },
            'go': {
                'file_extensions': ['.go'],
                'comment_prefix': '//',
                'indent_size': 1,  # Go uses tabs
                'common_imports': ['import', 'package'],
            },
            'rust': {
                'file_extensions': ['.rs'],
                'comment_prefix': '//',
                'indent_size': 4,
                'common_imports': ['use', 'mod'],
            },
            'swift': {
                'file_extensions': ['.swift'],
                'comment_prefix': '//',
                'indent_size': 4,
                'common_imports': ['import', 'class', 'struct'],
            }
        }
        
        logger.info("Initialized LLM Patch Generator")

    def generate_patch_from_error_context(self, 
                                         error_context: Dict[str, Any],
                                         source_code: Optional[str] = None,
                                         additional_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a patch from error context using LLM analysis.
        
        Args:
            error_context: Error context information
            source_code: Source code containing the error
            additional_context: Additional context like project structure, dependencies
            
        Returns:
            Generated patch information or None if generation fails
        """
        try:
            # Store the error context for LLM processing
            context_id = self.context_manager.store_error_context(
                error_context=error_context,
                additional_analysis=additional_context
            )
            
            # Prepare LLM prompt for patch generation
            prompt_data = self.context_manager.prepare_llm_prompt(
                context_id, 
                template_type="patch_generation"
            )
            
            # Enhance prompt with source code if available
            if source_code:
                prompt_data = self._enhance_prompt_with_source_code(
                    prompt_data, 
                    source_code, 
                    error_context
                )
            
            # Generate patch using LLM
            patch_response = self._generate_llm_patch(prompt_data, error_context)
            
            if patch_response:
                # Process and validate the generated patch
                processed_patch = self._process_llm_response(
                    patch_response, 
                    error_context, 
                    source_code
                )
                
                # Update context with results
                self.context_manager.update_context_with_results(
                    context_id,
                    {"patch_generated": True, "patch_content": processed_patch}
                )
                
                return processed_patch
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating LLM patch: {e}")
            return None

    def generate_patch_from_analysis(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a patch from analysis results using LLM.
        
        Args:
            analysis_result: Analysis results from error detection
            
        Returns:
            Generated patch information or None if generation fails
        """
        # Extract key information from analysis
        error_type = analysis_result.get('error_type', 'unknown')
        error_message = analysis_result.get('error_message', '')
        file_path = analysis_result.get('file_path', '')
        line_number = analysis_result.get('line_number', 0)
        root_cause = analysis_result.get('root_cause', '')
        confidence = analysis_result.get('confidence', 0.0)
        
        # Use comprehensive language and framework detection
        language_info = self.language_detector.detect_language_and_frameworks(
            file_path=file_path,
            source_code=source_code
        )
        language = language_info.language.value
        
        # Read source code if file path is provided
        source_code = None
        if file_path and Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            except Exception as e:
                logger.warning(f"Could not read source file {file_path}: {e}")
        
        # Create error context with comprehensive language information
        error_context = {
            'error_type': error_type,
            'error_message': error_message,
            'file_path': file_path,
            'line_number': line_number,
            'root_cause': root_cause,
            'confidence': confidence,
            'language': language,
            'language_info': language_info,
            'analysis_result': analysis_result
        }
        
        # Generate patch
        return self.generate_patch_from_error_context(
            error_context,
            source_code,
            {'analysis_method': 'rule_based'}
        )

    def _enhance_prompt_with_source_code(self, 
                                        prompt_data: Dict[str, Any], 
                                        source_code: str,
                                        error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the LLM prompt with source code context.
        
        Args:
            prompt_data: Original prompt data
            source_code: Source code to include
            error_context: Error context information
            
        Returns:
            Enhanced prompt data
        """
        # Extract relevant source code section
        relevant_code = self._extract_relevant_code_section(
            source_code, 
            error_context.get('line_number', 0),
            context_lines=10
        )
        
        # Get comprehensive language information
        language_info = error_context.get('language_info')
        if language_info:
            # Use comprehensive language info
            language = language_info.language.value
            frameworks = [f.name for f in language_info.frameworks]
            language_features = language_info.language_features
            llm_context = self.language_detector.get_llm_context_for_language(language_info)
        else:
            # Fallback to basic language detection
            language = error_context.get('language', 'unknown')
            frameworks = []
            language_features = self.language_configs.get(language, {})
            llm_context = {'language': language}
        
        # Analyze code style from the source code
        style_conventions = None
        if source_code:
            style_conventions = self.style_analyzer.analyze_file_style(
                error_context.get('file_path', ''),
                language,
                source_code
            )
        
        # Build enhanced prompt
        enhanced_prompt = prompt_data.get('user_prompt', '')
        enhanced_prompt += f"""

RELEVANT SOURCE CODE:
```{language}
{relevant_code}
```

LANGUAGE CONTEXT:
- Language: {language}
- Frameworks: {', '.join(frameworks) if frameworks else 'None detected'}
- Comment style: {language_features.get('comment_style', '#')}
- Indent style: {language_features.get('indent_style', 'spaces')}
- Typical indent: {language_features.get('typical_indent', 4)}
- Language guidance: {llm_context.get('llm_guidance', {})}

CODE STYLE CONVENTIONS:
{self._format_style_conventions(style_conventions) if style_conventions else 'No specific style conventions detected. Use language defaults.'}

FRAMEWORK-SPECIFIC GUIDANCE:
{self._get_framework_specific_guidance(frameworks, language)}

REQUIREMENTS:
1. Generate a complete, working fix that addresses the specific error
2. Preserve the existing code style and indentation patterns
3. Follow the detected language and framework conventions
4. Include only the necessary changes, don't rewrite unrelated code
5. Ensure the fix is compatible with the detected frameworks: {', '.join(frameworks) if frameworks else 'standard language features'}
6. Provide the fix in a structured format that can be easily applied

OUTPUT FORMAT:
Provide your response in the following JSON format:
{{
    "analysis": "Brief analysis of the issue and proposed solution",
    "fix_type": "Type of fix (e.g., 'syntax_fix', 'logic_fix', 'import_fix', etc.)",
    "changes": [
        {{
            "line_start": <start_line_number>,
            "line_end": <end_line_number>,
            "original_code": "original code to replace",
            "new_code": "new code to insert",
            "reason": "explanation of why this change is needed"
        }}
    ],
    "test_suggestions": ["list of tests that should be added or modified"],
    "confidence": 0.9
}}
"""
        
        prompt_data['user_prompt'] = enhanced_prompt
        prompt_data['max_tokens'] = self.max_tokens
        
        return prompt_data

    def _extract_relevant_code_section(self, 
                                     source_code: str, 
                                     error_line: int, 
                                     context_lines: int = 10) -> str:
        """
        Extract a relevant section of source code around the error line.
        
        Args:
            source_code: Full source code
            error_line: Line number where error occurred
            context_lines: Number of context lines to include
            
        Returns:
            Relevant code section with line numbers
        """
        lines = source_code.split('\n')
        
        # Calculate the range to extract
        start_line = max(0, error_line - context_lines - 1)  # -1 for 0-based indexing
        end_line = min(len(lines), error_line + context_lines)
        
        # Extract relevant lines with line numbers
        relevant_lines = []
        for i in range(start_line, end_line):
            line_num = i + 1
            marker = " -> " if line_num == error_line else "    "
            relevant_lines.append(f"{line_num:4d}{marker}{lines[i]}")
        
        return '\n'.join(relevant_lines)

    def _generate_llm_patch(self, 
                          prompt_data: Dict[str, Any], 
                          error_context: Dict[str, Any]) -> Optional[str]:
        """
        Generate a patch using the LLM.
        
        Args:
            prompt_data: Prepared prompt data
            error_context: Error context information
            
        Returns:
            LLM response or None if generation fails
        """
        try:
            # Create LLM request
            messages = []
            
            # Add system prompt
            system_prompt = prompt_data.get('system_prompt', 
                "You are an expert software engineer. Generate precise code fixes for the given errors.")
            
            # Add user prompt
            user_prompt = prompt_data.get('user_prompt', '')
            messages.append(LLMMessage(role='user', content=user_prompt))
            
            # Create request
            request = LLMRequest(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=prompt_data.get('max_tokens', self.max_tokens),
                temperature=self.temperature
            )
            
            # Get response from LLM
            response = self.llm_manager.complete(request)
            
            logger.info(f"Generated LLM patch response: {len(response.content)} characters")
            return response.content
            
        except LLMError as e:
            logger.error(f"LLM generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during LLM patch generation: {e}")
            return None

    def _process_llm_response(self, 
                            llm_response: str, 
                            error_context: Dict[str, Any],
                            source_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Process the LLM response into a structured patch format.
        
        Args:
            llm_response: Raw LLM response
            error_context: Error context information
            source_code: Original source code
            
        Returns:
            Processed patch information
        """
        try:
            # Try to parse JSON response
            patch_data = self._extract_json_from_response(llm_response)
            
            if patch_data:
                # Format changes according to detected style
                formatted_changes = self._format_changes_with_style(
                    patch_data.get('changes', []),
                    error_context.get('language', 'unknown'),
                    source_code
                )
                
                # Create structured patch
                return {
                    'patch_id': str(uuid.uuid4()),
                    'patch_type': 'llm_generated',
                    'error_context': error_context,
                    'llm_analysis': patch_data.get('analysis', ''),
                    'fix_type': patch_data.get('fix_type', 'unknown'),
                    'changes': formatted_changes,
                    'test_suggestions': patch_data.get('test_suggestions', []),
                    'confidence': patch_data.get('confidence', 0.5),
                    'generated_by': 'llm',
                    'raw_response': llm_response,
                    'file_path': error_context.get('file_path', ''),
                    'language': error_context.get('language', 'unknown')
                }
            else:
                # Fallback: try to extract code from response
                code_changes = self._extract_code_from_response(llm_response)
                
                return {
                    'patch_id': str(uuid.uuid4()),
                    'patch_type': 'llm_generated_fallback',
                    'error_context': error_context,
                    'llm_analysis': 'LLM provided code changes without structured format',
                    'fix_type': 'code_replacement',
                    'changes': code_changes,
                    'confidence': 0.3,  # Lower confidence for unstructured response
                    'generated_by': 'llm',
                    'raw_response': llm_response,
                    'file_path': error_context.get('file_path', ''),
                    'language': error_context.get('language', 'unknown')
                }
                
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            
            # Return basic patch structure
            return {
                'patch_id': str(uuid.uuid4()),
                'patch_type': 'llm_error',
                'error_context': error_context,
                'llm_analysis': f'Error processing LLM response: {e}',
                'confidence': 0.1,
                'generated_by': 'llm',
                'raw_response': llm_response,
                'file_path': error_context.get('file_path', ''),
                'language': error_context.get('language', 'unknown')
            }

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON data from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON data or None if extraction fails
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Try parsing the entire response as JSON
            return json.loads(response)
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _extract_code_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from LLM response as a fallback.
        
        Args:
            response: LLM response text
            
        Returns:
            List of code changes
        """
        changes = []
        
        # Look for code blocks in markdown format
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        
        for i, code_block in enumerate(code_blocks):
            changes.append({
                'line_start': 1,  # Default values since we don't have specific line info
                'line_end': -1,
                'original_code': '',
                'new_code': code_block.strip(),
                'reason': f'Code block {i+1} from LLM response'
            })
        
        return changes

    def _get_framework_specific_guidance(self, frameworks: List[str], language: str) -> str:
        """
        Get framework-specific guidance for patch generation.
        
        Args:
            frameworks: List of detected frameworks
            language: Programming language
            
        Returns:
            Framework-specific guidance text
        """
        if not frameworks:
            return "No specific frameworks detected. Follow standard language conventions."
        
        guidance_map = {
            'django': [
                "- Use Django's ORM patterns and model methods",
                "- Follow Django naming conventions for models, views, and URLs",
                "- Use Django's built-in error handling and validation",
                "- Consider middleware and decorator patterns",
                "- Use Django's template system appropriately"
            ],
            'flask': [
                "- Use Flask's route decorators and blueprints",
                "- Follow Flask application factory patterns",
                "- Use Flask's request context appropriately",
                "- Handle errors with Flask's error handlers",
                "- Use Flask extensions following their conventions"
            ],
            'fastapi': [
                "- Use FastAPI's dependency injection system",
                "- Leverage Pydantic models for validation",
                "- Use async/await patterns where appropriate",
                "- Follow FastAPI's path operation decorators",
                "- Use FastAPI's automatic documentation features"
            ],
            'react': [
                "- Use React hooks (useState, useEffect) properly",
                "- Follow React component lifecycle patterns",
                "- Use JSX syntax correctly",
                "- Handle state immutably",
                "- Use proper key props for lists"
            ],
            'vue': [
                "- Use Vue's composition API or options API consistently",
                "- Follow Vue's reactivity patterns",
                "- Use Vue's template syntax correctly",
                "- Handle component props and events properly",
                "- Use Vue's lifecycle hooks appropriately"
            ],
            'angular': [
                "- Use Angular's dependency injection",
                "- Follow Angular's component structure",
                "- Use Angular's service patterns",
                "- Handle observables with RxJS",
                "- Use Angular's template syntax"
            ],
            'spring': [
                "- Use Spring's annotation-based configuration",
                "- Follow Spring Boot conventions",
                "- Use dependency injection properly",
                "- Handle transactions with @Transactional",
                "- Use Spring's exception handling"
            ],
            'gin': [
                "- Use Gin's router and middleware patterns",
                "- Handle HTTP context properly",
                "- Use Gin's binding and validation",
                "- Follow Go's error handling conventions",
                "- Use structured logging"
            ],
            'actix': [
                "- Use Actix Web's handler patterns",
                "- Handle async operations with Rust's async/await",
                "- Use Actix's middleware system",
                "- Follow Rust's ownership and borrowing rules",
                "- Use proper error handling with Result types"
            ]
        }
        
        guidance_lines = []
        for framework in frameworks:
            if framework in guidance_map:
                guidance_lines.append(f"\n{framework.upper()} Framework Guidelines:")
                guidance_lines.extend(guidance_map[framework])
        
        if not guidance_lines:
            return f"Detected frameworks: {', '.join(frameworks)}. Follow their standard conventions."
        
        return '\n'.join(guidance_lines)

    def _format_style_conventions(self, style_conventions) -> str:
        """
        Format style conventions for inclusion in LLM prompts.
        
        Args:
            style_conventions: Detected style conventions
            
        Returns:
            Formatted style information
        """
        if not style_conventions:
            return "No style conventions detected."
        
        lines = [
            f"- Indentation: {style_conventions.indent_size} {style_conventions.indent_style}",
            f"- Quote style: {style_conventions.quote_style} quotes",
            f"- Line length: {style_conventions.line_length} characters max",
        ]
        
        # Add naming conventions
        if style_conventions.naming_conventions:
            lines.append("- Naming conventions:")
            for item_type, convention in style_conventions.naming_conventions.items():
                lines.append(f"  - {item_type}: {convention}")
        
        # Add architectural patterns
        if style_conventions.architectural_patterns:
            lines.append(f"- Architectural patterns: {', '.join(style_conventions.architectural_patterns)}")
        
        lines.append(f"- Style confidence: {style_conventions.confidence:.2f}")
        
        return '\n'.join(lines)

    def _format_changes_with_style(self, 
                                  changes: List[Dict[str, Any]], 
                                  language: str,
                                  source_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format code changes according to detected style conventions.
        
        Args:
            changes: List of code changes
            language: Programming language
            source_code: Original source code for style analysis
            
        Returns:
            List of formatted changes
        """
        if not changes or not source_code:
            return changes
        
        # Analyze style from source code
        try:
            style_conventions = self.style_analyzer.analyze_file_style(
                '', language, source_code
            )
            
            formatted_changes = []
            for change in changes:
                formatted_change = change.copy()
                
                # Format the new code according to style conventions
                new_code = change.get('new_code', '')
                if new_code:
                    formatted_code = self.style_analyzer.format_code_to_style(
                        new_code, style_conventions, language
                    )
                    formatted_change['new_code'] = formatted_code
                
                formatted_changes.append(formatted_change)
            
            return formatted_changes
            
        except Exception as e:
            logger.warning(f"Could not format changes with style: {e}")
            return changes

    def apply_llm_patch(self, 
                       patch: Dict[str, Any], 
                       target_file: str) -> bool:
        """
        Apply an LLM-generated patch to a target file.
        
        Args:
            patch: LLM-generated patch information
            target_file: Path to target file
            
        Returns:
            True if patch was applied successfully
        """
        try:
            target_path = Path(target_file)
            if not target_path.exists():
                logger.error(f"Target file does not exist: {target_file}")
                return False
            
            # Read current file content
            with open(target_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply changes
            modified_content = self._apply_changes_to_content(
                original_content,
                patch.get('changes', [])
            )
            
            if modified_content != original_content:
                # Create backup
                backup_path = target_path.with_suffix(target_path.suffix + '.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write modified content
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                logger.info(f"Applied LLM patch to {target_file}")
                return True
            else:
                logger.warning(f"No changes applied to {target_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying LLM patch: {e}")
            return False

    def _apply_changes_to_content(self, 
                                content: str, 
                                changes: List[Dict[str, Any]]) -> str:
        """
        Apply a list of changes to file content.
        
        Args:
            content: Original file content
            changes: List of changes to apply
            
        Returns:
            Modified content
        """
        lines = content.split('\n')
        
        # Sort changes by line number in reverse order to avoid offset issues
        sorted_changes = sorted(
            changes, 
            key=lambda x: x.get('line_start', 0), 
            reverse=True
        )
        
        for change in sorted_changes:
            start_line = change.get('line_start', 1) - 1  # Convert to 0-based
            end_line = change.get('line_end', start_line + 1) - 1
            new_code = change.get('new_code', '').split('\n')
            
            # Validate line numbers
            if start_line < 0 or start_line >= len(lines):
                logger.warning(f"Invalid start line {start_line + 1}")
                continue
            
            if end_line < start_line:
                end_line = start_line
            
            if end_line >= len(lines):
                end_line = len(lines) - 1
            
            # Apply the change
            lines[start_line:end_line + 1] = new_code
        
        return '\n'.join(lines)

    def get_patch_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about LLM patch generation.
        
        Returns:
            Statistics dictionary
        """
        # This would typically query the context manager for statistics
        return {
            'total_patches_generated': 0,  # Would be implemented with actual tracking
            'success_rate': 0.0,
            'average_confidence': 0.0,
            'supported_languages': list(self.language_configs.keys()),
            'available_providers': self.llm_manager.get_available_providers()
        }


def create_llm_patch_generator(config: Optional[Dict[str, Any]] = None) -> LLMPatchGenerator:
    """
    Factory function to create an LLM patch generator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LLM patch generator
    """
    api_key_manager = APIKeyManager()
    context_manager = LLMContextManager()
    
    return LLMPatchGenerator(
        api_key_manager=api_key_manager,
        context_manager=context_manager,
        config=config
    )


if __name__ == "__main__":
    # Test the LLM patch generator
    print("Testing LLM Patch Generator")
    print("==========================")
    
    # Create generator
    generator = create_llm_patch_generator()
    
    # Test error context
    test_error = {
        'error_type': 'NameError',
        'error_message': "name 'undefined_variable' is not defined",
        'file_path': 'test.py',
        'line_number': 10,
        'root_cause': 'undefined_variable',
        'confidence': 0.9,
        'language': 'python'
    }
    
    # Test source code
    test_source = '''
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    
    # Error on the next line - undefined_variable is not defined
    tax_rate = undefined_variable
    
    return total * (1 + tax_rate)
'''
    
    # Generate patch
    patch = generator.generate_patch_from_error_context(
        test_error,
        test_source
    )
    
    if patch:
        print(f"Generated patch: {patch['patch_id']}")
        print(f"Fix type: {patch['fix_type']}")
        print(f"Confidence: {patch['confidence']}")
        print(f"Analysis: {patch['llm_analysis']}")
    else:
        print("No patch generated")
    
    # Get statistics
    stats = generator.get_patch_statistics()
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Available providers: {stats['available_providers']}")