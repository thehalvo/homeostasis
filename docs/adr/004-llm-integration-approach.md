# ADR-004: LLM Integration Approach

Technical Story: #ARCH-004

## Context

While rule-based error detection and fixing works well for known patterns, many software errors are unique or complex, requiring understanding of code semantics and intent. Large Language Models (LLMs) have shown remarkable ability to understand and generate code. We need to decide how to integrate LLMs into Homeostasis while maintaining reliability, performance, and cost-effectiveness.

## Decision Drivers

- Accuracy: High-quality fix generation for complex errors
- Reliability: Consistent results and fallback mechanisms
- Performance: Acceptable response times for fix generation
- Cost: Manage API costs for LLM providers
- Privacy: Protect sensitive code and data
- Flexibility: Support multiple LLM providers
- Control: Ability to guide and constrain LLM outputs

## Considered Options

1. **LLM-Only Approach** - Replace all logic with LLM calls
2. **Rule-Based Only** - No LLM integration
3. **Hybrid Approach** - Rules first, LLM for complex cases
4. **LLM Validation** - Generate with rules, validate with LLM
5. **Local LLM Deployment** - Self-hosted models only

## Decision Outcome

Chosen option: "Hybrid Approach", using rule-based detection and fixes as the primary mechanism, with LLM integration for complex cases that rules cannot handle, because it balances accuracy, cost, and performance while maintaining system reliability.

### Positive Consequences

- **Cost Efficiency**: LLM used only when necessary
- **Fast Response**: Most fixes use quick rule-based approach
- **High Success Rate**: LLM handles edge cases rules miss
- **Predictable Behavior**: Rules provide consistency
- **Gradual Improvement**: Can expand rules based on LLM insights
- **Privacy Control**: Sensitive code can skip LLM
- **Provider Flexibility**: Can switch LLM providers easily

### Negative Consequences

- **Complexity**: Two different fix generation paths
- **Maintenance**: Need to maintain both rules and prompts
- **Decision Logic**: Must determine when to use LLM
- **Inconsistency Risk**: LLM and rules may suggest different fixes
- **Integration Overhead**: Additional API management
- **Prompt Engineering**: Requires ongoing optimization

## Implementation Details

### Decision Flow

```python
def generate_fix(error: Error, context: Context) -> Fix:
    # Step 1: Try rule-based fix
    rule_fix = rule_engine.attempt_fix(error)
    if rule_fix and rule_fix.confidence > 0.8:
        return rule_fix
    
    # Step 2: Check if LLM is appropriate
    if not should_use_llm(error, context):
        return rule_fix or Fix.none()
    
    # Step 3: Prepare context for LLM
    llm_context = prepare_llm_context(error, context)
    
    # Step 4: Generate fix with LLM
    llm_fix = llm_provider.generate_fix(llm_context)
    
    # Step 5: Validate and sanitize
    validated_fix = validate_llm_output(llm_fix)
    
    # Step 6: Choose best fix
    return choose_best_fix(rule_fix, validated_fix)
```

### LLM Provider Interface

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_fix(self, context: LLMContext) -> LLMResponse:
        pass
    
    @abstractmethod
    def estimate_cost(self, context: LLMContext) -> float:
        pass
    
    @abstractmethod
    def check_content_policy(self, content: str) -> bool:
        pass
```

### Supported Providers

1. **OpenAI GPT-4** - Primary provider
2. **Anthropic Claude** - Secondary provider
3. **Local Models** - Llama 2, CodeLlama for sensitive code
4. **Google PaLM** - Experimental support

### Prompt Engineering

#### Error Analysis Prompt Template
```
You are a senior software engineer debugging code. Analyze this error:

Error Type: {error_type}
Error Message: {error_message}
File: {file_path}
Line: {line_number}

Code Context:
```
{code_context}
```

Recent Changes:
```
{recent_changes}
```

Identify the root cause and explain the issue concisely.
```

#### Fix Generation Prompt Template
```
Based on this error analysis:
{error_analysis}

Generate a minimal fix that:
1. Resolves the error
2. Maintains existing functionality
3. Follows the codebase style
4. Includes necessary imports

Provide only the code change, no explanation.
```

### Context Preparation

```python
def prepare_llm_context(error: Error, context: Context) -> LLMContext:
    return LLMContext(
        error=sanitize_error(error),
        code_context=extract_relevant_code(context, max_lines=50),
        recent_changes=get_recent_changes(context.file_path, max_commits=5),
        dependencies=extract_dependencies(context),
        test_cases=find_related_tests(context),
        codebase_patterns=extract_patterns(context)
    )
```

### Cost Management

- **Token Limits**: Maximum 4K tokens per request
- **Caching**: Cache similar error fixes for 24 hours
- **Batching**: Group similar errors when possible
- **Fallback**: Use cheaper models for simple cases
- **Budget Alerts**: Notify when approaching limits

### Privacy and Security

```python
def should_use_llm(error: Error, context: Context) -> bool:
    # Check privacy settings
    if context.privacy_level == "strict":
        return False
    
    # Check for sensitive patterns
    if contains_sensitive_data(context.code):
        return False
    
    # Check error complexity
    if error.complexity_score < LLM_THRESHOLD:
        return False
    
    # Check cost budget
    if daily_llm_cost > DAILY_BUDGET:
        return False
    
    return True
```

### Output Validation

```python
def validate_llm_output(fix: LLMFix) -> ValidatedFix:
    # Syntax validation
    if not is_valid_syntax(fix.code):
        return ValidatedFix.invalid()
    
    # Security scanning
    if has_security_issues(fix.code):
        return ValidatedFix.invalid()
    
    # Semantic validation
    if not maintains_functionality(fix.code):
        return ValidatedFix.invalid()
    
    # Style checking
    fix.code = apply_code_style(fix.code)
    
    return ValidatedFix(fix.code, confidence=calculate_confidence(fix))
```

### Monitoring and Feedback

- Track LLM success rates
- Monitor cost per fix
- Measure fix quality scores
- Collect user feedback
- Update rules based on LLM patterns

### Fallback Strategy

1. If LLM fails, fall back to rule-based fix
2. If LLM is unavailable, queue for later
3. If cost limit reached, use rules only
4. If privacy concerns, use local model

## Links

- [LLM Provider Extension Guide](../llm_provider_extension_guide.md)
- [ADR-006: Security Approval Workflow](006-security-approval-workflow.md)
- [Cost Management Documentation](../api_keys.md)