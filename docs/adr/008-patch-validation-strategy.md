# ADR-008: Patch Validation Strategy

Technical Story: #PATCH-001

## Context

Generated patches must be thoroughly validated before deployment to ensure they fix the issue without introducing new problems. Validation must be comprehensive yet fast enough to maintain the self-healing system's responsiveness. We need a multi-layered validation strategy that catches different types of issues while optimizing for speed and accuracy.

## Decision Drivers

- Safety: Patches must not break existing functionality
- Speed: Validation should be fast for rapid healing
- Accuracy: Catch all potential issues before deployment
- Coverage: Test various aspects (syntax, logic, performance)
- Confidence: Provide metrics on patch quality
- Scalability: Handle multiple patches concurrently
- Language Support: Work across all supported languages

## Considered Options

1. **Unit Tests Only** - Run only unit tests
2. **Full Test Suite** - Run all available tests
3. **Progressive Validation** - Start simple, escalate if needed
4. **Statistical Validation** - Sample-based testing
5. **Multi-Stage Pipeline** - Different validation levels

## Decision Outcome

Chosen option: "Multi-Stage Pipeline", implementing a progressive validation pipeline with early exit on failure, because it provides the best balance between thoroughness and speed while allowing quick deployment of simple fixes.

### Positive Consequences

- **Fast Feedback**: Simple issues fail fast
- **Comprehensive Coverage**: Complex issues get full validation
- **Resource Efficiency**: Only use resources when needed
- **Parallel Execution**: Multiple validations run concurrently
- **Confidence Scoring**: Clear metrics on patch quality
- **Customizable**: Stages can be configured per environment
- **Language Agnostic**: Works with any language plugin

### Negative Consequences

- **Pipeline Complexity**: Multiple stages to manage
- **Configuration Overhead**: Each stage needs setup
- **Debugging Difficulty**: Issues may appear in any stage
- **Time Investment**: Full pipeline can be slow
- **False Negatives**: Early stages might miss issues
- **Resource Planning**: Hard to predict resource needs

## Implementation Details

### Validation Pipeline Stages

```python
class ValidationPipeline:
    def __init__(self):
        self.stages = [
            SyntaxValidation(),      # Stage 1: ~1 second
            StaticAnalysis(),        # Stage 2: ~5 seconds
            UnitTestValidation(),    # Stage 3: ~30 seconds
            IntegrationTestValidation(), # Stage 4: ~2 minutes
            PerformanceValidation(), # Stage 5: ~5 minutes
            SecurityValidation(),    # Stage 6: ~3 minutes
            ChaosValidation()       # Stage 7: ~10 minutes (optional)
        ]
    
    async def validate(self, patch: Patch) -> ValidationResult:
        results = []
        
        for stage in self.stages:
            if not self._should_run_stage(stage, patch, results):
                continue
                
            result = await stage.validate(patch)
            results.append(result)
            
            if not result.passed:
                return ValidationResult(
                    passed=False,
                    stage_failed=stage.name,
                    results=results
                )
        
        return ValidationResult(
            passed=True,
            confidence=self._calculate_confidence(results),
            results=results
        )
```

### Stage 1: Syntax Validation

```python
class SyntaxValidation(ValidationStage):
    def __init__(self):
        self.timeout = 1  # second
        
    async def validate(self, patch: Patch) -> StageResult:
        language_plugin = self.get_language_plugin(patch)
        
        try:
            # Parse the patched code
            ast = language_plugin.parse(patch.new_code)
            
            # Check for syntax errors
            if ast.errors:
                return StageResult(
                    passed=False,
                    errors=ast.errors,
                    duration=self.timer.elapsed()
                )
            
            # Verify imports/dependencies
            missing_imports = language_plugin.check_imports(ast)
            if missing_imports:
                return StageResult(
                    passed=False,
                    errors=[f"Missing import: {imp}" for imp in missing_imports],
                    duration=self.timer.elapsed()
                )
            
            return StageResult(passed=True, duration=self.timer.elapsed())
            
        except TimeoutError:
            return StageResult(
                passed=False,
                errors=["Syntax validation timeout"],
                duration=self.timeout
            )
```

### Stage 2: Static Analysis

```python
class StaticAnalysis(ValidationStage):
    def __init__(self):
        self.timeout = 5  # seconds
        self.tools = {
            'python': ['pylint', 'mypy', 'bandit'],
            'javascript': ['eslint', 'tslint', 'jshint'],
            'java': ['spotbugs', 'checkstyle', 'pmd'],
            'go': ['golint', 'go vet', 'staticcheck']
        }
    
    async def validate(self, patch: Patch) -> StageResult:
        language = patch.language
        tools = self.tools.get(language, [])
        
        issues = []
        for tool in tools:
            tool_issues = await self._run_tool(tool, patch)
            issues.extend(tool_issues)
        
        # Filter issues by severity
        critical_issues = [i for i in issues if i.severity == 'critical']
        
        if critical_issues:
            return StageResult(
                passed=False,
                errors=critical_issues,
                warnings=[i for i in issues if i.severity == 'warning'],
                duration=self.timer.elapsed()
            )
        
        return StageResult(
            passed=True,
            warnings=[i for i in issues if i.severity == 'warning'],
            duration=self.timer.elapsed()
        )
```

### Stage 3: Unit Test Validation

```python
class UnitTestValidation(ValidationStage):
    def __init__(self):
        self.timeout = 30  # seconds
        
    async def validate(self, patch: Patch) -> StageResult:
        # Find affected unit tests
        affected_tests = self._find_affected_tests(patch)
        
        if not affected_tests:
            # No tests found, try to generate them
            affected_tests = await self._generate_tests(patch)
        
        # Run tests in parallel
        test_results = await asyncio.gather(*[
            self._run_test(test) for test in affected_tests
        ])
        
        failed_tests = [r for r in test_results if not r.passed]
        
        if failed_tests:
            return StageResult(
                passed=False,
                errors=[f"Test failed: {t.name}" for t in failed_tests],
                test_coverage=self._calculate_coverage(test_results),
                duration=self.timer.elapsed()
            )
        
        return StageResult(
            passed=True,
            test_coverage=self._calculate_coverage(test_results),
            tests_run=len(test_results),
            duration=self.timer.elapsed()
        )
```

### Stage 4: Integration Test Validation

```python
class IntegrationTestValidation(ValidationStage):
    def __init__(self):
        self.timeout = 120  # seconds
        
    async def validate(self, patch: Patch) -> StageResult:
        # Deploy to test environment
        test_env = await self._provision_test_environment(patch)
        
        try:
            # Run integration test suite
            test_suite = self._get_integration_suite(patch)
            results = await test_suite.run(test_env)
            
            # Check for regressions
            baseline = await self._get_baseline_results(patch)
            regressions = self._compare_results(baseline, results)
            
            if regressions:
                return StageResult(
                    passed=False,
                    errors=[f"Regression: {r}" for r in regressions],
                    duration=self.timer.elapsed()
                )
            
            return StageResult(
                passed=True,
                tests_passed=results.passed_count,
                duration=self.timer.elapsed()
            )
            
        finally:
            await test_env.cleanup()
```

### Stage 5: Performance Validation

```python
class PerformanceValidation(ValidationStage):
    def __init__(self):
        self.timeout = 300  # seconds
        self.thresholds = {
            'response_time_p95': 1.1,  # 10% degradation allowed
            'throughput': 0.9,          # 10% degradation allowed
            'cpu_usage': 1.2,           # 20% increase allowed
            'memory_usage': 1.15        # 15% increase allowed
        }
    
    async def validate(self, patch: Patch) -> StageResult:
        # Run performance tests
        baseline = await self._get_performance_baseline(patch)
        patched = await self._measure_performance(patch)
        
        violations = []
        for metric, threshold in self.thresholds.items():
            baseline_value = baseline.get(metric)
            patched_value = patched.get(metric)
            
            if patched_value > baseline_value * threshold:
                violations.append(
                    f"{metric}: {patched_value} exceeds threshold "
                    f"(baseline: {baseline_value}, limit: {baseline_value * threshold})"
                )
        
        if violations:
            return StageResult(
                passed=False,
                errors=violations,
                metrics={
                    'baseline': baseline,
                    'patched': patched
                },
                duration=self.timer.elapsed()
            )
        
        return StageResult(
            passed=True,
            metrics={
                'baseline': baseline,
                'patched': patched,
                'improvement': self._calculate_improvement(baseline, patched)
            },
            duration=self.timer.elapsed()
        )
```

### Stage 6: Security Validation

```python
class SecurityValidation(ValidationStage):
    def __init__(self):
        self.timeout = 180  # seconds
        self.scanners = {
            'sast': SASTScanner(),
            'dependency': DependencyScanner(),
            'secrets': SecretsScanner(),
            'vulnerability': VulnerabilityScanner()
        }
    
    async def validate(self, patch: Patch) -> StageResult:
        scan_results = await asyncio.gather(*[
            scanner.scan(patch) for scanner in self.scanners.values()
        ])
        
        all_issues = []
        for result in scan_results:
            all_issues.extend(result.issues)
        
        critical_issues = [i for i in all_issues if i.severity == 'critical']
        high_issues = [i for i in all_issues if i.severity == 'high']
        
        if critical_issues:
            return StageResult(
                passed=False,
                errors=critical_issues,
                warnings=high_issues,
                duration=self.timer.elapsed()
            )
        
        return StageResult(
            passed=True,
            warnings=high_issues + [i for i in all_issues if i.severity == 'medium'],
            duration=self.timer.elapsed()
        )
```

### Stage 7: Chaos Validation (Optional)

```python
class ChaosValidation(ValidationStage):
    def __init__(self):
        self.timeout = 600  # seconds
        self.chaos_scenarios = [
            NetworkLatencyScenario(),
            CPUStressScenario(),
            MemoryPressureScenario(),
            DependencyFailureScenario()
        ]
    
    async def validate(self, patch: Patch) -> StageResult:
        # Only run for critical services
        if patch.service_criticality != 'critical':
            return StageResult(passed=True, skipped=True)
        
        failures = []
        for scenario in self.chaos_scenarios:
            result = await scenario.test(patch)
            if not result.survived:
                failures.append(f"Failed {scenario.name}: {result.reason}")
        
        if failures:
            return StageResult(
                passed=False,
                errors=failures,
                duration=self.timer.elapsed()
            )
        
        return StageResult(
            passed=True,
            scenarios_tested=len(self.chaos_scenarios),
            duration=self.timer.elapsed()
        )
```

### Confidence Scoring

```python
class ConfidenceCalculator:
    def calculate(self, validation_results: List[StageResult]) -> float:
        weights = {
            'syntax': 0.1,
            'static_analysis': 0.15,
            'unit_tests': 0.25,
            'integration_tests': 0.2,
            'performance': 0.15,
            'security': 0.1,
            'chaos': 0.05
        }
        
        score = 0.0
        for stage_name, result in validation_results.items():
            if result.passed:
                stage_score = 1.0
                
                # Adjust based on warnings
                if hasattr(result, 'warnings'):
                    stage_score -= len(result.warnings) * 0.05
                
                # Adjust based on coverage
                if hasattr(result, 'test_coverage'):
                    stage_score *= result.test_coverage
                
                score += weights[stage_name] * max(0, stage_score)
        
        return min(1.0, score)
```

### Early Exit Optimization

```python
def _should_run_stage(self, stage: ValidationStage, patch: Patch, 
                     previous_results: List[StageResult]) -> bool:
    # Skip expensive stages for trivial changes
    if patch.complexity == 'trivial' and stage.name in ['chaos', 'performance']:
        return False
    
    # Skip if confidence already high
    current_confidence = self._calculate_confidence(previous_results)
    if current_confidence > 0.95 and stage.name == 'chaos':
        return False
    
    # Skip if patch is emergency
    if patch.priority == 'emergency' and stage.name in ['performance', 'chaos']:
        return False
    
    return True
```

## Links

- [Testing Module Documentation](../../modules/testing/README.md)
- [ADR-002: Parallel Environment Testing](002-parallel-environment-testing.md)
- [ADR-006: Security Approval Workflow](006-security-approval-workflow.md)