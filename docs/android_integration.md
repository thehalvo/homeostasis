# Java Android Integration

This document describes the Java Android integration for the Homeostasis self-healing framework, covering activity lifecycle monitoring, fragment transaction error handling, API compatibility layers, Java-Kotlin interoperability, and service/background task healing.

## Overview

The Java Android integration extends the base Java language plugin to provide comprehensive error detection and healing capabilities for Android applications. It supports both traditional Java Android development and modern Java-Kotlin interoperability scenarios.

## Features

### 1. Activity Lifecycle Monitoring

Automatically detects and fixes activity lifecycle-related errors:

- **Activity state violations**: Operations attempted on destroyed or finishing activities
- **Context null reference errors**: Improper context usage and lifecycle management
- **Memory leaks**: Activity context held after destruction

#### Example Error Detection

```java
// Problematic code that triggers detection
if (someCondition) {
    // Activity might be destroyed here
    startActivity(intent); // May throw IllegalStateException
}
```

#### Automatic Fix Generation

```java
// Generated fix with lifecycle checks
if (!isFinishing() && !isDestroyed()) {
    startActivity(intent);
} else {
    Log.w("MainActivity", "Activity is finishing or destroyed, skipping operation");
}
```

### 2. Fragment Transaction Error Handling

Handles common fragment transaction issues:

- **Fragment not attached errors**: Operations on detached fragments
- **Transaction timing issues**: Commits after `onSaveInstanceState`
- **Fragment lifecycle violations**: Invalid fragment state operations

#### Example Scenarios

```java
// Fragment not attached detection
if (isAdded() && getActivity() != null) {
    // Safe to access activity/context
    performFragmentOperation();
}

// Transaction timing fix
if (!isStateSaved()) {
    transaction.commit();
} else {
    transaction.commitAllowingStateLoss();
}
```

### 3. Legacy Android API Compatibility Layers

Provides backwards compatibility handling for different Android API levels:

- **API level checking**: Automatic version-specific code generation
- **Deprecated method migration**: Suggestions for modern alternatives
- **AndroidX compatibility**: Migration guidance for support libraries

#### Example API Compatibility Fix

```java
// Generated API level compatibility
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
    // Use newer API method
    setTextAppearance(R.style.TextAppearance);
} else {
    // Use legacy method for older versions
    setTextAppearance(this, R.style.TextAppearance);
}
```

### 4. Java-Kotlin Interoperability Support

Handles common issues when integrating Java and Kotlin code:

- **Null safety violations**: Platform types and nullable handling
- **Companion object access**: Proper static member access patterns
- **Data class instantiation**: Constructor and property access
- **Suspend function calls**: Coroutine integration patterns

#### Example Interop Fixes

```java
// Null safety handling
if (kotlinObject != null) {
    kotlinObject.performOperation();
} else {
    Log.w("MainActivity", "Kotlin object is null");
}

// Companion object access
String result = KotlinClass.Companion.getConstant();
// Or with @JvmStatic: KotlinClass.getConstant();
```

### 5. Android Service and Background Task Healing

Addresses modern Android background execution limitations:

- **Background service limitations**: API 26+ background execution limits
- **Service binding failures**: Connection and lifecycle management
- **Notification channel requirements**: API 26+ notification compliance
- **WorkManager migration**: Modern background task scheduling

#### Example Service Fixes

```java
// Background execution compliance (API 26+)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
    // Use foreground service
    startForegroundService(serviceIntent);
} else {
    // Use regular service for older APIs
    startService(serviceIntent);
}

// WorkManager for background tasks
OneTimeWorkRequest workRequest = new OneTimeWorkRequest.Builder(BackgroundWorker.class)
    .setConstraints(constraints)
    .build();
WorkManager.getInstance(this).enqueue(workRequest);
```

## Error Detection Rules

The system includes 25+ Android-specific error detection rules covering:

### Core Android Errors

- `ActivityNotFoundException`: Missing intent handlers
- `NetworkOnMainThreadException`: Network operations on UI thread
- `CalledFromWrongThreadException`: UI operations from background threads
- `BadTokenException`: Invalid window tokens for dialogs
- `SecurityException`: Permission-related errors

### Fragment-Specific Errors

- Fragment not attached to activity
- Fragment transaction timing issues
- Fragment lifecycle violations

### Service and Background Task Errors

- Service binding failures
- Background execution limit violations
- Notification channel requirements
- WorkManager constraint issues

### Java-Kotlin Interoperability Errors

- Null safety violations
- Companion object access issues
- Data class instantiation problems
- Suspend function call attempts

## Configuration

### Language Configuration

The Java language configuration includes Android-specific detection patterns:

```json
{
  "detection_patterns": [
    "(?:android\\.\\w+\\.\\w+Exception:)",
    "(?:ActivityNotFoundException)",
    "(?:NetworkOnMainThreadException)",
    "(?:Fragment.*not attached)",
    "(?:KotlinNullPointerException)"
  ],
  "platform_detection": {
    "android": ["android", "androidx", "Activity", "Fragment"],
    "kotlin_interop": ["kotlin", "Companion"]
  }
}
```

### Android-Specific Rules

Rules are organized by category:

- **Lifecycle**: Activity and fragment lifecycle management
- **Threading**: Main thread and background execution
- **Permissions**: Runtime permission handling
- **Services**: Service lifecycle and background tasks
- **Interoperability**: Java-Kotlin integration

## Usage Examples

### Basic Error Detection

```python
from modules.analysis.plugins.java_plugin import JavaLanguagePlugin

plugin = JavaLanguagePlugin()

error_data = {
    "error_type": "android.content.ActivityNotFoundException",
    "message": "No Activity found to handle Intent",
    "stack_trace": ["at android.app.Activity.startActivity(...)"],
    "language": "java",
    "framework": "android"
}

analysis = plugin.analyze_error(error_data)
patch = plugin.generate_fix(analysis, {"framework": "android"})
```

### Integration with Monitoring

```python
# Monitor Android application logs
from modules.monitoring.logger import Logger

logger = Logger()
logger.configure_android_monitoring({
    "activity_lifecycle": True,
    "fragment_transactions": True,
    "permission_checks": True,
    "background_tasks": True
})
```

## Testing

Comprehensive test coverage includes:

- Activity lifecycle scenarios
- Fragment transaction edge cases
- API compatibility across versions
- Java-Kotlin interoperability patterns
- Service and background task scenarios

### Running Tests

```bash
# Run Java Android plugin tests
python -m pytest tests/test_java_android_plugin.py -v

# Run integration tests
python -m pytest tests/test_java_android_integration.py -v
```

## Best Practices

### 1. Activity Lifecycle Management

- Always check `isFinishing()` and `isDestroyed()` before operations
- Use `WeakReference` for activity contexts in long-lived objects
- Prefer application context for non-UI operations

### 2. Fragment Management

- Check `isAdded()` before accessing activity/context
- Use `commitAllowingStateLoss()` only when state loss is acceptable
- Implement proper fragment lifecycle observers

### 3. Threading Best Practices

- Perform network operations in background threads
- Use `runOnUiThread()` or `Handler.post()` for UI updates
- Consider modern alternatives to `AsyncTask`

### 4. Java-Kotlin Interoperability

- Use `@Nullable`/`@NonNull` annotations in Java
- Handle platform types carefully in Kotlin
- Use `@JvmStatic` for better Java access to Kotlin code

### 5. Background Task Management

- Use foreground services for long-running tasks
- Migrate to WorkManager for reliable background processing
- Implement proper notification channels for API 26+

## Troubleshooting

### Common Issues

1. **Rules not loading**: Check file permissions and JSON syntax
2. **Detection not working**: Verify error format matches patterns
3. **Patch generation failing**: Ensure template variables are available
4. **Integration issues**: Check language configuration and plugin registration

### Debug Mode

Enable debug logging for detailed analysis:

```python
import logging
logging.getLogger('modules.analysis.plugins.java_plugin').setLevel(logging.DEBUG)
```

## Contributing

When adding new Android error detection rules:

1. Add rule to `modules/analysis/rules/java/android_errors.json`
2. Create corresponding patch template in `modules/analysis/patch_generation/templates/java/`
3. Add test cases to `tests/test_java_android_plugin.py`
4. Update documentation with examples

### Rule Format

```json
{
  "id": "java_android_error_name",
  "pattern": "regex_pattern_for_detection",
  "error_type": "ExceptionType",
  "description": "Clear description of the issue",
  "root_cause": "categorized_root_cause",
  "suggestion": "Actionable fix suggestion",
  "confidence": "high|medium|low",
  "severity": "critical|high|medium|low",
  "category": "android",
  "framework": "android"
}
```

## See Also

- [Java Integration Guide](java_integration.md)
- [Kotlin Integration Guide](kotlin_integration.md)
- [Mobile Framework Support](mobile_frameworks.md)
- [Architecture Overview](architecture.md)