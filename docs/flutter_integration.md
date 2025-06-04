# Flutter Integration

Homeostasis provides support for Flutter applications, including error detection, analysis, and automated fix generation for Dart language and Flutter framework issues.

## Overview

The Flutter plugin handles errors specific to Flutter mobile, web, and desktop development, including:

- Widget lifecycle and build errors
- Dart null safety violations
- Layout overflow and constraint issues
- State management problems
- Async/await and Future handling
- Flutter-specific runtime errors

## Supported Error Types

### Widget Errors
- Widget build method failures
- Layout overflow issues (RenderFlex)
- Widget tree structure problems
- Custom widget implementation errors

### Dart Language Errors
- Null safety violations
- Type casting errors
- Async/await handling issues
- Collection and iteration errors

### State Management
- StatefulWidget lifecycle issues
- setState() after dispose errors
- Provider/Riverpod state errors
- BLoC pattern violations

### Performance Issues
- Excessive widget rebuilds
- Memory leaks in listeners
- Inefficient widget structure
- Animation performance problems

## Configuration

Add Flutter support to your `config.yaml`:

```yaml
analysis:
  language_plugins:
    - flutter
  
frameworks:
  flutter:
    pubspec_path: "pubspec.yaml"
    lib_path: "lib/"
    analysis_options_path: "analysis_options.yaml"
    dart_sdk_path: "/usr/local/flutter/bin/cache/dart-sdk"
```

## Example Error Detection

```dart
// Error: RenderFlex overflowed by 123 pixels
Row(
  children: [
    Text('Very long text that will overflow'),
    Icon(Icons.star),
  ],
)

// Homeostasis will detect and suggest:
// 1. Wrap with Expanded widget
// 2. Use Flexible for adaptive sizing
// 3. Add SingleChildScrollView for scrollable content
```

## Automatic Fixes

Homeostasis can automatically generate fixes for:

1. **Null Safety**: Add null checks and safe navigation operators
2. **Widget Overflow**: Wrap widgets with Expanded, Flexible, or scrollable containers
3. **State Lifecycle**: Add mounted checks before setState calls
4. **Async Errors**: Add proper error handling for Future operations
5. **Type Safety**: Add safe type casting and checks

## Common Fix Patterns

### Null Safety Fixes
```dart
// Generated null safety pattern
String? nullableString = getValue();
String result = nullableString ?? 'default';
print(nullableString?.length); // Safe navigation
```

### Widget Overflow Solutions
```dart
// Generated overflow fix
Row(
  children: [
    Expanded(
      child: Text('Long text that might overflow'),
    ),
    Icon(Icons.star),
  ],
)
```

### Safe setState Usage
```dart
// Generated lifecycle safety pattern
void updateState() {
  if (mounted) {
    setState(() {
      // Safe state update
    });
  }
}
```

### Async Error Handling
```dart
// Generated async error handling
Future<void> fetchData() async {
  try {
    final result = await apiCall();
    // Handle success
  } catch (error) {
    // Handle error
    print('Error: $error');
  }
}
```

## Best Practices

1. **Null Safety**: Always use null-aware operators and proper null checks
2. **Widget Structure**: Design responsive layouts that handle different screen sizes
3. **State Management**: Use proper lifecycle methods and dispose resources
4. **Performance**: Minimize unnecessary widget rebuilds and use const constructors
5. **Error Handling**: Implement proper error handling for async operations

## Flutter-Specific Features

### Widget Testing Support
```dart
// Homeostasis can help fix widget tests
testWidgets('Counter increments smoke test', (WidgetTester tester) async {
  await tester.pumpWidget(MyApp());
  expect(find.text('0'), findsOneWidget);
});
```

### Platform-Specific Code
```dart
// Generated platform detection
import 'dart:io';

if (Platform.isIOS) {
  // iOS-specific code
} else if (Platform.isAndroid) {
  // Android-specific code
}
```

## Troubleshooting

### Common Issues

1. **Widget overflow errors**
   - Use Expanded or Flexible widgets
   - Consider SingleChildScrollView for scrollable content
   - Check parent widget constraints

2. **Null safety violations**
   - Add null checks before accessing properties
   - Use null-aware operators (?., ??, !)
   - Initialize variables properly

3. **setState after dispose**
   - Check if widget is mounted before setState
   - Cancel async operations in dispose()
   - Use proper widget lifecycle management

4. **Type casting errors**
   - Use safe casting with 'as?'
   - Check types with 'is' operator
   - Verify generic types match expected types

## Integration with Development Tools

### Flutter Doctor Integration
```bash
# Check Flutter installation
flutter doctor

# Run with Homeostasis analysis
python -m homeostasis.orchestrator --flutter --analyze
```

### Hot Reload Support
Homeostasis integrates with Flutter's hot reload for rapid fix testing:

```bash
# Apply fixes with hot reload
flutter run --hot
# Homeostasis will automatically test fixes with hot reload
```

## Performance Optimization

Homeostasis can detect and suggest fixes for:

1. **Widget Rebuilds**: Unnecessary widget reconstructions
2. **Memory Usage**: Memory leaks and inefficient resource usage
3. **Animation Performance**: Janky animations and transitions
4. **Build Performance**: Slow build method execution

For more information, see the [Flutter Best Practices](best_practices.md) guide.