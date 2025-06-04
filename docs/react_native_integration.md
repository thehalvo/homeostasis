# React Native Integration

Homeostasis provides support for React Native applications, including error detection, analysis, and automated fix generation for common mobile development challenges.

## Overview

The React Native plugin handles errors specific to hybrid mobile application development, including:

- Metro bundler configuration and dependency resolution
- Native module linking and integration issues
- Platform-specific build problems (iOS/Android)
- Bridge communication errors
- Mobile permissions and capabilities
- React Native component and lifecycle issues

## Supported Error Types

### Metro Bundler Errors
- Module resolution failures
- Bundle compilation issues
- Cache corruption problems
- Dependency conflicts

### Native Module Issues
- Missing or unlinked native modules
- iOS CocoaPods configuration problems
- Android Gradle build failures
- Platform-specific dependency conflicts

### Bridge Communication
- Native bridge communication failures
- Callback timing issues
- Data serialization problems
- Thread safety violations

### Mobile Platform Issues
- iOS provisioning and certificate problems
- Android SDK/NDK configuration
- Platform permission handling
- App lifecycle management

## Configuration

Add React Native support to your `config.yaml`:

```yaml
analysis:
  language_plugins:
    - react_native
  
frameworks:
  react_native:
    metro_config_path: "metro.config.js"
    package_json_path: "package.json"
    ios_project_path: "ios/"
    android_project_path: "android/"
```

## Example Error Detection

```javascript
// Error: Native module RCTCamera cannot be null
import { Camera } from 'react-native-camera';

// Homeostasis will detect and suggest:
// 1. Install the camera module: npm install react-native-camera
// 2. Link the native module: npx react-native run-ios
// 3. Add proper null checks
```

## Automatic Fixes

Homeostasis can automatically generate fixes for:

1. **Missing Dependencies**: Install and link native modules
2. **Metro Configuration**: Update bundler settings and resolve paths
3. **Permission Handling**: Add proper runtime permission requests
4. **Platform Builds**: Fix iOS/Android build configuration issues
5. **Bridge Safety**: Add null checks and error handling for native calls

## Common Fix Patterns

### Native Module Installation
```bash
# Automatically generated commands
npm install react-native-module-name
npx react-native run-ios
npx react-native run-android
```

### Safe Native Module Usage
```javascript
// Generated fix pattern
import { NativeModules } from 'react-native';

const { MyNativeModule } = NativeModules;

if (MyNativeModule) {
  MyNativeModule.someMethod();
} else {
  console.warn('MyNativeModule not available');
}
```

### Permission Handling
```javascript
// Generated permission request pattern
import { PermissionsAndroid, Platform } from 'react-native';

const requestPermission = async () => {
  if (Platform.OS === 'android') {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.CAMERA
    );
    return granted === PermissionsAndroid.RESULTS.GRANTED;
  }
  return true;
};
```

## Best Practices

1. **Dependency Management**: Keep React Native and native modules updated
2. **Platform Testing**: Test on both iOS and Android devices
3. **Error Boundaries**: Use React error boundaries for component errors
4. **Native Module Safety**: Always check availability before using native modules
5. **Performance Monitoring**: Monitor bridge communication performance

## Troubleshooting

### Common Issues

1. **Metro bundler fails to start**
   - Clear Metro cache: `npx react-native start --reset-cache`
   - Check for conflicting dependencies

2. **Native module not found**
   - Verify installation: `npm list react-native-module`
   - Re-link modules: `npx react-native run-ios`

3. **Build failures**
   - Clean build folders: `cd ios && xcodebuild clean`
   - Update dependencies: `cd ios && pod install`

## Integration with CI/CD

Homeostasis can be integrated into your React Native CI/CD pipeline:

```yaml
# Example GitHub Actions integration
- name: Run Homeostasis Analysis
  run: |
    python -m homeostasis.orchestrator --analyze
    python -m homeostasis.orchestrator --fix --react-native
```

For more information, see the [Integration Guides](integration_guides.md).