# Capacitor/Cordova Integration

Homeostasis provides support for Capacitor and Cordova hybrid applications, including error detection, analysis, and automated fix generation for hybrid mobile app development challenges.

## Overview

The Capacitor/Cordova plugin handles errors specific to hybrid mobile app development, including:

- Plugin installation and integration issues
- Native bridge communication problems
- Mobile permissions and capabilities errors
- WebView configuration and CSP violations
- Platform-specific build and deployment issues
- Hybrid app lifecycle and state management

## Supported Error Types

### Plugin Integration
- Missing or unlinked plugins
- Plugin version compatibility issues
- Platform-specific plugin configuration
- Plugin method availability problems

### Native Bridge
- Bridge communication failures
- Callback timing and lifecycle issues
- Data serialization between web and native
- Thread safety in bridge operations

### WebView Issues
- Content loading and rendering problems
- Content Security Policy violations
- Resource access and CORS issues
- WebView performance optimization

### Platform Build
- iOS Xcode configuration and provisioning
- Android Gradle and SDK issues
- Platform-specific dependency conflicts
- App packaging and signing problems

## Configuration

Add Capacitor/Cordova support to your `config.yaml`:

```yaml
analysis:
  language_plugins:
    - capacitor_cordova
  
frameworks:
  capacitor:
    config_path: "capacitor.config.ts"
    web_dir: "dist"
    ios_project_path: "ios/"
    android_project_path: "android/"
  
  cordova:
    config_path: "config.xml"
    www_dir: "www"
    platforms_dir: "platforms/"
```

## Example Error Detection

```javascript
// Error: Plugin 'Camera' not found
import { Camera } from '@capacitor/camera';

async function takePicture() {
  const image = await Camera.getPhoto({
    quality: 90,
    allowEditing: true,
    resultType: CameraResultType.Uri
  }); // Plugin not installed
}

// Homeostasis will detect and suggest:
// 1. Install plugin: npm install @capacitor/camera
// 2. Sync platforms: npx cap sync
// 3. Add availability check
```

## Automatic Fixes

Homeostasis can automatically generate fixes for:

1. **Plugin Installation**: Install and configure missing plugins
2. **Permission Handling**: Add proper runtime permission requests
3. **Bridge Safety**: Add availability checks and error handling
4. **CSP Configuration**: Update Content Security Policy for hybrid apps
5. **Platform Build Issues**: Fix iOS/Android build configuration

## Common Fix Patterns

### Safe Plugin Usage
```javascript
// Generated plugin safety pattern
import { Capacitor } from '@capacitor/core';

async function safePluginUsage() {
  // Check if plugin is available
  if (Capacitor.isPluginAvailable('Camera')) {
    const { Camera } = await import('@capacitor/camera');
    try {
      const result = await Camera.getPhoto({
        quality: 90,
        allowEditing: true,
        resultType: CameraResultType.Uri
      });
      return result;
    } catch (error) {
      console.error('Camera plugin error:', error);
      // Provide fallback or user notification
    }
  } else {
    console.warn('Camera plugin not available on this platform');
    // Provide web fallback if possible
  }
}
```

### Permission Handling
```javascript
// Generated permission request pattern
import { Capacitor } from '@capacitor/core';

async function requestCameraPermission() {
  if (Capacitor.getPlatform() === 'web') {
    // Web permissions handled differently
    return true;
  }
  
  try {
    const { Permissions } = await import('@capacitor/permissions');
    
    // Check current permission status
    const status = await Permissions.query({ name: 'camera' });
    
    if (status.state === 'granted') {
      return true;
    }
    
    if (status.state === 'denied') {
      // Show explanation to user
      showPermissionExplanation();
      return false;
    }
    
    // Request permission
    const result = await Permissions.request({ name: 'camera' });
    return result.state === 'granted';
    
  } catch (error) {
    console.error('Permission request failed:', error);
    return false;
  }
}
```

### Platform Detection
```javascript
// Generated platform detection pattern
import { Capacitor } from '@capacitor/core';

class PlatformManager {
  static isWeb() {
    return Capacitor.getPlatform() === 'web';
  }
  
  static isIOS() {
    return Capacitor.getPlatform() === 'ios';
  }
  
  static isAndroid() {
    return Capacitor.getPlatform() === 'android';
  }
  
  static async useFeature(featureName, nativeImpl, webImpl) {
    if (this.isWeb() && webImpl) {
      return await webImpl();
    } else if (Capacitor.isPluginAvailable(featureName)) {
      try {
        return await nativeImpl();
      } catch (error) {
        console.error(`Native ${featureName} failed:`, error);
        if (webImpl && this.isWeb()) {
          return await webImpl();
        }
        throw error;
      }
    } else {
      throw new Error(`Feature ${featureName} not available`);
    }
  }
}
```

### CSP Configuration
```html
<!-- Generated CSP fix for hybrid apps -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>App</title>
    
    <!-- Updated CSP for Capacitor apps -->
    <meta http-equiv="Content-Security-Policy" 
          content="default-src 'self' data: https://ssl.gstatic.com 'unsafe-eval' 'unsafe-inline'; 
                   object-src 'none'; 
                   style-src 'self' 'unsafe-inline'; 
                   script-src 'self' 'unsafe-inline' 'unsafe-eval'; 
                   media-src 'self' data: content:; 
                   img-src 'self' data: content: blob:; 
                   connect-src 'self' https: wss:; 
                   frame-src 'self';">
    
    <!-- For Cordova apps -->
    <meta http-equiv="Content-Security-Policy" 
          content="default-src 'self' data: gap: https://ssl.gstatic.com 'unsafe-eval'; 
                   style-src 'self' 'unsafe-inline'; 
                   media-src *; 
                   img-src 'self' data: content:;">
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

## Best Practices

1. **Plugin Management**: Always check plugin availability before usage
2. **Permission Handling**: Request permissions gracefully with explanations
3. **Platform Differences**: Handle platform-specific behavior appropriately
4. **Error Handling**: Provide fallbacks for native feature failures
5. **Performance**: Optimize WebView content and native bridge usage

## Capacitor-Specific Features

### Capacitor Configuration
```typescript
// Generated capacitor.config.ts
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.example.app',
  appName: 'MyApp',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 3000,
      launchAutoHide: true,
      backgroundColor: "#ffffffff",
      androidSplashResourceName: "splash",
      showSpinner: true,
      spinnerColor: "#999999"
    },
    Keyboard: {
      resize: "body",
      style: "dark",
      resizeOnFullScreen: true
    }
  },
  ios: {
    scheme: 'MyApp'
  },
  android: {
    allowMixedContent: true
  }
};

export default config;
```

## Cordova-Specific Features

### Device Ready Handling
```javascript
// Generated deviceready handling
document.addEventListener('deviceready', function() {
  console.log('Cordova device ready');
  
  // Safe to use Cordova plugins now
  if (window.cordova && window.cordova.plugins.SomePlugin) {
    window.cordova.plugins.SomePlugin.someMethod(
      function(success) {
        console.log('Plugin success:', success);
      },
      function(error) {
        console.error('Plugin error:', error);
      }
    );
  }
}, false);
```

## Troubleshooting

### Common Issues

1. **Plugin not found errors**
   - Check plugin installation: `npm list @capacitor/plugin-name`
   - Sync platforms: `npx cap sync`
   - Verify plugin compatibility with Capacitor version

2. **Permission denied errors**
   - Add permissions to native manifests
   - Request permissions at runtime
   - Provide user-friendly permission explanations

3. **Build failures**
   - Update platform SDKs and build tools
   - Check for conflicting dependencies
   - Verify native project configuration

4. **WebView issues**
   - Update Content Security Policy
   - Check resource loading paths
   - Test on different WebView versions

## Platform Build Configuration

### iOS Build Settings
```json
{
  "ios": {
    "scheme": "MyApp",
    "path": "ios",
    "cordovaSwiftVersion": "5"
  },
  "plugins": {
    "SplashScreen": {
      "launchShowDuration": 3000,
      "launchAutoHide": true
    }
  }
}
```

### Android Build Settings
```json
{
  "android": {
    "allowMixedContent": true,
    "captureInput": true,
    "webContentsDebuggingEnabled": true
  }
}
```

## Integration with Build Systems

### CI/CD Pipeline
```yaml
# Example GitHub Actions integration
- name: Install dependencies
  run: npm ci

- name: Build web assets
  run: npm run build

- name: Run Homeostasis Analysis
  run: python -m homeostasis.orchestrator --capacitor --analyze

- name: Sync Capacitor
  run: npx cap sync

- name: Build iOS
  run: npx cap build ios
```

### Automated Testing
```javascript
// Example E2E test with plugin validation
describe('Plugin Integration', () => {
  it('should handle camera plugin safely', async () => {
    await device.reloadReactNative();
    
    // Test plugin availability
    const cameraAvailable = await element(by.id('camera-available')).getAttributes();
    expect(cameraAvailable.text).toBe('true');
    
    // Test permission handling
    await element(by.id('request-camera')).tap();
    await waitFor(element(by.text('Camera permission granted')))
      .toBeVisible()
      .withTimeout(5000);
  });
});
```

For more information, see the [Hybrid App Development Guide](best_practices.md).