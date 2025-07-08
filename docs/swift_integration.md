# Swift Integration Guide

This guide explains how to use Homeostasis for Swift application error analysis and self-healing on iOS, macOS, watchOS, and tvOS platforms.

## Overview

The Swift language plugin provides error detection and healing capabilities for Swift applications, including:

- Force unwrapping nil optionals
- Array index out of bounds errors
- Memory management issues (EXC_BAD_ACCESS, retain cycles)
- Main thread checker violations
- SwiftUI and UIKit framework errors
- Core Data persistence issues
- Swift concurrency and async/await errors
- Swift Package Manager dependency analysis

## Supported Frameworks

### UI Frameworks
- **UIKit**: View controllers, auto layout, interface builder
- **SwiftUI**: State management, view updates, previews
- **AppKit**: macOS native applications

### System Frameworks
- **Foundation**: Core Swift APIs and utilities
- **Core Data**: Persistent data storage
- **Core Animation**: Animation and graphics
- **AVFoundation**: Audio and video processing
- **MapKit**: Location and mapping services

### Apple Platform Specific
- **HealthKit**: Health and fitness data
- **HomeKit**: Smart home automation
- **CloudKit**: iCloud integration
- **GameplayKit**: Game development
- **SceneKit**: 3D graphics
- **Metal**: High-performance graphics
- **Core ML**: Machine learning

## Error Detection Capabilities

### 1. Force Unwrapping Errors

**Problem:**
```swift
let user = getUser()
print(user!.name)  // Fatal error: unexpectedly found nil
```

**Detection:**
- Pattern: `fatal error: unexpectedly found nil while unwrapping an Optional value`
- Confidence: High
- Severity: High

**Auto-Fix:**
```swift
// Option 1: Safe unwrapping with if let
if let user = getUser() {
    print(user.name)
}

// Option 2: Guard let for early exit
guard let user = getUser() else { return }
print(user.name)

// Option 3: Nil coalescing
print((getUser()?.name) ?? "Unknown")
```

### 2. Array Bounds Errors

**Problem:**
```swift
let items = getItems()
let first = items[0]  // Fatal error: Index out of range
```

**Detection:**
- Pattern: `fatal error: Index out of range`
- Confidence: High
- Severity: High

**Auto-Fix:**
```swift
// Safe bounds checking
if items.indices.contains(0) {
    let first = items[0]
}

// Or use safe subscript extension
if let first = items.first {
    // Use first element
}
```

### 3. Main Thread Violations

**Problem:**
```swift
DispatchQueue.global().async {
    self.label.text = "Updated"  // Main Thread Checker violation
}
```

**Detection:**
- Pattern: `Main Thread Checker: UI API called on a background thread`
- Confidence: High
- Severity: High

**Auto-Fix:**
```swift
DispatchQueue.global().async {
    DispatchQueue.main.async {
        self.label.text = "Updated"
    }
}

// Or with modern async/await
Task {
    await MainActor.run {
        self.label.text = "Updated"
    }
}
```

### 4. SwiftUI State Errors

**Problem:**
```swift
struct ContentView: View {
    @State private var count = 0
    
    var body: some View {
        Button("Increment") {
            // Modifying state during view update
            count += 1
        }
    }
}
```

**Detection:**
- Pattern: `Publishing changes from within view updates`
- Confidence: High
- Severity: High

**Auto-Fix:**
```swift
Button("Increment") {
    DispatchQueue.main.async {
        count += 1
    }
}

// Or use proper action handling
Button("Increment", action: incrementCount)

private func incrementCount() {
    count += 1
}
```

### 5. Core Data Threading Issues

**Problem:**
```swift
// Accessing context from wrong thread
context.save()  // Threading violation
```

**Detection:**
- Pattern: `NSManagedObjectContext was accessed from multiple threads`
- Confidence: High
- Severity: Critical

**Auto-Fix:**
```swift
context.perform {
    do {
        try context.save()
    } catch {
        // Handle error
    }
}
```

### 6. Swift Concurrency Errors

**Problem:**
```swift
func fetchData() async throws {
    let data = await networkCall()  // Missing 'await' or 'try'
}
```

**Detection:**
- Pattern: `Expression is 'async' but is not marked with 'await'`
- Confidence: High
- Severity: High

**Auto-Fix:**
```swift
func fetchData() async throws {
    do {
        let data = try await networkCall()
        // Handle data
    } catch {
        // Handle error
    }
}
```

## Swift Package Manager Integration

### Dependency Analysis

Homeostasis can analyze Swift Package Manager projects and detect:

- Missing dependencies
- Version conflicts
- Package.swift syntax errors
- Build configuration issues

**Example Package.swift:**
```swift
// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "MyApp",
    platforms: [.iOS(.v14), .macOS(.v11)],
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.0.0")
    ],
    targets: [
        .target(name: "MyApp", dependencies: ["Alamofire"])
    ]
)
```

### Common SPM Issues

1. **Missing Module Error:**
   ```
   No such module 'Alamofire'
   ```
   - **Fix:** Add package to dependencies and target dependencies

2. **Version Conflicts:**
   ```
   dependency version conflict detected
   ```
   - **Fix:** Update version requirements to compatible ranges

3. **Package.swift Syntax:**
   ```
   Package.swift:5:12: error: expected ']'
   ```
   - **Fix:** Correct syntax errors in package declaration

## Configuration

### Basic Setup

1. **Install Homeostasis** (following main installation guide)

2. **Configure for Swift Projects**
   ```yaml
   # homeostasis.yml
   language: swift
   platforms:
     - ios
     - macos
     - watchos
     - tvos
   
   frameworks:
     - UIKit
     - SwiftUI
     - Foundation
     - CoreData
   
   monitoring:
     crash_reporting: true
     performance_tracking: true
     ui_thread_checking: true
   ```

3. **Integration with Xcode**
   - Add build phase script for automatic error detection
   - Configure crash reporting integration
   - Set up continuous monitoring

### Advanced Configuration

```yaml
swift:
  analysis:
    force_unwrapping_detection: true
    memory_leak_detection: true
    concurrency_checks: true
    main_thread_enforcement: true
  
  frameworks:
    swiftui:
      state_mutation_checks: true
      preview_error_handling: true
    
    uikit:
      auto_layout_validation: true
      outlet_connection_checks: true
    
    core_data:
      threading_validation: true
      save_error_handling: true
  
  package_manager:
    dependency_analysis: true
    version_conflict_detection: true
    build_optimization: true
```

## Best Practices

### 1. Optional Safety
- Avoid force unwrapping (`!`) in production code
- Use nil coalescing (`??`) for default values
- Prefer `guard let` for early exit patterns
- Use optional chaining (`?.`) for safe access

### 2. Memory Management
- Use `[weak self]` in closures to prevent retain cycles
- Implement proper cleanup in `deinit`
- Monitor memory usage in instruments
- Use unowned references only when guaranteed

### 3. Concurrency
- Always update UI on the main thread
- Use `@MainActor` for UI-related classes
- Handle task cancellation properly
- Use structured concurrency patterns

### 4. Error Handling
- Implement error handling with `do-catch`
- Use `Result` type for fallible operations
- Log errors with appropriate detail levels
- Provide user-friendly error messages

### 5. Testing
- Write unit tests for error conditions
- Test memory management scenarios
- Validate UI updates on main thread
- Test concurrency edge cases

## Integration Examples

### iOS App with UIKit

```swift
import UIKit
import Homeostasis

class ViewController: UIViewController {
    @IBOutlet weak var label: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Homeostasis will detect and fix common issues
        loadData()
    }
    
    private func loadData() {
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            // Homeostasis detects main thread violations
            DispatchQueue.main.async {
                self?.updateUI(with: data)
            }
        }.resume()
    }
    
    private func updateUI(with data: Data?) {
        // Homeostasis detects force unwrapping issues
        guard let data = data else { return }
        
        // Safe processing
        label.text = processData(data)
    }
}
```

### SwiftUI App

```swift
import SwiftUI
import Homeostasis

struct ContentView: View {
    @StateObject private var viewModel = ViewModel()
    
    var body: some View {
        NavigationView {
            List(viewModel.items) { item in
                ItemRow(item: item)
            }
            .task {
                // Homeostasis handles async errors
                await viewModel.loadItems()
            }
        }
    }
}

@MainActor
class ViewModel: ObservableObject {
    @Published var items: [Item] = []
    
    func loadItems() async {
        do {
            // Homeostasis detects concurrency issues
            items = try await ItemService.fetchItems()
        } catch {
            // Proper error handling
            handleError(error)
        }
    }
}
```

### Core Data Integration

```swift
import CoreData
import Homeostasis

class DataManager {
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "DataModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                // Homeostasis detects Core Data setup issues
                Homeostasis.reportError(error)
            }
        }
        return container
    }()
    
    func saveContext() {
        let context = persistentContainer.viewContext
        
        // Homeostasis ensures proper threading
        context.perform {
            if context.hasChanges {
                do {
                    try context.save()
                } catch {
                    // Proper error handling
                    Homeostasis.reportError(error)
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**
   - Verify Swift plugin is installed
   - Check plugin registry logs
   - Ensure proper language detection

2. **False Positives**
   - Adjust confidence thresholds
   - Configure ignore patterns
   - Review rule specificity

3. **Performance Impact**
   - Enable only necessary checks
   - Configure sampling rates
   - Use development builds for testing

### Debug Mode

Enable debug logging for detailed analysis:

```yaml
logging:
  level: debug
  swift_plugin: true
  pattern_matching: true
  fix_generation: true
```

## Conclusion

The Swift integration provides error detection and healing for iOS, macOS, watchOS, and tvOS applications. By following the patterns and best practices outlined in this guide, you can create more robust and maintainable Swift applications with automatic error recovery capabilities.

For more information, see the main Homeostasis documentation and the Swift plugin source code.