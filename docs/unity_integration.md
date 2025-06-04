# Unity Integration

Homeostasis provides support for Unity applications, including error detection, analysis, and automated fix generation for Unity mobile games, C# scripting issues, and cross-platform deployment challenges.

## Overview

The Unity plugin handles errors specific to game development and Unity framework, including:

- Unity C# scripting and MonoBehaviour errors
- GameObject and Component lifecycle issues
- Mobile platform build and deployment problems
- Unity UI (uGUI) and Canvas errors
- Coroutine and async operation management
- Performance and optimization issues

## Supported Error Types

### Scripting Errors
- NullReferenceException in Unity context
- Missing Component errors
- Destroyed object access attempts
- Coroutine lifecycle issues

### Mobile Build Errors
- Android SDK/NDK configuration problems
- iOS Xcode and provisioning issues
- Platform-specific compilation errors
- Asset bundling and packaging failures

### Unity UI Issues
- Canvas and UI component errors
- Event system configuration problems
- UI scaling and responsive design issues
- Input handling errors

### Performance Issues
- Memory leaks and garbage collection
- Inefficient Update() method usage
- Texture and asset optimization
- Frame rate and rendering problems

## Configuration

Add Unity support to your `config.yaml`:

```yaml
analysis:
  language_plugins:
    - unity
  
frameworks:
  unity:
    project_path: "Assets/"
    project_settings_path: "ProjectSettings/"
    unity_version: "2022.3.0f1"
    build_target: "Android" # or "iOS"
    development_build: true
```

## Example Error Detection

```csharp
// Error: NullReferenceException - GameObject reference not set
public class PlayerController : MonoBehaviour
{
    public GameObject target; // Not assigned in Inspector
    
    void Start()
    {
        target.SetActive(true); // NullReferenceException
    }
}

// Homeostasis will detect and suggest:
// 1. Add null check before accessing
// 2. Use RequireComponent attribute
// 3. Initialize in Awake() or Start()
```

## Automatic Fixes

Homeostasis can automatically generate fixes for:

1. **Null Reference Prevention**: Add null checks for GameObjects and Components
2. **Component Requirements**: Add RequireComponent attributes
3. **Coroutine Safety**: Ensure proper coroutine lifecycle management
4. **Mobile Build Issues**: Fix platform-specific configuration problems
5. **Performance Optimization**: Suggest efficient Unity patterns

## Common Fix Patterns

### Safe GameObject Access
```csharp
// Generated null safety pattern
public class SafePlayerController : MonoBehaviour
{
    [SerializeField] private GameObject target;
    
    void Start()
    {
        if (target != null)
        {
            target.SetActive(true);
        }
        else
        {
            Debug.LogWarning("Target GameObject not assigned!");
        }
    }
}
```

### Component Requirements
```csharp
// Generated component requirement pattern
[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(Collider))]
public class PhysicsObject : MonoBehaviour
{
    private Rigidbody rb;
    private Collider col;
    
    void Awake()
    {
        // Components guaranteed to exist
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
    }
}
```

### Safe Coroutine Usage
```csharp
// Generated coroutine safety pattern
public class CoroutineManager : MonoBehaviour
{
    private Coroutine currentCoroutine;
    
    void Start()
    {
        if (gameObject.activeInHierarchy && enabled)
        {
            currentCoroutine = StartCoroutine(SafeCoroutine());
        }
    }
    
    IEnumerator SafeCoroutine()
    {
        while (this != null && gameObject.activeInHierarchy)
        {
            // Coroutine work
            yield return new WaitForSeconds(1f);
            
            if (this == null) yield break;
        }
    }
    
    void OnDisable()
    {
        if (currentCoroutine != null)
        {
            StopCoroutine(currentCoroutine);
            currentCoroutine = null;
        }
    }
}
```

### Unity UI Safety
```csharp
// Generated UI safety pattern
using UnityEngine;
using UnityEngine.UI;

public class SafeUIManager : MonoBehaviour
{
    [SerializeField] private Button myButton;
    [SerializeField] private Text myText;
    
    void Start()
    {
        if (myButton != null)
        {
            myButton.onClick.AddListener(OnButtonClick);
        }
        
        UpdateText("Hello World");
    }
    
    void UpdateText(string newText)
    {
        if (myText != null)
        {
            myText.text = newText ?? string.Empty;
        }
    }
    
    void OnDestroy()
    {
        if (myButton != null)
        {
            myButton.onClick.RemoveListener(OnButtonClick);
        }
    }
}
```

## Best Practices

1. **Null Checks**: Always verify GameObject and Component references
2. **Component Management**: Use RequireComponent for essential dependencies
3. **Coroutine Lifecycle**: Properly start and stop coroutines
4. **Mobile Performance**: Optimize for mobile device constraints
5. **Memory Management**: Avoid memory leaks and excessive allocations

## Mobile Platform Support

### Android Build Configuration
```csharp
// Generated Android optimization
public class AndroidOptimizer : MonoBehaviour
{
    void Awake()
    {
        if (Application.platform == RuntimePlatform.Android)
        {
            // Android-specific optimizations
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 0;
            QualitySettings.antiAliasing = 0;
        }
    }
}
```

### iOS Build Configuration
```csharp
// Generated iOS optimization
public class iOSOptimizer : MonoBehaviour
{
    void Awake()
    {
        if (Application.platform == RuntimePlatform.IPhonePlayer)
        {
            // iOS-specific optimizations
            Application.targetFrameRate = 60;
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **NullReferenceException**
   - Assign references in Inspector or via code
   - Add null checks before accessing objects
   - Use FindObjectOfType with null validation

2. **Missing Component errors**
   - Add RequireComponent attribute
   - Attach components manually in Inspector
   - Use TryGetComponent for optional components

3. **Coroutine failures**
   - Ensure GameObject is active when starting coroutines
   - Stop coroutines in OnDisable/OnDestroy
   - Check object validity during coroutine execution

4. **Mobile build failures**
   - Update Unity and platform SDKs
   - Check minimum API levels and deployment targets
   - Verify platform-specific settings and permissions

## Performance Optimization

### Memory Management
```csharp
// Generated memory optimization pattern
public class MemoryOptimizedController : MonoBehaviour
{
    private List<GameObject> pooledObjects = new List<GameObject>();
    
    // Object pooling instead of Instantiate/Destroy
    public GameObject GetPooledObject()
    {
        foreach (GameObject obj in pooledObjects)
        {
            if (!obj.activeInHierarchy)
            {
                return obj;
            }
        }
        
        // Create new if none available
        GameObject newObj = Instantiate(prefab);
        pooledObjects.Add(newObj);
        return newObj;
    }
}
```

### Frame Rate Optimization
```csharp
// Generated frame rate optimization
public class FrameRateManager : MonoBehaviour
{
    [SerializeField] private int targetFrameRate = 60;
    
    void Awake()
    {
        Application.targetFrameRate = targetFrameRate;
        
        // Mobile-specific settings
        if (Application.isMobilePlatform)
        {
            QualitySettings.vSyncCount = 0;
            QualitySettings.antiAliasing = 0;
        }
    }
}
```

## Integration with Unity Cloud Build

```yaml
# Example Unity Cloud Build integration
build_script: |
  # Run Homeostasis analysis before build
  python -m homeostasis.orchestrator --unity --analyze
  
  # Build with Unity
  /Applications/Unity/Unity.app/Contents/MacOS/Unity \
    -batchmode \
    -quit \
    -projectPath . \
    -buildTarget Android \
    -executeMethod BuildScript.BuildAndroid
```

## Unity Editor Integration

Homeostasis can integrate with Unity Editor for real-time error detection:

```csharp
// Example Editor integration
#if UNITY_EDITOR
using UnityEditor;

[InitializeOnLoad]
public class HomeostasisEditorIntegration
{
    static HomeostasisEditorIntegration()
    {
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
    }
    
    private static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.EnteredPlayMode)
        {
            // Run Homeostasis analysis
            RunAnalysis();
        }
    }
}
#endif
```

For more information, see the [Game Development Best Practices](best_practices.md) guide.