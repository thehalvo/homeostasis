# Xamarin Integration

Homeostasis provides support for Xamarin applications, including error detection, analysis, and automated fix generation for Xamarin.Forms, Xamarin.iOS, and Xamarin.Android development challenges.

## Overview

The Xamarin plugin handles errors specific to cross-platform mobile development, including:

- Xamarin.Forms UI and data binding issues
- Platform-specific integration problems
- DependencyService and IOC container issues
- Custom renderer and effect errors
- MVVM pattern violations
- Mobile permissions and lifecycle management

## Supported Error Types

### Xamarin.Forms Errors
- XAML binding path errors
- BindingContext configuration issues
- Custom control rendering problems
- Navigation and page lifecycle errors

### Platform Binding Issues
- Native API access problems
- Platform-specific code integration
- Resource loading and asset management
- Hardware feature access errors

### Dependency Service
- Interface registration problems
- Implementation resolution failures
- Platform-specific service issues
- IOC container configuration errors

### MVVM Pattern Issues
- PropertyChanged notification problems
- Command binding errors
- ViewModel lifecycle issues
- Data validation failures

## Configuration

Add Xamarin support to your `config.yaml`:

```yaml
analysis:
  language_plugins:
    - xamarin
  
frameworks:
  xamarin:
    solution_path: "MyApp.sln"
    shared_project_path: "MyApp/"
    ios_project_path: "MyApp.iOS/"
    android_project_path: "MyApp.Android/"
    packages_config_path: "packages.config"
```

## Example Error Detection

```csharp
// Error: DependencyService could not resolve IMyService
var service = DependencyService.Get<IMyService>();
service.DoSomething(); // NullReferenceException

// Homeostasis will detect and suggest:
// 1. Register implementation in platform projects
// 2. Add [assembly: Dependency] attribute
// 3. Check interface accessibility
```

## Automatic Fixes

Homeostasis can automatically generate fixes for:

1. **DependencyService Registration**: Add proper service registration
2. **XAML Binding Fixes**: Correct binding paths and BindingContext setup
3. **Custom Renderer Registration**: Add ExportRenderer attributes
4. **Null Reference Prevention**: Add null checks for platform-specific code
5. **Async Pattern Fixes**: Proper async/await usage in mobile context

## Common Fix Patterns

### DependencyService Registration
```csharp
// Generated service registration
[assembly: Dependency(typeof(MyServiceImplementation))]
namespace MyApp.iOS
{
    public class MyServiceImplementation : IMyService
    {
        public void DoSomething()
        {
            // Platform-specific implementation
        }
    }
}

// Safe usage pattern
var service = DependencyService.Get<IMyService>();
service?.DoSomething();
```

### XAML Binding Fixes
```xml
<!-- Generated binding fix -->
<ContentPage xmlns="http://xamarin.com/schemas/2014/forms"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml">
    
    <ContentPage.BindingContext>
        <local:MyViewModel />
    </ContentPage.BindingContext>
    
    <Label Text="{Binding PropertyName, FallbackValue='Default Text'}" />
</ContentPage>
```

### Custom Renderer Registration
```csharp
// Generated renderer registration
[assembly: ExportRenderer(typeof(MyCustomControl), typeof(MyCustomControlRenderer))]
namespace MyApp.iOS.Renderers
{
    public class MyCustomControlRenderer : ViewRenderer<MyCustomControl, UIView>
    {
        protected override void OnElementChanged(ElementChangedEventArgs<MyCustomControl> e)
        {
            base.OnElementChanged(e);
            
            if (e.NewElement != null)
            {
                var nativeControl = new UIView();
                SetNativeControl(nativeControl);
            }
        }
    }
}
```

### Safe Async Patterns
```csharp
// Generated mobile-safe async pattern
private async void OnButtonClicked(object sender, EventArgs e)
{
    try
    {
        await PerformAsyncOperation().ConfigureAwait(false);
        
        Device.BeginInvokeOnMainThread(() => {
            // UI updates on main thread
        });
    }
    catch (Exception ex)
    {
        await DisplayAlert("Error", ex.Message, "OK");
    }
}
```

## Best Practices

1. **DependencyService**: Always register implementations in platform projects
2. **XAML Binding**: Set BindingContext before binding evaluation
3. **Async Operations**: Use ConfigureAwait(false) and proper thread handling
4. **Custom Renderers**: Follow platform-specific guidelines
5. **Memory Management**: Properly dispose of resources and unsubscribe from events

## Platform-Specific Features

### iOS Integration
```csharp
// Generated iOS-specific code
#if __IOS__
using UIKit;

public void IOSSpecificMethod()
{
    if (UIDevice.CurrentDevice.CheckSystemVersion(13, 0))
    {
        // iOS 13+ specific functionality
    }
}
#endif
```

### Android Integration
```csharp
// Generated Android-specific code
#if __ANDROID__
using Android;
using AndroidX.Core.Content;

public bool CheckPermission()
{
    var context = Platform.CurrentActivity ?? Android.App.Application.Context;
    return ContextCompat.CheckSelfPermission(context, 
        Manifest.Permission.Camera) == Permission.Granted;
}
#endif
```

## Troubleshooting

### Common Issues

1. **DependencyService resolution failures**
   - Verify interface is in shared project
   - Check implementation is registered in platform projects
   - Ensure [assembly: Dependency] attribute is present

2. **XAML binding errors**
   - Verify property names match exactly (case-sensitive)
   - Ensure BindingContext is set before binding evaluation
   - Check if property implements INotifyPropertyChanged

3. **Custom renderer not found**
   - Verify [assembly: ExportRenderer] attribute
   - Check renderer inherits from correct base class
   - Ensure renderer is in platform-specific project

4. **Platform build failures**
   - Update NuGet packages to compatible versions
   - Check minimum OS version requirements
   - Verify platform-specific dependencies

## MVVM Pattern Support

### ViewModel Base Class
```csharp
// Generated MVVM base pattern
public class BaseViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler PropertyChanged;
    
    protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    protected bool SetProperty<T>(ref T backingStore, T value,
        [CallerMemberName] string propertyName = "",
        Action onChanged = null)
    {
        if (EqualityComparer<T>.Default.Equals(backingStore, value))
            return false;

        backingStore = value;
        onChanged?.Invoke();
        OnPropertyChanged(propertyName);
        return true;
    }
}
```

## Integration with Build Systems

### MSBuild Integration
```xml
<!-- Example MSBuild integration -->
<Target Name="HomeostasisAnalysis" BeforeTargets="Build">
  <Exec Command="python -m homeostasis.orchestrator --xamarin --analyze" />
</Target>
```

### CI/CD Pipeline
```yaml
# Example Azure DevOps integration
- task: PythonScript@0
  displayName: 'Run Homeostasis Analysis'
  inputs:
    scriptSource: 'inline'
    script: |
      python -m homeostasis.orchestrator --analyze --xamarin
```

For more information, see the [Mobile Development Best Practices](best_practices.md) guide.