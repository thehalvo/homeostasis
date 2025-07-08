# Kotlin Integration Guide

This guide covers the Kotlin language support in Homeostasis, including Android development, coroutines, Jetpack Compose, Room database, and Kotlin Multiplatform projects.

## Overview

The Kotlin plugin for Homeostasis provides intelligent error detection, analysis, and automated healing for:

- **Core Kotlin**: Null safety, type casting, collections, and basic language features
- **Android Development**: Activities, Fragments, Services, lifecycle management, and UI components
- **Kotlin Coroutines**: Async/await patterns, cancellation handling, structured concurrency
- **Jetpack Compose**: UI state management, recomposition issues, performance optimization
- **Room Database**: SQL operations, threading, migrations, and data persistence
- **Kotlin Multiplatform**: Cross-platform code sharing, expect/actual declarations, platform-specific implementations

## Quick Start

### Installation

The Kotlin plugin is automatically registered when Homeostasis detects Kotlin code. No additional installation is required.

### Configuration

Add Kotlin support to your project by ensuring your error reporting includes:

```json
{
  "language": "kotlin",
  "error_type": "KotlinNullPointerException",
  "message": "Attempt to invoke virtual method on a null object reference",
  "stack_trace": [
    "at com.example.MainActivity.onCreate(MainActivity.kt:42)",
    "at android.app.Activity.performCreate(Activity.java:7136)"
  ],
  "android": {
    "api_level": 33,
    "device": "Pixel 6"
  }
}
```

## Supported Error Categories

### 1. Core Kotlin Errors

#### Null Safety Issues
- **KotlinNullPointerException**: Null reference access
- **UninitializedPropertyAccessException**: Lateinit property accessed before initialization
- **TypeCastException**: Unsafe type casting

**Example Detection:**
```kotlin
// Problematic code
val user: User? = getUser()
val name = user.name  // KotlinNullPointerException

// Homeostasis suggests:
val name = user?.name ?: "Unknown"
```

#### Collection Errors
- **IndexOutOfBoundsException**: Invalid array/list access
- **NoSuchElementException**: Empty collection access
- **ConcurrentModificationException**: Collection modified during iteration

### 2. Android Lifecycle Errors

#### Fragment Management
- **IllegalStateException**: Fragment not attached to activity
- **ActivityNotFoundException**: Intent target not found
- **WindowManager.BadTokenException**: Invalid window token

**Example Auto-Fix:**
```kotlin
// Before
class UserFragment : Fragment() {
    override fun onResume() {
        super.onResume()
        // This can crash if fragment is detached
        val context = requireContext()
        showToast(context, "Welcome")
    }
}

// After Homeostasis healing
class UserFragment : Fragment() {
    override fun onResume() {
        super.onResume()
        // Safe context access
        if (isAdded) {
            context?.let { ctx ->
                showToast(ctx, "Welcome")
            }
        }
    }
}
```

#### Memory Management
- **Memory Leaks**: Activity context held by long-lived objects
- **Resource Leaks**: Unclosed cursors, unregistered receivers

### 3. Coroutines and Concurrency

#### Cancellation Handling
- **CancellationException**: Proper coroutine cancellation
- **TimeoutCancellationException**: Operation timeout
- **JobCancellationException**: Parent job cancelled

**Example Healing:**
```kotlin
// Problematic code
suspend fun fetchData() {
    val data = networkCall()  // May throw CancellationException
    processData(data)
}

// Homeostasis suggests:
suspend fun fetchData() {
    try {
        val data = networkCall()
        processData(data)
    } catch (e: CancellationException) {
        // Cleanup if needed
        cleanup()
        throw e  // Re-throw to maintain structured concurrency
    }
}
```

#### Threading Issues
- **Main Thread Violations**: Network/database operations on UI thread
- **Context Switching**: Improper dispatcher usage
- **Shared State**: Concurrent access to mutable state

### 4. Jetpack Compose UI

#### State Management
- **State Not Remembered**: Recomposition state loss
- **Infinite Recomposition**: Performance loops
- **Side Effects**: Improper effect usage

**Example Fix:**
```kotlin
// Problematic
@Composable
fun UserProfile() {
    val state = mutableStateOf("")  // Recreated on each recomposition
    
    TextField(
        value = state.value,
        onValueChange = { state.value = it }
    )
}

// Fixed by Homeostasis
@Composable
fun UserProfile() {
    val state = remember { mutableStateOf("") }  // Properly remembered
    
    TextField(
        value = state.value,
        onValueChange = { state.value = it }
    )
}
```

#### Performance Issues
- **Unnecessary Recompositions**: Inefficient state updates
- **Heavy Computations**: Blocking UI thread
- **Memory Leaks**: Unreleased Compose resources

### 5. Room Database

#### Threading Violations
- **Main Thread Database Access**: SQL operations blocking UI
- **Missing Transactions**: Data consistency issues
- **Migration Errors**: Schema version conflicts

**Example Resolution:**
```kotlin
// Problematic
class UserRepository(private val dao: UserDao) {
    fun getUsers(): List<User> {
        return dao.getAllUsers()  // Main thread violation
    }
}

// Homeostasis healing
class UserRepository(private val dao: UserDao) {
    suspend fun getUsers(): List<User> {
        return withContext(Dispatchers.IO) {
            dao.getAllUsers()
        }
    }
    
    // Alternative: LiveData approach
    fun getUsersLiveData(): LiveData<List<User>> {
        return dao.getAllUsersLiveData()
    }
}
```

#### Data Integrity
- **Foreign Key Violations**: Referential integrity
- **Unique Constraints**: Duplicate key insertion
- **Type Converter Issues**: Custom data type handling

### 6. Kotlin Multiplatform

#### Platform Abstractions
- **Expect/Actual Mismatches**: Missing platform implementations
- **Platform Dependencies**: Unavailable libraries on target platforms
- **Native Interop**: C/Objective-C integration issues

**Example Fix:**
```kotlin
// Common source set
expect class PlatformLogger {
    fun log(message: String)
}

// Android actual implementation
actual class PlatformLogger {
    actual fun log(message: String) {
        android.util.Log.d("App", message)
    }
}

// iOS actual implementation  
actual class PlatformLogger {
    actual fun log(message: String) {
        println(message)  // Basic implementation
    }
}
```

## Configuration Options

### Framework Detection

Homeostasis automatically detects Kotlin frameworks based on:

```kotlin
// Coroutines detection
import kotlinx.coroutines.*

// Android detection  
import androidx.activity.ComponentActivity
import androidx.fragment.app.Fragment

// Compose detection
import androidx.compose.runtime.*
import androidx.compose.ui.*

// Room detection
import androidx.room.*

// Multiplatform detection
expect class Platform
actual class Platform
```

### Custom Rules

Add project-specific rules in `homeostasis.yaml`:

```yaml
kotlin:
  enabled: true
  frameworks: ["android", "coroutines", "compose", "room"]
  
  custom_rules:
    - name: "company_specific_null_check"
      pattern: ".*CompanyUser.*null.*"
      suggestion: "Use CompanyUser.safe() method instead"
      
  ignored_patterns:
    - ".*test.*"  # Ignore test files
    - ".*generated.*"  # Ignore generated code
    
  severity_overrides:
    KotlinNullPointerException: "critical"
    CancellationException: "warning"
```

### Android-Specific Settings

```yaml
kotlin:
  android:
    min_api_level: 21
    target_api_level: 34
    
    lifecycle_tracking: true
    memory_leak_detection: true
    
    ignored_activities:
      - "TestActivity"
      - "MockActivity"
      
    custom_permissions:
      - "com.company.CUSTOM_PERMISSION"
```

## Common Patterns and Solutions

### 1. Null Safety Best Practices

```kotlin
// Instead of unsafe access
user!!.name

// Use safe calls
user?.name

// With default values
user?.name ?: "Unknown"

// Safe let blocks
user?.let { u ->
    processUser(u)
}
```

### 2. Coroutine Error Handling

```kotlin
// Structured exception handling
class DataRepository {
    private val coroutineExceptionHandler = CoroutineExceptionHandler { _, exception ->
        logger.error("Uncaught coroutine exception", exception)
    }
    
    suspend fun fetchData(): Result<Data> = withContext(Dispatchers.IO + coroutineExceptionHandler) {
        try {
            val data = api.getData()
            Result.success(data)
        } catch (e: CancellationException) {
            throw e  // Re-throw cancellation
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

### 3. Android Lifecycle Safety

```kotlin
class MainActivity : AppCompatActivity() {
    private var job: Job? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        job = lifecycleScope.launch {
            // Automatically cancelled when activity destroyed
            collectData()
        }
    }
    
    override fun onDestroy() {
        job?.cancel()  // Explicit cancellation
        super.onDestroy()
    }
}
```

### 4. Compose State Management

```kotlin
@Composable
fun DataScreen(
    viewModel: DataViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    LaunchedEffect(Unit) {
        viewModel.loadData()
    }
    
    when (uiState) {
        is Loading -> LoadingIndicator()
        is Success -> DataContent(uiState.data)
        is Error -> ErrorMessage(uiState.message)
    }
}
```

### 5. Room Database Setup

```kotlin
@Database(
    entities = [User::class],
    version = 2,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    
    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE User ADD COLUMN email TEXT")
            }
        }
    }
}
```

## Debugging and Monitoring

### Error Analytics

Homeostasis tracks Kotlin-specific metrics:

- **Null safety violations**: Frequency and location
- **Coroutine cancellations**: Success rate and timing
- **Android lifecycle issues**: Activity/Fragment errors
- **Compose recompositions**: Performance impact
- **Room query performance**: Database operation timing

### Integration with Android Studio

Install the Homeostasis plugin for Android Studio to get:

- **Real-time error highlighting**
- **Inline fix suggestions**
- **Performance recommendations**
- **Compose preview error handling**

### CI/CD Integration

Add Kotlin analysis to your build pipeline:

```bash
# Gradle task for error analysis
./gradlew homeostasisAnalyze

# Generate report
./gradlew homeostasisReport --output-format json
```

## Advanced Features

### Custom Error Adapters

Create domain-specific error handling:

```kotlin
class CompanyKotlinAdapter : KotlinErrorAdapter() {
    override fun analyzeCustomError(error: CustomError): AnalysisResult {
        return when (error.type) {
            "COMPANY_SPECIFIC" -> AnalysisResult(
                severity = HIGH,
                suggestion = "Use CompanyUtils.handleError()",
                autoFixAvailable = true
            )
            else -> super.analyzeCustomError(error)
        }
    }
}
```

### Multi-Module Support

Configure for multi-module Android projects:

```yaml
kotlin:
  modules:
    app:
      frameworks: ["android", "compose"]
    data:
      frameworks: ["room", "coroutines"]
    domain:
      frameworks: ["coroutines"]
    network:
      frameworks: ["coroutines", "ktor"]
```

### Performance Optimization

Homeostasis provides Kotlin-specific optimizations:

- **Inline function suggestions**
- **Collection operation improvements**
- **Coroutine dispatcher recommendations**
- **Compose performance enhancements**

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**
   - Verify Kotlin files are detected (`.kt` extension)
   - Check language configuration in project settings

2. **Android Detection Failing**
   - Ensure `android` block is present in error data
   - Verify AndroidManifest.xml is accessible

3. **Coroutine Errors Not Caught**
   - Check for `kotlinx.coroutines` imports
   - Verify coroutine context is included in error reporting

4. **Compose Issues Not Detected**
   - Ensure `@Composable` annotations are present
   - Check for Compose imports in stack traces

### Getting Help

- **Documentation**: `/docs/kotlin_integration.md`
- **Examples**: `/examples/kotlin/`
- **Issues**: Report bugs at project repository
- **Community**: Join the Homeostasis Kotlin discussion

## Examples

### Complete Android Application

See `/examples/kotlin/android-sample/` for a full Android application with:
- MVVM architecture
- Jetpack Compose UI
- Room database
- Coroutines for async operations
- Comprehensive error handling

### Multiplatform Project

See `/examples/kotlin/multiplatform-sample/` for:
- Shared business logic
- Platform-specific implementations
- Common error handling patterns
- Cross-platform testing strategies

### Server-Side Kotlin

See `/examples/kotlin/ktor-server/` for:
- Ktor web server
- Coroutine-based request handling
- Database integration
- Error monitoring and reporting

---

For more information, visit the [Homeostasis Documentation](../README.md) or check out the [API Reference](../api/kotlin.md).