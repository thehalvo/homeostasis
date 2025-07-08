# C# Integration

This document describes the C# integration capabilities in Homeostasis, including error detection, analysis, and automated healing for C# applications.

## Overview

The C# integration in Homeostasis provides error handling for C# and .NET applications. It supports:

- ASP.NET Core web applications and APIs
- Entity Framework Core database access
- Asynchronous programming patterns
- .NET dependency injection
- Azure cloud services
- Common C# runtime exceptions

## Integration Options

### 1. ASP.NET Core Middleware

```csharp
public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
{
    // Add Homeostasis exception handling middleware
    app.UseHomeostasisExceptionHandling(options => {
        options.EnableSelfHealing = true;
        options.CollectContextData = true;
        options.ReportToApi = true;
    });
    
    // Other middleware components
    app.UseRouting();
    app.UseAuthorization();
    app.UseEndpoints(endpoints => {
        endpoints.MapControllers();
    });
}
```

### 2. Exception Filter for Controller-based Applications

```csharp
public void ConfigureServices(IServiceCollection services)
{
    services.AddControllers(options => {
        options.Filters.Add<HomeostasisExceptionFilter>();
    });
    
    services.AddHomeostasisErrorTracking(Configuration);
}
```

### 3. Global Exception Handler

```csharp
AppDomain.CurrentDomain.UnhandledException += (sender, args) => {
    var exception = args.ExceptionObject as Exception;
    if (exception != null) {
        HomeostasisClient.TrackException(exception);
    }
};

TaskScheduler.UnobservedTaskException += (sender, args) => {
    HomeostasisClient.TrackException(args.Exception);
    args.SetObserved(); // Prevents application termination
};
```

### 4. Logger Integration

```csharp
public static IHostBuilder CreateHostBuilder(string[] args) =>
    Host.CreateDefaultBuilder(args)
        .ConfigureLogging((context, logging) => {
            logging.ClearProviders();
            logging.AddConsole();
            logging.AddHomeostasisLogger(options => {
                options.ApiKey = context.Configuration["Homeostasis:ApiKey"];
                options.ServiceName = "MyService";
                options.EnableSelfHealing = true;
            });
        })
        .ConfigureWebHostDefaults(webBuilder => {
            webBuilder.UseStartup<Startup>();
        });
```

## Supported Frameworks

The C# integration supports the following frameworks and libraries:

| Framework/Library | Supported Versions | Features |
|-------------------|-------------------|----------|
| ASP.NET Core | 3.1, 5.0, 6.0, 7.0+ | Route errors, model validation, middleware exceptions |
| Entity Framework Core | 3.1, 5.0, 6.0, 7.0+ | Database updates, concurrency conflicts, query errors |
| .NET Core / .NET | 3.1, 5.0, 6.0, 7.0+ | Runtime exceptions, async patterns, memory management |
| Azure SDK | Latest | Storage, service bus, functions, app service errors |
| Microsoft.Extensions | Latest | Configuration, DI, logging, options |

## Configuration Options

The C# integration can be configured via appsettings.json:

```json
{
  "Homeostasis": {
    "ApiKey": "your-api-key",
    "ServiceName": "YourServiceName",
    "Environment": "Production",
    "EnableSelfHealing": true,
    "CollectContextData": true,
    "ExcludedExceptions": [
      "System.OperationCanceledException"
    ],
    "ExceptionFilters": {
      "IncludeNamespaces": [
        "YourCompany.YourProduct"
      ],
      "ExcludeNamespaces": [
        "Microsoft.AspNetCore.StaticFiles"
      ]
    },
    "SelfHealing": {
      "MaxPatchesPerHour": 5,
      "RequireApproval": true,
      "NotificationEmail": "admin@example.com"
    }
  }
}
```

## Error Detection Capabilities

The C# integration can detect and analyze a wide range of errors:

### Core .NET Exceptions
- NullReferenceException
- ArgumentNullException
- InvalidOperationException
- IndexOutOfRangeException
- FormatException
- TimeoutException
- ObjectDisposedException

### ASP.NET Core Errors
- Routing errors
- Model validation failures
- Authentication/authorization issues
- CORS policy violations
- Middleware exceptions
- Missing dependencies

### Entity Framework Errors
- Database update exceptions
- Concurrency conflicts
- Connection issues
- Query translation errors
- Migration errors

### Async Programming Issues
- TaskCanceledException
- Deadlocks
- Unobserved task exceptions
- Context switching problems

## Automatic Error Fixes

The C# integration can automatically generate and apply fixes for common error patterns:

1. **Null Reference Handling**
   - Adding null checks
   - Using null-conditional operators (?.)
   - Implementing null-coalescing operators (??)

2. **Argument Validation**
   - Adding guard clauses
   - Parameter validation
   - Default value handling

3. **Entity Framework Fixes**
   - Transaction management
   - Concurrency conflict resolution
   - Connection resilience

4. **Async/Await Pattern Corrections**
   - Proper task cancellation handling
   - ConfigureAwait usage
   - Deadlock prevention

5. **Exception Handling Improvements**
   - Targeted exception handling
   - Graceful degradation
   - Retry policies

## Custom Rules

You can define custom error detection and fix rules using the rule definition format:

```json
{
  "id": "company_custom_error",
  "pattern": "YourCompany\\.CustomException: ([A-Za-z]+) error occurred",
  "type": "YourCompany.CustomException",
  "description": "Custom company-specific exception",
  "root_cause": "company_specific_error",
  "suggestion": "Handle this specific error by checking the component status",
  "confidence": "high",
  "severity": "medium",
  "category": "custom"
}
```

Place these rules in a file at:
```
/homeostasis/rules/csharp/custom_rules.json
```

## Best Practices

1. **Exception Handling**
   - Use specific exception types
   - Catch exceptions at appropriate boundaries
   - Include contextual information

2. **Logging**
   - Use structured logging
   - Include correlation IDs
   - Log appropriate exception details

3. **Database Access**
   - Use retries for transient errors
   - Implement proper concurrency handling
   - Use transactions for multi-entity updates

4. **Asynchronous Programming**
   - Always await Tasks
   - Use ConfigureAwait(false) when appropriate
   - Propagate cancellation tokens

5. **Dependency Injection**
   - Register services with appropriate lifetimes
   - Avoid service locator pattern
   - Prefer constructor injection

## Examples

### Detecting and Fixing a NullReferenceException

```csharp
// Original code with potential null reference
public decimal CalculateDiscount(Order order)
{
    return order.Customer.LoyaltyPoints * 0.01m;
}

// Fixed code with null checks
public decimal CalculateDiscount(Order order)
{
    if (order == null)
    {
        throw new ArgumentNullException(nameof(order));
    }
    
    // Use null conditional operator to safely access nested properties
    return order.Customer?.LoyaltyPoints * 0.01m ?? 0;
}
```

### Handling Entity Framework Concurrency Conflicts

```csharp
// Concurrency conflict handling
try
{
    await _context.SaveChangesAsync();
    return Ok("Changes saved successfully");
}
catch (DbUpdateConcurrencyException ex)
{
    // Get the entity with concurrency conflict
    var entry = ex.Entries.Single();
    var databaseValues = await entry.GetDatabaseValuesAsync();
    
    if (databaseValues == null)
    {
        // The entity was deleted by another user
        return NotFound("The record was deleted by another user");
    }
    
    // Keep client changes but update concurrency token
    var property = entry.Property("RowVersion");
    entry.OriginalValues[property.Metadata.Name] = databaseValues[property.Metadata.Name];
    
    // Try again
    await _context.SaveChangesAsync();
    return Ok("Changes saved successfully");
}
```

### Proper Async/Await Pattern

```csharp
// Properly handling task cancellation
public async Task<IActionResult> ProcessAsync(CancellationToken cancellationToken)
{
    try
    {
        // Pass the cancellation token to all async operations
        var result = await _service.LongRunningOperationAsync(cancellationToken);
        return Ok(result);
    }
    catch (OperationCanceledException)
    {
        // Handle cancellation gracefully
        _logger.LogInformation("Operation was canceled");
        return StatusCode(499, "Request canceled");
    }
}
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure the Homeostasis package is installed
   - Check that required middleware is properly registered

2. **Configuration Problems**
   - Verify appsettings.json has correct Homeostasis section
   - Check API key is properly set

3. **Integration Failures**
   - Ensure middleware order is correct in Startup.cs
   - Check logs for integration errors

### Diagnostic Steps

1. Enable debug logging for Homeostasis:
   ```json
   {
     "Logging": {
       "LogLevel": {
         "Default": "Information",
         "Homeostasis": "Debug"
       }
     }
   }
   ```

2. Verify connectivity to Homeostasis API:
   ```csharp
   var result = await _homeostasisClient.TestConnectionAsync();
   Console.WriteLine($"Connected: {result.IsConnected}");
   ```

3. Check error tracking with a test exception:
   ```csharp
   try
   {
       throw new Exception("Test exception for Homeostasis");
   }
   catch (Exception ex)
   {
       var tracked = HomeostasisClient.TrackException(ex);
       Console.WriteLine($"Exception tracked: {tracked}");
   }
   ```

## Further Reading

- [ASP.NET Core Documentation](https://docs.microsoft.com/en-us/aspnet/core)
- [Entity Framework Core Documentation](https://docs.microsoft.com/en-us/ef/core)
- [Asynchronous Programming in C#](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/async)
- [Exception Handling Best Practices](https://docs.microsoft.com/en-us/dotnet/standard/exceptions/best-practices-for-exceptions)