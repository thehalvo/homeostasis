# R Integration

Homeostasis provides robust support for R programming language, focusing on data science workflows, statistical analysis, and research computing environments.

## Overview

The R integration handles common patterns in data science including:
- Object and variable access in R environments
- Package management and dependency resolution
- Data frame and vector operations
- Statistical modeling and analysis
- Visualization and plotting workflows

## Supported Error Types

### Runtime Errors
- Object not found errors
- Function not found issues
- Subscript out of bounds
- Missing value handling
- Argument validation failures

### Data Manipulation
- Data frame column access
- Vector length mismatches
- Missing value handling
- Type conversion errors
- Dimension mismatches

### Package Management
- Package installation failures
- Library loading issues
- Namespace conflicts
- Version compatibility problems
- Dependency resolution

### Statistical Modeling
- Model fitting failures
- Singular matrix errors
- Degrees of freedom issues
- Formula specification problems
- Contrasts and factor handling

### Visualization
- Plotting device errors
- Graphics parameter issues
- Figure margin problems
- Color specification errors
- Layout and positioning

## Common Fix Patterns

### Object Access
```r
# Before: Unsafe object access
result <- my_object

# After: Safe object access
if (exists("my_object")) {
  result <- my_object
} else {
  warning("Object my_object not found")
  result <- NULL
}
```

### Package Loading
```r
# Before: Unsafe package loading
library(package_name)

# After: Safe package loading
if (!require(package_name)) {
  install.packages("package_name")
  library(package_name)
}
```

### Data Frame Operations
```r
# Before: Unsafe column access
values <- df$column_name

# After: Safe column access
if ("column_name" %in% names(df)) {
  values <- df$column_name
} else {
  warning("Column 'column_name' not found")
  values <- NULL
}
```

### Vector Operations
```r
# Before: Unsafe vector operations
combined <- c(vector1, vector2)

# After: Safe vector operations
if (length(vector1) == length(vector2)) {
  combined <- c(vector1, vector2)
} else {
  warning("Vector lengths differ")
  combined <- NULL
}
```

## Supported Frameworks

### Data Science
- **Tidyverse**: dplyr, ggplot2, tidyr, readr
- **Data.table**: High-performance data manipulation
- **Shiny**: Web application development
- **R Markdown**: Reproducible research documents

### Statistical Analysis
- **Caret**: Machine learning workflows
- **MLR3**: Modern machine learning framework
- **Bioconductor**: Bioinformatics analysis
- **Forecast**: Time series forecasting

### Visualization
- **ggplot2**: Grammar of graphics
- **Plotly**: Interactive visualizations
- **Leaflet**: Interactive maps
- **DT**: Data tables

## Configuration

### Plugin Settings
```yaml
# config.yaml
r_plugin:
  enabled: true
  frameworks:
    - tidyverse
    - shiny
    - caret
    - bioconductor
  error_detection:
    object_checking: true
    package_validation: true
    data_validation: true
  performance:
    memory_monitoring: true
    execution_timing: true
```

### Rule Configuration
```json
{
  "r_rules": {
    "object_not_found": {
      "enabled": true,
      "severity": "high",
      "auto_fix": true
    },
    "package_errors": {
      "enabled": true,
      "severity": "medium",
      "auto_fix": true
    },
    "data_operations": {
      "enabled": true,
      "severity": "medium",
      "auto_fix": false
    }
  }
}
```

## Best Practices

### Error Handling
1. **Check object existence** before access
2. **Validate function arguments** and types
3. **Handle missing values** appropriately
4. **Use safe package loading** patterns
5. **Check data frame structure** before operations

### Performance Considerations
1. **Use vectorized operations** instead of loops
2. **Avoid unnecessary data copying**
3. **Use appropriate data types**
4. **Monitor memory usage** with large datasets
5. **Optimize data frame operations**

### Data Management
1. **Validate data integrity** before analysis
2. **Handle missing values** consistently
3. **Use appropriate data structures**
4. **Document data transformations**
5. **Implement data validation checks**

## Example Fixes

### Object Not Found Error
```r
# Error: object 'my_data' not found
summary(my_data)

# Fix: Check existence before use
if (exists("my_data")) {
  summary(my_data)
} else {
  stop("Data object 'my_data' not found. Please load the data first.")
}
```

### Package Loading Error
```r
# Error: there is no package called 'ggplot2'
library(ggplot2)

# Fix: Install and load package
if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}
```

### Data Frame Column Error
```r
# Error: undefined columns selected
subset_data <- df[, c("col1", "col2")]

# Fix: Check column existence
required_cols <- c("col1", "col2")
available_cols <- intersect(required_cols, names(df))

if (length(available_cols) == length(required_cols)) {
  subset_data <- df[, available_cols]
} else {
  missing_cols <- setdiff(required_cols, available_cols)
  stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
}
```

### Vector Bounds Error
```r
# Error: subscript out of bounds
value <- my_vector[10]

# Fix: Check vector length
if (length(my_vector) >= 10) {
  value <- my_vector[10]
} else {
  warning("Vector index out of bounds")
  value <- NA
}
```

## Testing Integration

### Unit Testing
```r
# test_functions.R
library(testthat)

test_that("safe_access works correctly", {
  # Test with existing object
  test_obj <- 42
  expect_equal(safe_access("test_obj"), 42)
  
  # Test with non-existing object
  expect_null(safe_access("non_existent"))
})

test_that("safe_column_access works", {
  df <- data.frame(a = 1:3, b = 4:6)
  
  # Test existing column
  expect_equal(safe_column_access(df, "a"), 1:3)
  
  # Test non-existing column
  expect_null(safe_column_access(df, "c"))
})
```

### Integration Testing
```r
# test_integration.R
test_that("full workflow works", {
  # Load required packages
  if (!require(dplyr)) {
    install.packages("dplyr")
    library(dplyr)
  }
  
  # Create test data
  test_data <- data.frame(
    x = 1:10,
    y = 11:20
  )
  
  # Test data operations
  result <- test_data %>%
    filter(x > 5) %>%
    summarise(mean_y = mean(y))
  
  expect_true(nrow(result) == 1)
  expect_true(result$mean_y > 15)
})
```

## Troubleshooting

### Common Issues

1. **Object not found errors**
   - Check variable names for typos
   - Verify object is in current environment
   - Use `ls()` to list available objects
   - Check for proper data loading

2. **Package loading problems**
   - Verify package installation
   - Check package name spelling
   - Update package repositories
   - Resolve dependency conflicts

3. **Data frame operations**
   - Validate column names
   - Check data frame structure
   - Handle missing values appropriately
   - Verify data types

4. **Statistical modeling issues**
   - Check model formula syntax
   - Verify data completeness
   - Handle multicollinearity
   - Validate model assumptions

### Performance Issues

1. **Memory problems**
   - Monitor memory usage with `memory()`
   - Use data.table for large datasets
   - Implement chunked processing
   - Clear unused objects

2. **Slow execution**
   - Use vectorized operations
   - Avoid unnecessary loops
   - Optimize data access patterns
   - Consider parallel processing

## Related Documentation

- [Plugin Architecture](plugin_architecture.md)
- [Error Schema](error_schema.md)
- [Contributing Rules](contributing-rules.md)
- [Best Practices](best_practices.md)