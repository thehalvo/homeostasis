# Scala Integration

This document provides comprehensive information about Homeostasis' support for Scala applications. Homeostasis can analyze, detect, and fix common Scala errors, including Scala-specific patterns and errors from popular frameworks like Akka and Play Framework.

## Features

The Scala integration for Homeostasis provides the following features:

- Error detection and analysis for common Scala runtime exceptions
- Support for functional programming patterns and idioms specific to Scala
- Integration with popular Scala frameworks:
  - Akka actor system error detection
  - Play Framework error handling
  - SBT build tool errors
- Automatic fix generation for common Scala issues
- Support for JVM-related errors (since Scala runs on the JVM)

## Supported Error Types

### Core Scala Errors

- NullPointerException (with suggestions to use Option instead)
- NoSuchElementException (for collections and Option handling)
- MatchError (for incomplete pattern matching)
- IndexOutOfBoundsException
- ClassCastException
- Type mismatch errors
- Option-handling errors
- Division by zero
- Stack overflow (with suggestions for tail recursion)

### Concurrency Errors

- Future timeouts
- Deadlocks
- Thread interruption
- Race conditions
- Memory visibility issues
- Promise completion errors

### Akka Framework Errors

- ActorNotFound exceptions
- Invalid actor state
- Actor system configuration errors
- Actor initialization issues
- Dead letter detection
- Serialization errors
- Akka Streams issues

### Play Framework Errors

- Application initialization errors
- Routing exceptions
- Configuration errors
- Template compilation errors
- Form binding errors
- JSON validation errors
- WebSocket errors

### SBT Build Errors

- Compilation errors
- Dependency resolution issues
- Version conflicts
- Plugin configuration problems
- Task execution errors

## Integrating with Your Scala Application

### Prerequisites

- Scala 2.12+ application
- JVM 8 or higher
- Homeostasis core installed

### Basic Setup

Add the Homeostasis dependency to your build.sbt:

```scala
libraryDependencies += "com.example.homeostasis" %% "homeostasis-scala" % "0.1.0"
```

### Configuration

#### For Standard Scala Applications

For a typical Scala application, add the following to your application.conf:

```hocon
homeostasis {
  enabled = true
  rules {
    scala {
      enabled = true
      # Customize which rule categories to enable
      categories = ["core", "typesystem", "collections", "concurrency"]
    }
  }
  patch-generation {
    enabled = true
    templates-dir = "/path/to/custom/templates" # Optional
  }
}
```

#### For Akka Applications

For Akka-based applications, add:

```hocon
homeostasis {
  enabled = true
  rules {
    scala {
      enabled = true
      categories = ["core", "typesystem", "collections", "concurrency"]
    }
    akka {
      enabled = true
      categories = ["actor", "stream", "remote"]
    }
  }
  # Akka-specific settings
  akka {
    supervision-strategy = "restart-with-backoff"
    detect-dead-letters = true
    monitor-mailbox-size = true
  }
}
```

#### For Play Framework Applications

For Play Framework applications, add to your application.conf:

```hocon
homeostasis {
  enabled = true
  rules {
    scala {
      enabled = true
      categories = ["core", "typesystem", "collections", "concurrency"]
    }
    play {
      enabled = true
      categories = ["routes", "forms", "json", "templates"]
    }
  }
  # Play-specific settings
  play {
    monitor-action-composition = true
    track-request-errors = true
  }
}
```

### Initialization

#### For Standard Scala Applications

```scala
import com.example.homeostasis.scala.ScalaHealing

object MyApp {
  def main(args: Array[String]): Unit = {
    // Initialize Homeostasis
    ScalaHealing.initialize()
    
    // Your application code
    // ...
  }
}
```

#### For Akka Applications

```scala
import com.example.homeostasis.scala.AkkaHealing
import akka.actor.ActorSystem

object MyAkkaApp {
  def main(args: Array[String]): Unit = {
    val system = ActorSystem("my-system")
    
    // Initialize Homeostasis with Akka integration
    AkkaHealing.initialize(system)
    
    // Your Akka application code
    // ...
  }
}
```

#### For Play Framework Applications

In your app's Module class:

```scala
import com.example.homeostasis.scala.PlayHealing
import play.api.{Application, ApplicationLoader, Environment}

class HomeostasisModule extends Module {
  def bindings(environment: Environment, configuration: Configuration): Seq[Binding[_]] = {
    Seq(
      bind[PlayHealing].toSelf.eagerly()
    )
  }
}
```

## Advanced Configuration

### Custom Error Rules

You can add custom error detection rules to handle specific error patterns in your application:

```hocon
homeostasis {
  rules {
    custom {
      paths = ["/path/to/custom-rules.json"]
    }
  }
}
```

Example custom rule file (custom-rules.json):

```json
{
  "rules": [
    {
      "id": "my_custom_error",
      "pattern": "com\\.mycompany\\.CustomException: (.*)",
      "type": "CustomException",
      "description": "Custom business logic exception",
      "root_cause": "custom_business_logic_error",
      "suggestion": "Check the business logic flow and ensure valid inputs",
      "confidence": "high",
      "severity": "medium",
      "category": "business_logic",
      "framework": "custom"
    }
  ]
}
```

### Custom Fix Templates

You can provide custom templates for fix generation:

1. Create a directory for your custom templates
2. Add template files with the naming pattern: `{error_type}.scala.template`
3. Configure the path in your configuration:

```hocon
homeostasis {
  patch-generation {
    templates-dir = "/path/to/custom/templates"
  }
}
```

Example template file (custom_business_logic_error.scala.template):

```scala
// Custom fix for business logic error
try {
  // Original code that might throw CustomException
  ${ORIGINAL_CODE}
} catch {
  case ex: CustomException =>
    // Custom recovery logic
    logger.warn(s"Handling custom exception: ${ex.getMessage}")
    fallbackBehavior()
}
```

## Performance Considerations

The Scala integration adds minimal overhead to your application:

- Rule-based detection adds <1ms per error analysis
- Template-based fix generation is only performed when an error is detected
- Runtime monitoring is done using lightweight JVM hooks

To optimize performance:

- Disable unnecessary rule categories
- Configure appropriate logging levels
- For production, consider using the monitoring-only mode:

```hocon
homeostasis {
  mode = "monitor-only" # Options: full, monitor-only, suggestion-only
}
```

## Limitations

- Some complex Scala type errors may not be fully analyzable
- Advanced type-level programming patterns might not be supported
- Custom monadic error handling may need custom rules
- Macros and compile-time errors are not supported in runtime monitoring

## Examples

### Handling NullPointerException

Original code:

```scala
def processName(user: User): String = {
  val username = user.name
  username.toUpperCase
}
```

After Homeostasis healing:

```scala
def processName(user: User): String = {
  Option(user).flatMap(u => Option(u.name)) match {
    case Some(username) => username.toUpperCase
    case None => 
      log.warn("User or user.name was null")
      "UNKNOWN" // Default value
  }
}
```

### Fixing Incomplete Pattern Matching

Original code:

```scala
sealed trait PaymentMethod
case class CreditCard(number: String) extends PaymentMethod
case class BankTransfer(accountId: String) extends PaymentMethod
case class DigitalWallet(provider: String, id: String) extends PaymentMethod

def processPayment(method: PaymentMethod): String = method match {
  case CreditCard(number) => s"Processing credit card: $number"
  case BankTransfer(accountId) => s"Processing bank transfer: $accountId"
  // Missing DigitalWallet case!
}
```

After Homeostasis healing:

```scala
def processPayment(method: PaymentMethod): String = method match {
  case CreditCard(number) => s"Processing credit card: $number"
  case BankTransfer(accountId) => s"Processing bank transfer: $accountId"
  case DigitalWallet(provider, id) => s"Processing $provider wallet: $id"
}
```

### Fixing Akka ActorNotFound

Original code:

```scala
def sendMessage(message: Any): Unit = {
  val selection = context.actorSelection("user/processor")
  selection ! message
}
```

After Homeostasis healing:

```scala
def sendMessage(message: Any): Unit = {
  val selection = context.actorSelection("user/processor")
  
  // Resolve actor reference first
  implicit val timeout: Timeout = 5.seconds
  val future = selection.resolveOne()
  
  future.onComplete {
    case Success(actorRef) => 
      actorRef ! message
    case Failure(_) =>
      log.error(s"Actor not found at path: user/processor")
      // Fallback behavior
  }
}
```

## Troubleshooting

### Common Issues

1. **Homeostasis isn't detecting Scala errors**
   - Ensure the Scala plugin is enabled
   - Check log levels (debug for more detailed information)
   - Verify your application configuration

2. **Fix templates not working**
   - Check template path configuration
   - Ensure template files follow the naming convention
   - Validate template syntax

3. **Missing framework-specific detection**
   - Ensure framework-specific rules are enabled
   - Add custom rules if needed for your specific framework version

### Logging

Enable debug logging for more detailed information:

```hocon
homeostasis {
  logging {
    level = "DEBUG"
    include-stacktraces = true
  }
}
```

## Contributing

### Adding New Rules

To contribute new Scala error rules:

1. Identify common error patterns
2. Create a rule definition in JSON format
3. Test the rule with example stack traces
4. Submit a pull request with the new rules

### Improving Templates

To improve fix templates:

1. Identify patterns that could be better fixed
2. Create or modify a template file
3. Test with different error scenarios
4. Submit a pull request with the improved templates

## Further Reading

- [Homeostasis Core Documentation](./README.md)
- [Cross-Language Error Handling](./cross_language_features.md)
- [JVM Integration Guide](./java_integration.md)
- [Akka Documentation](https://doc.akka.io/)
- [Play Framework Documentation](https://www.playframework.com/documentation)
- [Scala Documentation](https://docs.scala-lang.org/)