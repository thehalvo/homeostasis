# USHS Compliance Test Suite

This directory contains the compliance test suite for the Universal Self-Healing Standard (USHS) v1.0.

## Overview

The compliance test suite validates that implementations conform to the USHS specification. It tests:

- Core API endpoints
- Data schema compliance
- Security requirements
- WebSocket protocol
- Performance benchmarks
- Enterprise features

## Certification Levels

The test suite supports four certification levels:

### Bronze Certification
- Basic interface compliance
- Core API functionality
- Schema validation
- Basic authentication

### Silver Certification
- All Bronze requirements
- WebSocket support
- Security controls (TLS, rate limiting)
- Audit logging

### Gold Certification
- All Silver requirements
- Advanced features (validation, deployment)
- Performance requirements
- Extension support

### Platinum Certification
- All Gold requirements
- Enterprise features (RBAC, SSO)
- Compliance controls (HIPAA, SOX)
- High availability

## Running Tests

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

Run Bronze-level compliance tests:

```bash
python compliance_runner.py \
  --config compliance-suite.yaml \
  --level bronze \
  --api-url https://api.example.com/ushs/v1 \
  --token your-auth-token
```

### Advanced Options

```bash
python compliance_runner.py \
  --config compliance-suite.yaml \
  --level gold \
  --api-url https://api.example.com/ushs/v1 \
  --ws-url wss://api.example.com/ushs/v1 \
  --token your-auth-token \
  --output ./results \
  --format json html junit
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--config` | Path to test configuration file | No (default: compliance-suite.yaml) |
| `--level` | Target certification level (bronze/silver/gold/platinum) | No (default: bronze) |
| `--api-url` | Base URL for API tests | Yes |
| `--ws-url` | Base URL for WebSocket tests | No (derived from api-url) |
| `--token` | Bearer authentication token | No* |
| `--api-key` | API key for authentication | No* |
| `--output` | Output directory for reports | No (default: ./compliance-results) |
| `--format` | Report formats (json/html/junit) | No (default: json html) |

*At least one authentication method is required

## Test Configuration

The test suite is configured via `compliance-suite.yaml`. Key sections:

### Test Categories
- `core-apis`: Basic API functionality
- `data-schemas`: Schema validation
- `basic-auth`: Authentication methods
- `websocket`: Real-time communication
- `security`: Security controls
- `audit`: Logging and tracking
- `performance`: Response time and throughput
- `enterprise`: Advanced features
- `compliance`: Regulatory requirements

### Test Types

#### API Tests
Test REST API endpoints with various scenarios:
```yaml
- id: error-reporting
  category: core-apis
  endpoints:
    - method: POST
      path: /errors
      scenarios:
        - name: Valid error submission
          request:
            body: {...}
          expectedResponse:
            status: 201
```

#### Schema Tests
Validate JSON schemas:
```yaml
- id: error-event-schema
  category: data-schemas
  schemaValidation:
    schema: ../schemas/error-event.json
    samples:
      - valid: {...}
      - invalid: {...}
```

#### Security Tests
Test security requirements:
```yaml
- id: tls-version
  category: security
  tlsTests:
    - version: TLS1.2
      expectedResult: connection_refused
```

#### Performance Tests
Validate performance requirements:
```yaml
- id: response-time
  category: performance
  requirements:
    - endpoint: GET /health
      maxResponseTime: 100ms
      percentile: p99
```

## Test Reports

The test runner generates reports in multiple formats:

### JSON Report
Machine-readable format with complete test details:
```json
{
  "suite_name": "ushs-compliance-v1.0",
  "certification_level": "silver",
  "summary": {
    "total": 25,
    "passed": 24,
    "failed": 1,
    "pass_rate": 96.0
  },
  "test_cases": [...]
}
```

### HTML Report
Human-readable report with:
- Visual summary
- Certification badge
- Detailed test results
- Failure reasons

### JUnit XML
Compatible with CI/CD systems:
```xml
<testsuites>
  <testsuite name="ushs-compliance-v1.0" tests="25" failures="1">
    <testcase classname="core-apis" name="Error Reporting API" time="0.234"/>
    ...
  </testsuite>
</testsuites>
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: USHS Compliance Tests

on: [push, pull_request]

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r standards/v1.0/tests/requirements.txt
          
      - name: Run compliance tests
        run: |
          python standards/v1.0/tests/compliance_runner.py \
            --level silver \
            --api-url ${{ secrets.API_URL }} \
            --token ${{ secrets.API_TOKEN }} \
            --format junit
            
      - name: Publish test results
        uses: dorny/test-reporter@v1
        if: always()
        with:
          name: USHS Compliance Results
          path: './compliance-results/*.xml'
          reporter: java-junit
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Compliance Tests') {
            steps {
                sh '''
                    python compliance_runner.py \
                        --level gold \
                        --api-url ${API_URL} \
                        --token ${API_TOKEN} \
                        --format junit
                '''
            }
            post {
                always {
                    junit 'compliance-results/*.xml'
                }
            }
        }
    }
}
```

## Manual Testing

Some tests require manual verification or specialized tools:

### TLS Testing
Use OpenSSL or similar tools:
```bash
# Test TLS versions
openssl s_client -connect api.example.com:443 -tls1_2
openssl s_client -connect api.example.com:443 -tls1_3
```

### Load Testing
Use the included Locust configuration:
```bash
locust -f load_tests.py --host=https://api.example.com/ushs/v1
```

### Security Scanning
Recommended tools:
- OWASP ZAP for API security testing
- Burp Suite for comprehensive security analysis
- nmap for network security verification

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify token/API key is valid
   - Check token expiration
   - Ensure proper header format

2. **Connection Errors**
   - Verify API URL is accessible
   - Check firewall/proxy settings
   - Validate SSL certificates

3. **Schema Validation Failures**
   - Ensure schemas are in correct location
   - Validate JSON syntax
   - Check for version mismatches

4. **WebSocket Connection Issues**
   - Verify WebSocket URL format
   - Check for proxy interference
   - Ensure authentication is passed correctly

### Debug Mode

Enable verbose logging:
```bash
export USHS_LOG_LEVEL=DEBUG
python compliance_runner.py --config compliance-suite.yaml ...
```

## Contributing

To add new tests:

1. Add test definition to `compliance-suite.yaml`
2. Assign to appropriate category
3. Include in certification level requirements
4. Update documentation
5. Test locally before submitting

## License

The USHS Compliance Test Suite is licensed under the MIT License.
