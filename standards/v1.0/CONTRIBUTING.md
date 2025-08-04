# Contributing to USHS

Thank you for your interest in contributing to the Universal Self-Healing Standard! This guide will help you get started.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Ways to Contribute](#ways-to-contribute)
3. [Development Process](#development-process)
4. [Coding Standards](#coding-standards)
5. [Submitting Changes](#submitting-changes)
6. [Review Process](#review-process)
7. [Community](#community)

## Getting Started

### Prerequisites

- Git and GitHub account
- Familiarity with Markdown
- Understanding of self-healing concepts (see [CONCEPTS.md](./docs/CONCEPTS.md))
- Development environment for your chosen language

### Setting Up

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/ushs-standards.git
   cd ushs-standards
   git remote add upstream https://github.com/ushs/standards.git
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

3. **Set Up Development Environment**
   ```bash
   # Python
   cd standards/v1.0/reference/python
   pip install -r requirements-dev.txt
   
   # TypeScript
   cd standards/v1.0/reference/typescript
   npm install
   
   # Go
   cd standards/v1.0/reference/go
   go mod download
   ```

## Ways to Contribute

### Documentation

**Perfect for first-time contributors!**

- Fix typos and grammar
- Improve clarity of explanations
- Add examples and use cases
- Translate documentation
- Write tutorials and guides

### Bug Reports

**Help us improve quality**

- Search existing issues first
- Use issue templates
- Provide minimal reproducible examples
- Include environment details
- Suggest potential fixes

### Feature Proposals

**Shape the future of USHS**

- Start with community discussion
- Write USHS Improvement Proposals (UIPs)
- Prototype new features
- Provide use case justification

### Code Contributions

**Implement and improve**

- Fix bugs
- Implement approved features
- Improve performance
- Add tests
- Enhance reference implementations

### Testing

**Ensure quality and compliance**

- Add test cases
- Improve test coverage
- Test on different platforms
- Validate compliance suite
- Performance testing

### Community Support

**Help others succeed**

- Answer questions in the community
- Review pull requests
- Mentor new contributors
- Write blog posts
- Give talks

## Development Process

### 1. Before You Start

- **Check existing work**: Search issues and PRs
- **Discuss large changes**: Open an issue first
- **Claim issues**: Comment to avoid duplicate work
- **Ask questions**: We're here to help!

### 2. Making Changes

#### For Specification Changes

1. **Read current spec carefully**
2. **Consider backward compatibility**
3. **Update all affected sections**
4. **Add to changelog**

Example:
```markdown
## Changes in v1.1.0

### Added
- New `healingPolicy` field in session schema (#123)

### Changed
- Extended `severity` enum to include 'info' level (#124)

### Deprecated
- `legacyFormat` field will be removed in v2.0 (#125)
```

#### For Code Changes

1. **Follow language conventions**
2. **Maintain consistent style**
3. **Add/update tests**
4. **Update documentation**

### 3. Testing Your Changes

#### Specification Changes
```bash
# Validate schemas
cd standards/v1.0
python scripts/validate_schemas.py

# Check internal links
python scripts/check_links.py
```

#### Reference Implementation Changes
```bash
# Python
cd reference/python
pytest
flake8
mypy ushs_client.py

# TypeScript
cd reference/typescript
npm test
npm run lint
npm run type-check

# Go
cd reference/go
go test ./...
go vet ./...
golangci-lint run
```

#### Compliance Suite Changes
```bash
cd tests
python compliance_runner.py --config compliance-suite.yaml --level bronze
```

## Coding Standards

### General Principles

1. **Clarity over cleverness**
2. **Consistent with existing code**
3. **Comprehensive error handling**
4. **Meaningful variable names**
5. **Comments for "why", not "what"**

### Language-Specific

#### Python
```python
# Follow PEP 8
# Use type hints
async def report_error(self, error: ErrorEvent) -> Tuple[str, str]:
    """Report an error and return (error_id, session_id)."""
    # Implementation
```

#### TypeScript
```typescript
// Use strict mode
// Prefer interfaces over types
interface ErrorEvent {
  severity: Severity;
  source: ErrorSource;
  error: ErrorDetails;
}
```

#### Go
```go
// Follow effective Go guidelines
// Handle all errors
if err != nil {
    return fmt.Errorf("report error: %w", err)
}
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Explain the "why" behind decisions
- Keep line length under 100 characters
- Use semantic line breaks

## Submitting Changes

### Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Commit your changes**
   ```bash
   # Use meaningful commit messages
   git commit -m "feat: add support for custom healing policies

   - Add healingPolicy field to session schema
   - Update validation logic
   - Add tests for policy enforcement
   
   Fixes #123"
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Use PR template
   - Reference related issues
   - Describe changes clearly
   - Include test results

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(api): add retry mechanism for failed healings

fix(python): handle connection timeout in WebSocket client

docs(adoption): add migration guide for AWS users
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123
Related to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Backward compatible
- [ ] Security implications considered

## Testing
Describe testing performed

## Screenshots (if applicable)
Add screenshots for UI changes
```

## Review Process

### Timeline

- **Initial Review**: Within 48 hours
- **Feedback Round**: 3-7 days
- **Final Decision**: Within 14 days

### Review Criteria

1. **Correctness**: Does it work as intended?
2. **Completeness**: Are all cases handled?
3. **Compatibility**: Does it break existing functionality?
4. **Quality**: Does it follow standards?
5. **Security**: Are there security implications?
6. **Performance**: Is there a performance impact?

### Review Etiquette

**For Contributors**:
- Be patient and respectful
- Respond to feedback promptly
- Be open to suggestions
- Ask for clarification when needed

**For Reviewers**:
- Be constructive and specific
- Suggest improvements
- Acknowledge good work
- Explain the "why" behind requests

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and features
- **GitHub Discussions**: General discussions
- **Community Chat**: Real-time discussions
- **Forum**: In-depth discussions
- **Mailing List**: Announcements

### Getting Help

- **Documentation**: Check docs first
- **Community Help**: Quick questions
- **GitHub Discussions**: Detailed help
- **Office Hours**: Tuesdays 3PM UTC

### Recognition

We recognize contributors through:
- Contributors file
- Release notes mentions
- Community spotlight
- Committer invitations

## Legal

### Contributor License Agreement

By submitting a pull request, you agree that:

1. You have the right to contribute the code
2. You grant us a perpetual, worldwide, non-exclusive, royalty-free license
3. Your contributions are under the same license as the project

### Code of Conduct

All contributors must follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Tips for Success

### First Time Contributors

1. Start small (documentation, typos)
2. Read existing code first
3. Ask questions early
4. Be patient with yourself
5. Celebrate small wins!

### Effective Contributions

1. **One PR = One Concern**: Keep PRs focused
2. **Test Thoroughly**: Add tests for new code
3. **Document Well**: Update all relevant docs
4. **Communicate Clearly**: Explain your reasoning
5. **Be Responsive**: Address feedback promptly

### Common Mistakes to Avoid

- Large, unfocused PRs
- Missing tests or documentation
- Breaking changes without discussion
- Ignoring CI failures
- Not following coding standards

## Thank You!

Your contributions make USHS better for everyone. We appreciate your time and effort in improving the standard.

---

*Happy Contributing!*
