# Proteus Actuarial Library Style Guide

This document outlines the coding standards and style guidelines for the Proteus Actuarial Library.

## Code Style

- **Line Length**: 88 characters (following Black's default)
- **Python Version**: 3.13+
- **Import Sorting**: Automatic via ruff (isort rules)
- **Code Formatting**: Automatic via ruff formatter
- **Whitespace**: No trailing whitespace at end of lines
- **Blank Lines**: No unnecessary blank lines (follow PEP 8 guidelines)

## Type Annotations

- **Required**: All public functions, methods, and classes must have complete type annotations
- **Authority**: Type hints are the authoritative source for type information
- **No Duplication**: Do not repeat type information in docstrings

### Type Ignore Comments

Only use `# type: ignore` when there's a legitimate reason. Always:
1. Use specific error codes, not broad ignores
2. Document why the ignore is necessary
3. Consider if there's a better solution (type assertion, protocol, etc.)

**Good:**
```python
# plotly-stubs has incomplete type information for Figure.show()
fig.show()  # type: ignore[misc]

# pandas-stubs incorrectly types DataFrame.pivot return value
pivoted_df = df.pivot(  # type: ignore[assignment]
    index="id", columns="category", values="amount"
)
```

**Bad:**
```python
fig.show()  # type: ignore  # Too broad, no explanation

# Unnecessary ignore - fix the actual type issue instead
result = some_function()  # type: ignore
```

**Legitimate reasons for type ignores:**
- Incomplete or incorrect type stubs in third-party libraries
- Known limitations in the type checker (e.g., complex generics)
- Dynamic code that's correct but hard for static analysis to understand

```python
# Good
def calculate_premium(base_amount: float, rate: float) -> float:
    """Calculate insurance premium based on base amount and rate.
    
    Args:
        base_amount: The base insurance amount
        rate: The premium rate as a decimal
        
    Returns:
        The calculated premium amount
    """
    return base_amount * rate

# Bad - type information duplicated in docstring
def calculate_premium(base_amount: float, rate: float) -> float:
    """Calculate insurance premium based on base amount and rate.
    
    Args:
        base_amount (float): The base insurance amount
        rate (float): The premium rate as a decimal
        
    Returns:
        float: The calculated premium amount
    """
    return base_amount * rate
```

## Comments

- **Purpose**: Comments should explain WHY, not WHAT the code does
- **Line Length**: Must not exceed 88 characters per line
- **Quality**: The code itself shows what it does - comments that repeat this are redundant noise
- **Good comments explain**:
  - Business logic and domain-specific rules
  - Non-obvious design decisions
  - Edge cases and their handling
  - Performance or security considerations

```python
# Bad - explains WHAT (redundant)
# Loop through all items in the list
for item in items:
    process(item)

# Good - explains WHY
# Process items in reverse order to avoid index shifting when removing elements
for item in reversed(items):
    if should_remove(item):
        items.remove(item)
```

## Docstrings

- **Style**: Google-style docstrings
- **Required**: All public modules, classes, functions, and methods
- **Required**: Test functions should have docstrings explaining what they test
- **No Types**: Do not include type information in docstrings (use type hints instead)

### Function/Method Docstrings

```python
def process_claims(claims: list[Claim], policy: Policy) -> ClaimResult:
    """Process a batch of insurance claims against a policy.
    
    Validates each claim against policy terms and calculates settlements.
    Claims that fail validation are marked as rejected.
    
    Args:
        claims: List of claims to process
        policy: The insurance policy to validate against
        
    Returns:
        Processing results including settlements and rejections
        
    Raises:
        PolicyExpiredError: If the policy has expired
        InvalidClaimError: If any claim is malformed
    """
```

### Class Docstrings

```python
class ActuarialModel:
    """Base class for stochastic actuarial modeling.
    
    Provides common functionality for Monte Carlo simulations,
    risk calculations, and statistical analysis of insurance portfolios.
    
    Attributes:
        simulation_count: Number of Monte Carlo iterations
        random_seed: Seed for reproducible random number generation
    """
```

### Test Docstrings

```python
def test_premium_calculation_with_zero_rate():
    """Test that premium calculation returns zero when rate is zero."""
    
def test_policy_validation_rejects_expired_policies():
    """Test that policy validation properly rejects expired policies."""
```

## Security

- **No Secrets**: Never commit API keys, passwords, or sensitive data
- **Input Validation**: Validate all external inputs
- **SQL Injection**: Use parameterized queries for database operations

## Static Analysis Tools

The following tools are configured and must pass in CI:

- **ruff**: Linting, formatting, import sorting, docstring validation
- **mypy**: Type checking with strict mode
- **bandit**: Security vulnerability scanning
- **vulture**: Dead code detection

## VS Code Configuration

Install these extensions for consistent development experience:

- **Ruff** (`charliermarsh.ruff`) - Primary linter and formatter
- **Pylance** (`ms-python.pylance`) - Type checking and IntelliSense
- **Python** (`ms-python.python`) - Core Python support

The project includes `.vscode/settings.json` with tool configurations that match CI.

## Enforcement

All static analysis checks must pass before code can be merged. The CI pipeline will:

1. Run ruff for linting and formatting checks
2. Run mypy for type checking
3. Run bandit for security scanning
4. Run vulture for dead code detection

Use `pdm run` commands locally to ensure compliance before pushing.