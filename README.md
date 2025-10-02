# {PROJECT_NAME}

## Overview

{AGENT_OVERVIEW_DESCRIPTION}
_Brief overview of the agent_

## Features

{AGENT_CAPABILITIES}
> _Highlight the agent capabilities, including tools, etc._

## Quick Start

### Prerequisites

- {PREREQUISITS}
> _A bulleted list with prerequisits like python version, python package manager, etc._

### Installation

1. **Clone and setup**
   ```bash
   git clone {REPO_URL}
   cd {PROJECT_NAME}
   uv sync
   uv pip install -e .
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the agent**
   ```bash
   uv run main.py
   ```

## Usage

### Basic Example

{BASIC_PYTHON_EXAMPLE}
> _A basic python example for demonstrating the use of the agent_


## Configuration

Key environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `{API_KEY_VAR}` | {LLM_PROVIDER} API key | ✅ |
| `{OPTIONAL_VAR_1}` | {OPTIONAL_VAR_1_DESC} | ❌ |
| `{OPTIONAL_VAR_2}` | {OPTIONAL_VAR_2_DESC} | ❌ |

> _A table in markdown syntax with the env variables_

## Project Structure

```
{PROJECT_TREE}
```
> _The project structure in a tree format_

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov={PACKAGE_NAME}

# Run specific test
pytest tests/test_agent.py::test_{SPECIFIC_TEST}
```

## API Reference

See [API Reference](docs/api_reference.md).

## License

{LICENSE_TYPE}

---

_Made with Flow AI_