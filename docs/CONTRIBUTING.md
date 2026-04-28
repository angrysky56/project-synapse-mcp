# Contributing to Project Synapse

First off, thank you for considering contributing to Project Synapse! It’s people like you that make Synapse a great tool for the community.

## 1. Code of Conduct

By participating in this project, you agree to abide by our code of conduct. We aim to foster a welcoming and inclusive environment for everyone.

## 2. How Can I Contribute?

### 2.1. Reporting Bugs
- Before creating a new issue, check if the bug has already been reported.
- Use a clear and descriptive title for the issue.
- Describe the exact steps to reproduce the problem.
- Include information about your environment (OS, Python version, Neo4j version).

### 2.2. Suggesting Enhancements
- Check if the feature has already been suggested.
- Explain why this enhancement would be useful to most Project Synapse users.
- Provide a clear and detailed description of the suggested enhancement.

### 2.3. Your First Code Contribution
Unsure where to begin? Look for issues labeled `good first issue` or `help wanted`.

## 3. Pull Request Process

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies** using `uv pip install -e ".[dev]"`.
3.  **Make your changes**. Ensure your code follows the established style guides.
4.  **Add tests** for any new functionality.
5.  **Run the test suite** with `uv run pytest` to ensure no regressions.
6.  **Run the linter and formatter**:
    - `uv run ruff check .`
    - `uv run black .`
    - `uv run mypy src`
7.  **Submit a Pull Request**. Provide a clear description of the changes and link to any relevant issues.

## 4. Style Guides

### 4.1. Python Style Guide
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use `black` for formatting.

### 4.2. Commit Messages
We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:
- `feat:` for new features.
- `fix:` for bug fixes.
- `docs:` for documentation changes.
- `refactor:` for code changes that neither fix a bug nor add a feature.
- `test:` for adding missing tests or correcting existing tests.

## 5. Community

If you have questions or want to discuss the project, feel free to open a GitHub Discussion or join our community (if applicable).

Happy coding!
