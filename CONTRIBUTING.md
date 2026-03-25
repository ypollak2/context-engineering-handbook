# Contributing to Context Engineering Handbook

Thanks for your interest in contributing. This is a community resource and all contributions are welcome, from fixing a typo to proposing entirely new patterns.

## Ways to Contribute

### Submit a New Pattern

1. Copy `patterns/_template/` to `patterns/your-pattern-name/`.
2. Fill in every section of the template. Provide code examples in **both Python and TypeScript**.
3. Include a working, self-contained example in the `examples/` directory if the pattern benefits from a runnable demo.
4. Open a PR with the prefix `pattern:` in the title (e.g., `pattern: add semantic caching`).

Before writing, check [open issues](../../issues) and existing patterns to avoid duplicating work.

### Improve an Existing Pattern

- Fix bugs in code examples (please verify they run before submitting).
- Add missing language examples (Python or TypeScript).
- Improve explanations, add diagrams, or clarify edge cases.
- Prefix your PR title with `improve:` (e.g., `improve: fix token counting in layered context example`).

### Report an Anti-Pattern

Seen a context engineering mistake in production? We want to hear about it.

1. Open an issue using the **New Pattern Proposal** template.
2. Describe the symptoms, root cause, and impact you observed.
3. Suggest a fix or reference an existing pattern that addresses it.

### Report a Bug or Inaccuracy

Use the **Bug Report** issue template to flag incorrect information, broken code examples, or outdated references.

## Code Style

- Keep examples **self-contained and runnable**. A reader should be able to copy-paste and execute.
- Use clear variable names. Avoid abbreviations that aren't universally understood.
- Include comments only to explain non-obvious decisions.
- Target Python 3.10+ and TypeScript 5+.
- Avoid external dependencies in examples unless the pattern specifically requires them. When dependencies are necessary, document them.

## PR Process

1. Fork the repo and create a branch from `main`.
2. Make your changes. Run any code examples to verify they work.
3. Write a clear PR description explaining what you changed and why.
4. Submit the PR. A maintainer will review it, usually within a few days.
5. Address any feedback. Once approved, a maintainer will merge.

## Code of Conduct

Be respectful, constructive, and assume good intent. We are all here to learn.
