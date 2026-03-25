# Context Construction Patterns

Context construction is the practice of **building the right prompt content** before sending it to a language model. Rather than dumping raw text into a context window, construction patterns help you assemble structured, relevant, and token-efficient prompts that produce consistent results.

These patterns answer the question: *How do I compose what the model sees?*

## Decision Tree

```
Start here: What is your primary challenge?
|
|-- "My system prompt is a tangled mess that's hard to maintain"
|     --> System Prompt Architecture
|
|-- "I'm burning tokens on context the model doesn't need yet"
|     --> Progressive Disclosure
|
|-- "My few-shot examples are static and often irrelevant"
|     --> Few-Shot Curation
|
|-- "I need structured prompts AND dynamic examples"
|     --> System Prompt Architecture + Few-Shot Curation
|
|-- "I need to manage a long multi-turn task with growing context"
|     --> Progressive Disclosure + System Prompt Architecture
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [System Prompt Architecture](system-prompt-architecture.md) | Modular, composable system prompt sections | Maintainability and reuse |
| [Progressive Disclosure](progressive-disclosure.md) | Reveal context incrementally as needed | Token efficiency and focus |
| [Few-Shot Curation](few-shot-curation.md) | Dynamically select the best examples per task | Relevance and output quality |

## How They Compose

These three patterns work together naturally:

- **System Prompt Architecture** defines the skeleton of your prompt. It decides *what sections exist*.
- **Progressive Disclosure** decides *when each section is included*. Not every section needs to appear from the start.
- **Few-Shot Curation** populates the examples section dynamically. Instead of hardcoding examples into your system prompt template, you select them at runtime.

A mature system often uses all three: a modular prompt template (Architecture), where sections are conditionally included based on conversation state (Progressive Disclosure), and the examples section is populated via embedding similarity (Few-Shot Curation).
