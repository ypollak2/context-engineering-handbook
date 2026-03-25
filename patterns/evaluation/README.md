# Context Evaluation Patterns

Context evaluation is the practice of **measuring and monitoring the quality of your context over time**. Building good context is necessary but not sufficient -- you also need to know when context has degraded, become contradictory, or drifted from its intended purpose. Evaluation patterns provide the observability layer for context engineering.

These patterns answer the question: *How do I know when my context is failing, and what do I do about it?*

## Decision Tree

```
Start here: What is your evaluation challenge?
|
|-- "My agent works well initially but degrades over long sessions"
|     --> Context Rot Detection (monitor and remediate context decay)
|
|-- "I suspect my context has contradictory or stale information"
|     --> Context Rot Detection
|
|-- "I need automated quality checks on context health"
|     --> Context Rot Detection
|
|-- "My production agents drift from their instructions over time"
|     --> Context Rot Detection (instruction adherence monitoring)
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Context Rot Detection](context-rot-detection.md) | Monitor context health and trigger remediation when quality degrades | Reliability over long-running sessions |

## Why Evaluation Matters

Context engineering without evaluation is like software engineering without tests. You might build something that works initially, but you have no way to know when it breaks. This is especially dangerous because context degradation is *silent* -- the model does not tell you it has forgotten your instructions or that its context is full of contradictions. It simply produces worse output.

Evaluation patterns close this feedback loop. They provide automated detection of context problems, enabling either human-in-the-loop review or automatic remediation. As context engineering matures as a discipline, expect this category to grow with patterns for A/B testing context strategies, measuring context ROI, and benchmarking context quality across providers.
