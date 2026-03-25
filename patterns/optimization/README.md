# Context Optimization Patterns

Context optimization is the practice of **making the most of every token in your context window** -- reducing cost, improving latency, and preserving the highest-signal information. Optimization patterns do not change *what* you put in context; they change *how efficiently* that context is structured, cached, and maintained.

These patterns answer the question: *How do I get better results from the same (or fewer) tokens?*

## Decision Tree

```
Start here: What is your optimization challenge?
|
|-- "My LLM calls are slow and expensive, especially repeated ones"
|     --> KV-Cache Optimization (maximize cache hit rates)
|
|-- "My agent keeps repeating the same mistakes in a session"
|     --> Error Preservation (keep full error context available)
|
|-- "I'm making many similar calls and paying full price each time"
|     --> KV-Cache Optimization
|
|-- "Errors get summarized away and the agent loses debugging context"
|     --> Error Preservation
|
|-- "I need both cost efficiency AND robust self-correction"
|     --> KV-Cache Optimization + Error Preservation
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [KV-Cache Optimization](kv-cache-optimization.md) | Structure context for maximum cache reuse across LLM calls | Latency and cost reduction (up to 7x) |
| [Error Preservation](error-preservation.md) | Preserve full error context instead of summarizing it away | Faster self-correction, fewer repeated mistakes |

## How They Compose

These patterns optimize different dimensions of context efficiency:

- **KV-Cache Optimization** is a structural optimization. It focuses on *how context is arranged* so that LLM providers can reuse cached computations. The key insight is that identical prefixes are free after the first call.
- **Error Preservation** is a signal-quality optimization. It focuses on *what stays in context* when space is limited. The key insight is that error details are the highest-value tokens for self-correction, yet they are often the first to be compressed away.

Both patterns work together naturally: KV-cache optimization structures the stable parts of your context (system prompt, tool definitions) for reuse, while error preservation ensures that the dynamic parts (recent errors, debugging context) retain their full diagnostic value. Together they give you a context window that is both cost-efficient and information-rich.
