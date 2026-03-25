# Context Compression Patterns

Context compression is the practice of **reducing the token footprint of existing context** without losing the information the model needs to continue performing well. As conversations grow, tool outputs accumulate, and agent loops iterate, the context window fills up. Compression patterns reclaim that space by summarizing, truncating, or masking content that has already served its purpose.

These patterns answer the question: *How do I keep a long-running session from hitting the context ceiling?*

## Decision Tree

```
Start here: What is filling your context window?
|
|-- "The conversation history is too long; old turns are crowding out new ones"
|     --> Conversation Compaction
|
|-- "Tool outputs and observations are bloating the context"
|     --> Observation Masking
|
|-- "Both conversation turns AND tool outputs are growing unbounded"
|     --> Observation Masking first (cheaper), then Conversation Compaction
|
|-- "I need lossless compression, not summarization"
|     --> Observation Masking (deterministic, no LLM call)
|
|-- "I need the model to retain nuanced decisions from earlier in the session"
|     --> Conversation Compaction (LLM-powered summarization preserves semantics)
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Conversation Compaction](conversation-compaction.md) | Summarize older conversation turns into condensed fact blocks | Unbounded session length with semantic preservation |
| [Observation Masking](observation-masking.md) | Selectively hide or truncate tool outputs that have been consumed | 50% cost savings without LLM summarization overhead |

## How They Compose

These two patterns operate on different parts of the context and complement each other naturally:

- **Observation Masking** is the first line of defense. It targets tool outputs (file contents, API responses, command results) that were useful when they appeared but are no longer needed. It is fast, deterministic, and requires no additional LLM calls.
- **Conversation Compaction** is the deeper intervention. It targets the conversational turns themselves, distilling multi-turn reasoning chains into compact summaries that preserve decisions, facts, and user preferences.

A production agent typically applies both: observation masking runs continuously to keep tool outputs lean, and conversation compaction triggers periodically when the overall token count crosses a threshold. Together, they can sustain sessions that would otherwise exhaust the context window within minutes.
