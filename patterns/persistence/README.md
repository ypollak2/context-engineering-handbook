# Context Persistence Patterns

Context persistence is the practice of **preserving knowledge and decisions across sessions** so that agents and LLM-powered systems do not start from zero every time. Without persistence, every conversation is amnesia -- the model re-asks questions it already answered, re-discovers preferences it already learned, and repeats mistakes it already corrected.

These patterns answer the question: *How do I give my system durable memory that survives session boundaries?*

## Decision Tree

```
Start here: What is your persistence challenge?
|
|-- "My agent forgets everything between sessions"
|     --> Episodic Memory (capture and recall past session episodes)
|
|-- "I need simple, transparent memory that humans can read and edit"
|     --> Filesystem-as-Memory (structured files on disk)
|
|-- "I want rich semantic recall of past experiences"
|     --> Episodic Memory
|
|-- "I need memory that works with git, code review, and existing tools"
|     --> Filesystem-as-Memory
|
|-- "I need both human-editable knowledge AND semantic recall"
|     --> Filesystem-as-Memory (primary) + Episodic Memory (index layer)
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Episodic Memory](episodic-memory.md) | Store and retrieve complete session snapshots indexed for semantic search | Rich contextual recall across sessions |
| [Filesystem-as-Memory](filesystem-as-memory.md) | Use structured files on disk as a persistent, human-readable memory store | Transparency, simplicity, and tool compatibility |

## How They Compose

These two patterns represent different points on the complexity-transparency spectrum:

- **Filesystem-as-Memory** is the simpler, more transparent approach. Memory lives in files that humans can read, edit, and version-control. It works with existing tools (editors, grep, git) and requires no additional infrastructure. This is where most projects should start.
- **Episodic Memory** adds semantic indexing on top of raw storage. It captures richer context (tool usage, decision rationale, outcomes) and enables similarity-based retrieval. It shines when the volume of past experience is large enough that file browsing becomes impractical.

A mature system often layers both: filesystem-based memory for structured knowledge (preferences, decisions, project context) with episodic memory as a semantic index over interaction history. The filesystem provides the ground truth; the episodic layer provides fast retrieval.
