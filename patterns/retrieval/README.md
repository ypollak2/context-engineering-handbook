# Context Retrieval Patterns

Context retrieval is the practice of **fetching the right external knowledge** at the right time and assembling it into a form the model can use effectively. Rather than stuffing everything into the prompt upfront or hoping the model knows the answer, retrieval patterns dynamically pull in relevant information from vector stores, tool registries, APIs, and file systems.

These patterns answer the question: *How do I get the right information into the context window when the model needs it?*

## Decision Tree

```
Start here: What is your retrieval challenge?
|
|-- "I'm pre-loading too much context that might never be used"
|     --> Just-in-Time Retrieval
|
|-- "I'm doing RAG but my retrieved chunks are noisy, redundant, or blow my token budget"
|     --> RAG Context Assembly
|
|-- "My agent has too many tools to fit all their descriptions in the prompt"
|     --> Semantic Tool Selection
|
|-- "I need lazy-loaded context AND clean chunk assembly"
|     --> Just-in-Time Retrieval + RAG Context Assembly
|
|-- "My agent needs to pick tools AND retrieve knowledge for those tools"
|     --> Semantic Tool Selection + RAG Context Assembly
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Just-in-Time Retrieval](just-in-time-retrieval.md) | Fetch context only at the moment it's needed | Freshness and token efficiency |
| [RAG Context Assembly](rag-context-assembly.md) | Retrieve, rank, deduplicate, and budget retrieved chunks | Retrieval quality and coherence |
| [Semantic Tool Selection](semantic-tool-selection.md) | Dynamically select relevant tool descriptions via embedding similarity | Scalability with large tool catalogs |

## How They Compose

These three patterns address different stages of the retrieval pipeline:

- **Just-in-Time Retrieval** decides *when* to fetch. It prevents wasteful pre-loading by triggering retrieval only when the conversation state or user intent demands it.
- **RAG Context Assembly** decides *what makes it into the prompt*. Once retrieval fires, raw chunks need ranking, deduplication, token budgeting, and source attribution before they belong in a context window.
- **Semantic Tool Selection** is a specialized retrieval problem for *capabilities rather than knowledge*. When an agent has dozens or hundreds of tools, this pattern narrows the tool menu to only what is relevant.

A production agent often combines all three: tool descriptions are selected semantically (Semantic Tool Selection), knowledge retrieval is deferred until the agent actually needs it (Just-in-Time), and retrieved chunks are assembled carefully before injection (RAG Context Assembly).
