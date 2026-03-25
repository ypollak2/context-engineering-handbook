# Framework Integrations

Production-ready implementations of Context Engineering Handbook patterns using popular AI/LLM frameworks.

Each integration maps the handbook's abstract patterns to the framework's idiomatic APIs, showing how to apply context engineering principles in real applications.

## Integrations

| Framework | Language(s) | Patterns | Focus |
|-----------|-------------|----------|-------|
| [LangChain](./langchain/) | Python, TypeScript | 5 | Most comprehensive -- covers the full context lifecycle |
| [LlamaIndex](./llamaindex/) | Python, TypeScript | 3 | RAG-focused -- retrieval, memory, and context health |
| [Semantic Kernel](./semantic-kernel/) | Python | 3 | Enterprise AI -- prompt architecture and optimization |
| [Vercel AI SDK](./vercel-ai-sdk/) | TypeScript | 3 | Edge/streaming -- progressive disclosure and error handling |

## Pattern Coverage Matrix

| Pattern | LangChain | LlamaIndex | Semantic Kernel | Vercel AI SDK |
|---------|-----------|------------|-----------------|---------------|
| [Progressive Disclosure](../patterns/construction/progressive-disclosure.md) | Yes | - | - | Yes |
| [Conversation Compaction](../patterns/compression/conversation-compaction.md) | Yes | - | - | Yes |
| [RAG Context Assembly](../patterns/retrieval/rag-context-assembly.md) | Yes | Yes | - | - |
| [Semantic Tool Selection](../patterns/retrieval/semantic-tool-selection.md) | Yes | - | Yes | - |
| [Sub-Agent Delegation](../patterns/isolation/sub-agent-delegation.md) | Yes | - | - | - |
| [Episodic Memory](../patterns/persistence/episodic-memory.md) | - | Yes | - | - |
| [Context Rot Detection](../patterns/evaluation/context-rot-detection.md) | - | Yes | - | - |
| [System Prompt Architecture](../patterns/construction/system-prompt-architecture.md) | - | - | Yes | - |
| [KV-Cache Optimization](../patterns/optimization/kv-cache-optimization.md) | - | - | Yes | - |
| [Error Preservation](../patterns/optimization/error-preservation.md) | - | - | - | Yes |

## Quick Start

Each integration has its own prerequisites and setup instructions. The general flow is:

1. Pick the framework you are already using (or planning to use).
2. Read its README for prerequisites and setup.
3. Run any example file directly -- each has a `main` function or `if __name__ == "__main__"` block.
4. Adapt the patterns into your application.

## Design Principles

Every integration follows these principles:

- **Idiomatic**: Uses the framework's native abstractions, not generic wrappers.
- **Runnable**: Every file can be executed with clear imports and a main entry point.
- **Type-safe**: Full type hints (Python) or strict TypeScript types throughout.
- **Documented**: Inline comments explain how each piece maps back to the handbook pattern.
- **Production-oriented**: Error handling, token budgeting, and graceful degradation included.

## Choosing a Framework

- **LangChain**: Best if you need a general-purpose agent framework with chains, tools, and memory. The most flexible option with the broadest pattern coverage.
- **LlamaIndex**: Best if your primary use case is RAG and you need deep control over retrieval, indexing, and context quality.
- **Semantic Kernel**: Best if you are building enterprise AI on Microsoft infrastructure or need tight integration with Azure OpenAI.
- **Vercel AI SDK**: Best if you are building a Next.js or edge-deployed application with streaming responses.
