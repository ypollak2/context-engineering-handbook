# LangChain Integration

Implements 5 Context Engineering Handbook patterns using LangChain v0.3+ (LCEL, Runnables, and the modern package structure).

## Patterns Covered

| Pattern | File (Python) | File (TypeScript) | Why This Framework |
|---------|---------------|--------------------|--------------------|
| [Progressive Disclosure](../../patterns/construction/progressive-disclosure.md) | `progressive_disclosure.py` | `progressive-disclosure.ts` | Custom `Runnable` that stages context injection based on conversation state |
| [Conversation Compaction](../../patterns/compression/conversation-compaction.md) | `conversation_compaction.py` | `conversation-compaction.ts` | Replaces `ConversationSummaryBufferMemory` with structured fact extraction |
| [RAG Context Assembly](../../patterns/retrieval/rag-context-assembly.md) | `rag_context_assembly.py` | `rag-context-assembly.ts` | Custom retriever chain with re-ranking, dedup, and token budgeting |
| [Semantic Tool Selection](../../patterns/retrieval/semantic-tool-selection.md) | `semantic_tool_selection.py` | `semantic-tool-selection.ts` | Dynamic tool filtering by embedding similarity before agent execution |
| [Sub-Agent Delegation](../../patterns/isolation/sub-agent-delegation.md) | `sub_agent_delegation.py` | `sub-agent-delegation.ts` | `AgentExecutor` composition with isolated contexts per sub-agent |

## Prerequisites

- Python 3.11+ or Node.js 20+
- An OpenAI API key (set `OPENAI_API_KEY` environment variable)
- LangChain v0.3+ packages

## Quick Start

### Python

```bash
cd integrations/langchain/python
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

# Run any example
python progressive_disclosure.py
python conversation_compaction.py
python rag_context_assembly.py
python semantic_tool_selection.py
python sub_agent_delegation.py
```

### TypeScript

```bash
cd integrations/langchain/typescript
npm install
export OPENAI_API_KEY="sk-..."

# Run any example
npx tsx progressive-disclosure.ts
npx tsx conversation-compaction.ts
npx tsx rag-context-assembly.ts
npx tsx semantic-tool-selection.ts
npx tsx sub-agent-delegation.ts
```

## Framework-Specific Notes

### How LangChain Differs from the Generic Patterns

1. **Runnables as the unit of composition**: LangChain v0.3 uses the LCEL (LangChain Expression Language) to compose chains. Each pattern is implemented as a custom `Runnable` or `RunnablePassthrough` chain that plugs into any existing LCEL pipeline.

2. **Message types**: LangChain has its own message classes (`HumanMessage`, `AIMessage`, `SystemMessage`). The integrations convert between handbook message types and LangChain types at boundaries.

3. **Callbacks and streaming**: LangChain's callback system enables observability hooks. The compaction and rot detection patterns use callbacks to trigger checks without modifying the main chain.

4. **Tool binding**: LangChain tools are defined with `@tool` decorators or `StructuredTool`. The semantic tool selection pattern filters tools before they reach the agent, using LangChain's native tool binding.

5. **Memory abstraction**: LangChain's memory system is deprecated in v0.3 in favor of explicit message management. The compaction pattern shows the modern approach: manage messages directly with structured extraction.

### Package Structure

All imports use the modern LangChain package split:

- `langchain_core` -- base abstractions (Runnables, messages, prompts)
- `langchain_openai` -- OpenAI model wrappers
- `langchain_anthropic` -- Anthropic model wrappers
- `langchain_community` -- community integrations
- `langchain` -- chains and agents

Avoid importing from the legacy monolithic `langchain` package for core abstractions.
