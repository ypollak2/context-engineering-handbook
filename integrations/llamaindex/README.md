# LlamaIndex Integration

Implements 3 Context Engineering Handbook patterns using LlamaIndex v0.11+ APIs, focused on RAG, memory, and context quality monitoring.

## Patterns Covered

| Pattern | File (Python) | File (TypeScript) | Why This Framework |
|---------|---------------|--------------------|--------------------|
| [RAG Context Assembly](../../patterns/retrieval/rag-context-assembly.md) | `rag_context_assembly.py` | `rag-context-assembly.ts` | Custom `NodePostprocessor` pipeline with re-ranking and token budgeting |
| [Episodic Memory](../../patterns/persistence/episodic-memory.md) | `episodic_memory.py` | `episodic-memory.ts` | Custom `ChatStore` that captures and retrieves session episodes |
| [Context Rot Detection](../../patterns/evaluation/context-rot-detection.md) | `context_rot_detection.py` | `context-rot-detection.ts` | Custom evaluator module that checks context health |

## Why These Patterns

LlamaIndex is a RAG-first framework. These three patterns target the areas where LlamaIndex provides the strongest native abstractions:

1. **RAG Context Assembly**: LlamaIndex's `NodePostprocessor` pipeline is purpose-built for transforming retrieval results. The handbook's assembly pattern maps directly to a chain of postprocessors.

2. **Episodic Memory**: LlamaIndex's `ChatStore` and `ChatMemoryBuffer` provide the storage interface. Episodes extend this with structured capture and semantic retrieval.

3. **Context Rot Detection**: LlamaIndex's evaluation module (`ResponseEvaluator`, `FaithfulnessEvaluator`) provides the framework for custom health checks. Context rot detection extends this to monitor conversation health.

## Prerequisites

- Python 3.11+ or Node.js 20+
- An OpenAI API key (set `OPENAI_API_KEY` environment variable)
- LlamaIndex v0.11+ packages

## Quick Start

### Python

```bash
cd integrations/llamaindex/python
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

python rag_context_assembly.py
python episodic_memory.py
python context_rot_detection.py
```

### TypeScript

```bash
cd integrations/llamaindex/typescript
npm install
export OPENAI_API_KEY="sk-..."

npx tsx rag-context-assembly.ts
npx tsx episodic-memory.ts
npx tsx context-rot-detection.ts
```

## Framework-Specific Notes

### How LlamaIndex Differs from the Generic Patterns

1. **Node-based architecture**: LlamaIndex operates on `TextNode` objects rather than raw text chunks. Each node carries metadata, relationships, and embeddings. The RAG assembly pattern leverages this structure for deduplication and attribution.

2. **Postprocessor pipeline**: Instead of a monolithic assembly function, LlamaIndex uses a chain of `NodePostprocessor` objects. Each stage (re-ranking, dedup, budgeting) is a separate postprocessor that can be independently configured and tested.

3. **Index-centric retrieval**: Retrieval is tied to index types (`VectorStoreIndex`, `SummaryIndex`, etc.). The assembly pattern works as a postprocessor layer that sits on top of any index's retriever.

4. **Evaluation framework**: LlamaIndex has a built-in evaluation module for RAG quality (faithfulness, relevancy). Context rot detection extends this to monitor conversation-level health rather than single-query quality.
