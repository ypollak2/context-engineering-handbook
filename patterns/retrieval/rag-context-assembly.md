# RAG Context Assembly

> Retrieve, rank, deduplicate, and budget retrieved chunks into a coherent context block with source attribution.

## Problem

Basic RAG implementations treat retrieval as a solved problem: embed the query, pull the top-k chunks, and paste them into the prompt. In practice, this produces noisy, redundant, and poorly formatted context that confuses the model. Chunks from the same document appear multiple times, irrelevant results dilute attention, retrieved content blows the token budget leaving no room for the model to reason, and the model cannot cite its sources because the chunks lack attribution. The gap between "we have a vector store" and "the model gives accurate, grounded answers" is the assembly layer.

## Solution

RAG Context Assembly is a multi-stage pipeline that transforms raw retrieval results into a high-quality context block. After the initial vector search returns candidate chunks, the pipeline applies re-ranking (using a cross-encoder or LLM-based scorer) to improve relevance ordering, deduplicates overlapping content, enforces a strict token budget to prevent context overflow, and formats each chunk with source metadata so the model can attribute its answers.

The key insight is that retrieval quality is not just about the embedding model or the chunking strategy -- it is about what happens *after* the vector search returns results. A well-assembled context block with 5 excellent chunks outperforms a raw dump of 20 mediocre ones.

## How It Works

```
User Query
     |
     v
+-------------------+
| Vector Search     |  Retrieve top-N candidates (N > final k)
| (over-retrieve)   |  e.g., fetch 20 when you need 5
+-------------------+
     |
     v
+-------------------+
| Re-Rank           |  Score each chunk against the query
| (cross-encoder)   |  using a more expensive but accurate model
+-------------------+
     |
     v
+-------------------+
| Deduplicate       |  Remove chunks with >80% content overlap
| (similarity hash) |  Prefer higher-ranked version
+-------------------+
     |
     v
+-------------------+
| Token Budget      |  Fill from top-ranked down until budget
| (greedy fill)     |  is exhausted; drop the rest
+-------------------+
     |
     v
+-------------------+
| Format & Attribute|  Wrap each chunk with [Source: ...]
| (structured block)|  metadata for citation
+-------------------+
     |
     v
  Context Block
  (injected into prompt)
```

## Implementation

### Python

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned from vector search."""
    content: str
    source: str         # e.g., "billing-faq.md"
    section: str        # e.g., "Refund Policy"
    score: float        # similarity score from vector search
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class AssembledContext:
    """The final assembled context block ready for prompt injection."""
    chunks: tuple[RetrievedChunk, ...]
    total_tokens: int
    dropped_count: int
    context_block: str


class RAGContextAssembler:
    """
    Multi-stage pipeline: retrieve -> re-rank -> deduplicate -> budget -> format.
    """

    def __init__(
        self,
        token_budget: int = 3000,
        similarity_threshold: float = 0.85,
        min_relevance_score: float = 0.3,
    ) -> None:
        self.token_budget = token_budget
        self.similarity_threshold = similarity_threshold
        self.min_relevance_score = min_relevance_score

    def assemble(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        reranker: Reranker | None = None,
    ) -> AssembledContext:
        """Run the full assembly pipeline."""
        # Stage 1: Re-rank if a reranker is provided
        if reranker is not None:
            chunks = reranker.rerank(query, chunks)

        # Stage 2: Filter low-relevance chunks
        chunks = [c for c in chunks if c.score >= self.min_relevance_score]

        # Stage 3: Deduplicate overlapping content
        chunks = self._deduplicate(chunks)

        # Stage 4: Budget-constrained selection
        selected, total_tokens, dropped = self._apply_budget(chunks)

        # Stage 5: Format with source attribution
        context_block = self._format(selected)

        return AssembledContext(
            chunks=tuple(selected),
            total_tokens=total_tokens,
            dropped_count=dropped,
            context_block=context_block,
        )

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks, keeping the higher-scored version."""
        if not chunks:
            return []

        kept: list[RetrievedChunk] = []
        seen_hashes: list[set[str]] = []

        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
            chunk_shingles = self._shingle(chunk.content)
            is_duplicate = False

            for existing_shingles in seen_hashes:
                overlap = len(chunk_shingles & existing_shingles) / max(
                    len(chunk_shingles | existing_shingles), 1
                )
                if overlap >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(chunk)
                seen_hashes.append(chunk_shingles)

        return kept

    def _apply_budget(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], int, int]:
        """Greedily fill the token budget from highest-ranked chunks."""
        selected: list[RetrievedChunk] = []
        total_tokens = 0
        dropped = 0

        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
            chunk_tokens = self._estimate_tokens(chunk.content)
            # Reserve tokens for the attribution header (~20 tokens per chunk)
            chunk_cost = chunk_tokens + 20

            if total_tokens + chunk_cost > self.token_budget:
                dropped += 1
                continue

            selected.append(chunk)
            total_tokens += chunk_cost

        return selected, total_tokens, dropped

    def _format(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks into a context block with source attribution."""
        if not chunks:
            return (
                "<retrieved_context>\n"
                "No relevant information found.\n"
                "</retrieved_context>"
            )

        sections: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}: {chunk.source} > {chunk.section}]"
            sections.append(f"{header}\n{chunk.content}")

        body = "\n\n---\n\n".join(sections)
        return f"<retrieved_context>\n{body}\n</retrieved_context>"

    @staticmethod
    def _shingle(text: str, n: int = 3) -> set[str]:
        """Create character n-gram shingles for overlap detection."""
        words = text.lower().split()
        if len(words) < n:
            return {text.lower()}
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 characters per token for English."""
        return len(text) // 4


class Reranker:
    """
    Re-ranks chunks using a cross-encoder model.
    In production, use a model like BAAI/bge-reranker-v2-m3 or Cohere Rerank.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        # In production: load the cross-encoder model here
        # from sentence_transformers import CrossEncoder
        # self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Re-score chunks against the query and return sorted by new score."""
        # In production, use the cross-encoder:
        # pairs = [(query, chunk.content) for chunk in chunks]
        # scores = self.model.predict(pairs)
        #
        # Simulated: boost chunks containing query terms
        query_terms = set(query.lower().split())
        reranked: list[RetrievedChunk] = []

        for chunk in chunks:
            content_terms = set(chunk.content.lower().split())
            term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            new_score = chunk.score * 0.4 + term_overlap * 0.6

            reranked.append(RetrievedChunk(
                content=chunk.content,
                source=chunk.source,
                section=chunk.section,
                score=new_score,
                metadata=chunk.metadata,
            ))

        return sorted(reranked, key=lambda c: c.score, reverse=True)


# --- Example usage ---

def main() -> None:
    assembler = RAGContextAssembler(
        token_budget=2000,
        similarity_threshold=0.85,
        min_relevance_score=0.2,
    )
    reranker = Reranker()

    # Simulated retrieval results (in production, these come from your vector store)
    raw_chunks = [
        RetrievedChunk(
            content="Refunds are processed within 5-10 business days after approval. "
                    "Contact support with your order ID to initiate a refund request.",
            source="billing-faq.md",
            section="Refund Policy",
            score=0.89,
        ),
        RetrievedChunk(
            content="Refund requests must include the order ID. Refunds take 5-10 "
                    "business days to process after the request is approved by support.",
            source="support-guide.md",
            section="Processing Refunds",
            score=0.87,
        ),
        RetrievedChunk(
            content="Invoices are generated on the 1st of each month and sent to "
                    "the billing email on file. Past invoices are available in the dashboard.",
            source="billing-faq.md",
            section="Invoice Schedule",
            score=0.62,
        ),
        RetrievedChunk(
            content="Our office hours are Monday through Friday, 9 AM to 5 PM EST.",
            source="general-info.md",
            section="Contact Us",
            score=0.25,
        ),
    ]

    result = assembler.assemble(
        query="How do I get a refund?",
        chunks=raw_chunks,
        reranker=reranker,
    )

    print(f"Chunks used: {len(result.chunks)}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Chunks dropped: {result.dropped_count}")
    print()
    print(result.context_block)


if __name__ == "__main__":
    main()
```

### TypeScript

```typescript
// rag-context-assembly.ts

interface RetrievedChunk {
  readonly content: string;
  readonly source: string;
  readonly section: string;
  readonly score: number;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

interface AssembledContext {
  readonly chunks: ReadonlyArray<RetrievedChunk>;
  readonly totalTokens: number;
  readonly droppedCount: number;
  readonly contextBlock: string;
}

interface RerankerFn {
  (query: string, chunks: RetrievedChunk[]): RetrievedChunk[];
}

interface AssemblerConfig {
  readonly tokenBudget: number;
  readonly similarityThreshold: number;
  readonly minRelevanceScore: number;
}

const DEFAULT_CONFIG: AssemblerConfig = {
  tokenBudget: 3000,
  similarityThreshold: 0.85,
  minRelevanceScore: 0.3,
};

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function shingle(text: string, n = 3): Set<string> {
  const words = text.toLowerCase().split(/\s+/);
  if (words.length < n) return new Set([text.toLowerCase()]);
  const shingles = new Set<string>();
  for (let i = 0; i <= words.length - n; i++) {
    shingles.add(words.slice(i, i + n).join(" "));
  }
  return shingles;
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  const intersection = new Set([...a].filter((x) => b.has(x)));
  const union = new Set([...a, ...b]);
  return union.size === 0 ? 0 : intersection.size / union.size;
}

function deduplicate(
  chunks: RetrievedChunk[],
  threshold: number
): RetrievedChunk[] {
  const sorted = [...chunks].sort((a, b) => b.score - a.score);
  const kept: RetrievedChunk[] = [];
  const seenShingles: Set<string>[] = [];

  for (const chunk of sorted) {
    const chunkShingles = shingle(chunk.content);
    const isDuplicate = seenShingles.some(
      (existing) => jaccardSimilarity(chunkShingles, existing) >= threshold
    );

    if (!isDuplicate) {
      kept.push(chunk);
      seenShingles.push(chunkShingles);
    }
  }

  return kept;
}

function applyBudget(
  chunks: RetrievedChunk[],
  budget: number
): { selected: RetrievedChunk[]; totalTokens: number; dropped: number } {
  const sorted = [...chunks].sort((a, b) => b.score - a.score);
  const selected: RetrievedChunk[] = [];
  let totalTokens = 0;
  let dropped = 0;

  for (const chunk of sorted) {
    const cost = estimateTokens(chunk.content) + 20; // 20 tokens for attribution
    if (totalTokens + cost > budget) {
      dropped++;
      continue;
    }
    selected.push(chunk);
    totalTokens += cost;
  }

  return { selected, totalTokens, dropped };
}

function formatContext(chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) {
    return "<retrieved_context>\nNo relevant information found.\n</retrieved_context>";
  }

  const sections = chunks.map((chunk, i) => {
    const header = `[Source ${i + 1}: ${chunk.source} > ${chunk.section}]`;
    return `${header}\n${chunk.content}`;
  });

  return `<retrieved_context>\n${sections.join("\n\n---\n\n")}\n</retrieved_context>`;
}

function assembleContext(
  query: string,
  chunks: RetrievedChunk[],
  config: Partial<AssemblerConfig> = {},
  reranker?: RerankerFn
): AssembledContext {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Stage 1: Re-rank
  let processed = reranker ? reranker(query, chunks) : chunks;

  // Stage 2: Filter low-relevance
  processed = processed.filter((c) => c.score >= cfg.minRelevanceScore);

  // Stage 3: Deduplicate
  processed = deduplicate(processed, cfg.similarityThreshold);

  // Stage 4: Budget
  const { selected, totalTokens, dropped } = applyBudget(
    processed,
    cfg.tokenBudget
  );

  // Stage 5: Format
  const contextBlock = formatContext(selected);

  return {
    chunks: selected,
    totalTokens,
    droppedCount: dropped,
    contextBlock,
  };
}

// --- Simulated reranker ---

function simpleReranker(query: string, chunks: RetrievedChunk[]): RetrievedChunk[] {
  const queryTerms = new Set(query.toLowerCase().split(/\s+/));

  return chunks
    .map((chunk) => {
      const contentTerms = new Set(chunk.content.toLowerCase().split(/\s+/));
      const overlap =
        [...queryTerms].filter((t) => contentTerms.has(t)).length /
        Math.max(queryTerms.size, 1);
      const newScore = chunk.score * 0.4 + overlap * 0.6;
      return { ...chunk, score: newScore };
    })
    .sort((a, b) => b.score - a.score);
}

// --- Example usage ---

function main(): void {
  const rawChunks: RetrievedChunk[] = [
    {
      content:
        "Refunds are processed within 5-10 business days after approval. " +
        "Contact support with your order ID to initiate a refund request.",
      source: "billing-faq.md",
      section: "Refund Policy",
      score: 0.89,
    },
    {
      content:
        "Refund requests must include the order ID. Refunds take 5-10 " +
        "business days to process after the request is approved by support.",
      source: "support-guide.md",
      section: "Processing Refunds",
      score: 0.87,
    },
    {
      content:
        "Invoices are generated on the 1st of each month and sent to " +
        "the billing email on file.",
      source: "billing-faq.md",
      section: "Invoice Schedule",
      score: 0.62,
    },
    {
      content: "Our office hours are Monday through Friday, 9 AM to 5 PM EST.",
      source: "general-info.md",
      section: "Contact Us",
      score: 0.25,
    },
  ];

  const result = assembleContext(
    "How do I get a refund?",
    rawChunks,
    { tokenBudget: 2000, minRelevanceScore: 0.2 },
    simpleReranker
  );

  console.log(`Chunks used: ${result.chunks.length}`);
  console.log(`Tokens used: ${result.totalTokens}`);
  console.log(`Chunks dropped: ${result.droppedCount}`);
  console.log();
  console.log(result.contextBlock);
}

main();
```

## Trade-offs

| Pros | Cons |
|------|------|
| Dramatically improves answer quality over naive top-k retrieval | Re-ranking adds latency (50-200ms for cross-encoder models) |
| Token budget enforcement prevents context overflow | Requires tuning thresholds (similarity, min relevance, budget) |
| Source attribution enables verifiable, citable answers | Deduplication can occasionally remove legitimately similar but distinct content |
| Graceful degradation when retrieval quality is low | More complex pipeline means more failure modes to monitor |
| Works with any vector store or retrieval backend | Over-retrieval (fetching N to keep k) increases vector DB query cost |

## When to Use

- You are building a Q&A system, documentation chatbot, or search assistant backed by a knowledge base.
- Your vector store returns noisy or redundant results that degrade model performance.
- You need the model to cite its sources and users need to verify answers.
- Your context window budget is tight and you cannot afford to waste tokens on low-quality chunks.
- You have overlapping documents (e.g., an FAQ and a support guide covering the same topic).

## When NOT to Use

- Your retrieval corpus is tiny (under 50 documents) and a single top-k query consistently returns excellent results.
- You are not using retrieval at all -- this pattern is specifically for RAG pipelines.
- Latency is so critical that even 50ms of re-ranking overhead is unacceptable (e.g., autocomplete suggestions).
- Your chunks are already perfectly curated and deduplicated at indexing time.

## Related Patterns

- [Just-in-Time Retrieval](just-in-time-retrieval.md) -- Decides *when* to trigger retrieval; RAG Context Assembly decides *how* to process the results.
- [Semantic Tool Selection](semantic-tool-selection.md) -- Applies similar ranking and budgeting logic but to tool descriptions instead of knowledge chunks.
- [System Prompt Architecture](../construction/system-prompt-architecture.md) -- The retrieved context block is one section of a larger structured system prompt.

## Real-World Examples

- **Perplexity AI**: Retrieves web results, re-ranks them, and assembles a context block with inline source citations. The numbered citation format (`[1]`, `[2]`) is a direct implementation of the attribution formatting stage.
- **Amazon Q Business**: Retrieves enterprise documents from connected data sources, applies re-ranking and deduplication before injecting into the prompt, and returns answers with clickable source links.
- **Notion AI Q&A**: When answering questions about a user's workspace, retrieves relevant page chunks, deduplicates across similar pages, and budgets the context to fit within the model's window while preserving source page references.
- **LangChain / LlamaIndex**: Both frameworks provide built-in re-ranking, deduplication, and context assembly modules as part of their retrieval pipelines, making this pattern a first-class concept in the RAG ecosystem.
