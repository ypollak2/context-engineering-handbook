"""
RAG Context Assembly -- LangChain Integration

Implements a custom retriever chain with re-ranking, deduplication, and token
budgeting using LangChain's LCEL. The chain sits between a vector store
retriever and the LLM, transforming raw retrieval results into a high-quality,
budget-constrained context block.

Pattern: https://github.com/context-engineering-handbook/patterns/retrieval/rag-context-assembly.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda


# ---------------------------------------------------------------------------
# Assembly configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyConfig:
    """Configuration for the RAG context assembly pipeline."""

    token_budget: int = 3000
    similarity_threshold: float = 0.85
    min_relevance_score: float = 0.3
    over_retrieve_factor: int = 4  # Retrieve N * this, keep top N


@dataclass(frozen=True)
class AssembledContext:
    """The final assembled context block ready for prompt injection."""

    documents: tuple[Document, ...]
    total_tokens: int
    dropped_count: int
    context_block: str


# ---------------------------------------------------------------------------
# Pipeline stages as composable functions
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English."""
    return len(text) // 4


def _shingle(text: str, n: int = 3) -> set[str]:
    """Create word n-gram shingles for overlap detection."""
    words = text.lower().split()
    if len(words) < n:
        return {text.lower()}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two shingle sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / max(union, 1)


def rerank_documents(
    query: str, docs: list[Document]
) -> list[Document]:
    """Re-rank documents by term overlap with the query.

    In production, replace this with a cross-encoder model:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)

    Or use LangChain's built-in reranker integrations:
        from langchain_community.document_compressors import CrossEncoderReranker
    """
    query_terms = set(query.lower().split())
    scored: list[tuple[Document, float]] = []

    for doc in docs:
        content_terms = set(doc.page_content.lower().split())
        term_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
        original_score = doc.metadata.get("score", 0.5)
        new_score = original_score * 0.4 + term_overlap * 0.6

        new_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "score": new_score},
        )
        scored.append((new_doc, new_score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in scored]


def deduplicate_documents(
    docs: list[Document], threshold: float = 0.85
) -> list[Document]:
    """Remove near-duplicate documents, keeping the higher-scored version."""
    if not docs:
        return []

    # Sort by score descending (highest first)
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get("score", 0.0),
        reverse=True,
    )

    kept: list[Document] = []
    seen_shingles: list[set[str]] = []

    for doc in sorted_docs:
        doc_shingles = _shingle(doc.page_content)
        is_duplicate = any(
            _jaccard_similarity(doc_shingles, existing) >= threshold
            for existing in seen_shingles
        )

        if not is_duplicate:
            kept.append(doc)
            seen_shingles.append(doc_shingles)

    return kept


def apply_token_budget(
    docs: list[Document], budget: int
) -> tuple[list[Document], int, int]:
    """Greedily fill the token budget from highest-ranked documents."""
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get("score", 0.0),
        reverse=True,
    )

    selected: list[Document] = []
    total_tokens = 0
    dropped = 0

    for doc in sorted_docs:
        chunk_tokens = _estimate_tokens(doc.page_content)
        chunk_cost = chunk_tokens + 20  # 20 tokens for attribution header

        if total_tokens + chunk_cost > budget:
            dropped += 1
            continue

        selected.append(doc)
        total_tokens += chunk_cost

    return selected, total_tokens, dropped


def format_context_block(docs: list[Document]) -> str:
    """Format documents into a context block with source attribution."""
    if not docs:
        return (
            "<retrieved_context>\n"
            "No relevant information found.\n"
            "</retrieved_context>"
        )

    sections: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        header = f"[Source {i}: {source}"
        if section:
            header += f" > {section}"
        header += "]"
        sections.append(f"{header}\n{doc.page_content}")

    body = "\n\n---\n\n".join(sections)
    return f"<retrieved_context>\n{body}\n</retrieved_context>"


# ---------------------------------------------------------------------------
# LangChain Retriever: AssemblingRetriever
#
# Wraps a base retriever and applies the full assembly pipeline. Drop this
# into any LCEL chain that expects a Retriever.
# ---------------------------------------------------------------------------


class AssemblingRetriever(BaseRetriever):
    """A LangChain retriever that applies the full RAG assembly pipeline.

    Wraps a base retriever and applies re-ranking, deduplication, and
    token budgeting to the raw retrieval results.

    Usage:
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(docs, embeddings)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

        assembler = AssemblingRetriever(
            base_retriever=base_retriever,
            config=AssemblyConfig(token_budget=2000),
        )

        # Use in an LCEL chain
        chain = assembler | format_docs | llm
    """

    base_retriever: BaseRetriever
    config: AssemblyConfig = AssemblyConfig()

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Run the full assembly pipeline."""
        # Stage 1: Over-retrieve from base retriever
        raw_docs = self.base_retriever.invoke(query)

        # Stage 2: Re-rank
        reranked = rerank_documents(query, raw_docs)

        # Stage 3: Filter low-relevance
        filtered = [
            d
            for d in reranked
            if d.metadata.get("score", 0.0) >= self.config.min_relevance_score
        ]

        # Stage 4: Deduplicate
        deduped = deduplicate_documents(
            filtered, self.config.similarity_threshold
        )

        # Stage 5: Apply token budget
        selected, _, _ = apply_token_budget(deduped, self.config.token_budget)

        return selected


# ---------------------------------------------------------------------------
# LCEL chain builder
# ---------------------------------------------------------------------------


def build_rag_assembly_chain(
    config: AssemblyConfig | None = None,
) -> Runnable[tuple[str, list[Document]], AssembledContext]:
    """Build an LCEL chain that takes (query, raw_documents) and returns
    an AssembledContext.

    This is useful when you want to control retrieval separately and only
    apply the assembly pipeline.

    Usage:
        chain = build_rag_assembly_chain(AssemblyConfig(token_budget=2000))
        result = chain.invoke(("How do I get a refund?", raw_docs))
        print(result.context_block)
    """
    cfg = config or AssemblyConfig()

    def assemble(input_pair: tuple[str, list[Document]]) -> AssembledContext:
        query, raw_docs = input_pair

        # Pipeline stages
        reranked = rerank_documents(query, raw_docs)
        filtered = [
            d
            for d in reranked
            if d.metadata.get("score", 0.0) >= cfg.min_relevance_score
        ]
        deduped = deduplicate_documents(filtered, cfg.similarity_threshold)
        selected, total_tokens, dropped = apply_token_budget(
            deduped, cfg.token_budget
        )
        context_block = format_context_block(selected)

        return AssembledContext(
            documents=tuple(selected),
            total_tokens=total_tokens,
            dropped_count=dropped,
            context_block=context_block,
        )

    return RunnableLambda(assemble)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate RAG Context Assembly without requiring API keys."""

    # Simulated retrieval results
    raw_docs = [
        Document(
            page_content=(
                "Refunds are processed within 5-10 business days after approval. "
                "Contact support with your order ID to initiate a refund request."
            ),
            metadata={
                "source": "billing-faq.md",
                "section": "Refund Policy",
                "score": 0.89,
            },
        ),
        Document(
            page_content=(
                "Refund requests must include the order ID. Refunds take 5-10 "
                "business days to process after the request is approved by support."
            ),
            metadata={
                "source": "support-guide.md",
                "section": "Processing Refunds",
                "score": 0.87,
            },
        ),
        Document(
            page_content=(
                "Invoices are generated on the 1st of each month and sent to "
                "the billing email on file. Past invoices are available in the dashboard."
            ),
            metadata={
                "source": "billing-faq.md",
                "section": "Invoice Schedule",
                "score": 0.62,
            },
        ),
        Document(
            page_content=(
                "Our office hours are Monday through Friday, 9 AM to 5 PM EST."
            ),
            metadata={
                "source": "general-info.md",
                "section": "Contact Us",
                "score": 0.25,
            },
        ),
    ]

    # Build and invoke the assembly chain
    chain = build_rag_assembly_chain(
        AssemblyConfig(
            token_budget=2000,
            similarity_threshold=0.85,
            min_relevance_score=0.2,
        )
    )

    result = chain.invoke(("How do I get a refund?", raw_docs))

    print(f"Chunks used: {len(result.documents)}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Chunks dropped: {result.dropped_count}")
    print()
    print(result.context_block)


if __name__ == "__main__":
    main()
