"""
RAG Context Assembly -- LlamaIndex Integration

Implements the RAG Context Assembly pattern as a pipeline of LlamaIndex
NodePostprocessor objects. Each stage (re-ranking, deduplication, token
budgeting, attribution formatting) is a separate postprocessor that plugs
into any LlamaIndex retriever pipeline.

Pattern: https://github.com/context-engineering-handbook/patterns/retrieval/rag-context-assembly.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


# ---------------------------------------------------------------------------
# Stage 1: Re-ranking postprocessor
# ---------------------------------------------------------------------------


class TermOverlapReranker(BaseNodePostprocessor):
    """Re-ranks nodes by term overlap with the query.

    In production, replace with:
        from llama_index.postprocessor.cohere_rerank import CohereRerank
        reranker = CohereRerank(top_n=10)

    Or use a cross-encoder:
        from llama_index.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3", top_n=10
        )
    """

    original_weight: float = 0.4
    overlap_weight: float = 0.6

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            return nodes

        query_terms = set(query_bundle.query_str.lower().split())

        reranked: list[NodeWithScore] = []
        for node_with_score in nodes:
            content = node_with_score.node.get_content()
            content_terms = set(content.lower().split())
            term_overlap = len(query_terms & content_terms) / max(
                len(query_terms), 1
            )

            original_score = node_with_score.score or 0.5
            new_score = (
                original_score * self.original_weight
                + term_overlap * self.overlap_weight
            )

            reranked.append(
                NodeWithScore(node=node_with_score.node, score=new_score)
            )

        reranked.sort(key=lambda n: n.score or 0, reverse=True)
        return reranked


# ---------------------------------------------------------------------------
# Stage 2: Deduplication postprocessor
# ---------------------------------------------------------------------------


class DeduplicationPostprocessor(BaseNodePostprocessor):
    """Removes near-duplicate nodes using Jaccard similarity on word shingles.

    Keeps the higher-scored version when duplicates are detected.
    """

    similarity_threshold: float = 0.85
    shingle_size: int = 3

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        # Sort by score descending
        sorted_nodes = sorted(
            nodes, key=lambda n: n.score or 0, reverse=True
        )

        kept: list[NodeWithScore] = []
        seen_shingles: list[set[str]] = []

        for node_with_score in sorted_nodes:
            content = node_with_score.node.get_content()
            node_shingles = self._shingle(content)

            is_duplicate = any(
                self._jaccard(node_shingles, existing)
                >= self.similarity_threshold
                for existing in seen_shingles
            )

            if not is_duplicate:
                kept.append(node_with_score)
                seen_shingles.append(node_shingles)

        return kept

    def _shingle(self, text: str) -> set[str]:
        words = text.lower().split()
        n = self.shingle_size
        if len(words) < n:
            return {text.lower()}
        return {
            " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
        }

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)


# ---------------------------------------------------------------------------
# Stage 3: Token budget postprocessor
# ---------------------------------------------------------------------------


class TokenBudgetPostprocessor(BaseNodePostprocessor):
    """Enforces a strict token budget by greedily selecting top-ranked nodes.

    Nodes are added from highest score to lowest until the budget is exhausted.
    The attribution_overhead_per_node accounts for the source header tokens.
    """

    token_budget: int = 3000
    attribution_overhead_per_node: int = 20

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        sorted_nodes = sorted(
            nodes, key=lambda n: n.score or 0, reverse=True
        )

        selected: list[NodeWithScore] = []
        total_tokens = 0

        for node_with_score in sorted_nodes:
            content = node_with_score.node.get_content()
            chunk_tokens = len(content) // 4  # Rough estimate
            chunk_cost = chunk_tokens + self.attribution_overhead_per_node

            if total_tokens + chunk_cost > self.token_budget:
                continue

            selected.append(node_with_score)
            total_tokens += chunk_cost

        return selected


# ---------------------------------------------------------------------------
# Stage 4: Attribution formatter
# ---------------------------------------------------------------------------


class AttributionFormatter(BaseNodePostprocessor):
    """Adds source attribution metadata to each node for citation.

    After this postprocessor, each node's metadata includes a 'citation'
    field that can be used in the response template.
    """

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        result: list[NodeWithScore] = []

        for i, node_with_score in enumerate(nodes, 1):
            node = node_with_score.node
            source = node.metadata.get("source", "unknown")
            section = node.metadata.get("section", "")

            citation = f"[Source {i}: {source}"
            if section:
                citation += f" > {section}"
            citation += "]"

            # Create a new node with the citation in metadata (immutable pattern)
            new_metadata = {**node.metadata, "citation": citation, "source_index": i}
            new_node = TextNode(
                text=node.get_content(),
                metadata=new_metadata,
                id_=node.node_id,
            )
            result.append(NodeWithScore(node=new_node, score=node_with_score.score))

        return result


# ---------------------------------------------------------------------------
# Assembly pipeline builder
# ---------------------------------------------------------------------------


def build_assembly_pipeline(
    token_budget: int = 3000,
    similarity_threshold: float = 0.85,
    min_relevance_score: float = 0.3,
) -> list[BaseNodePostprocessor]:
    """Build the full assembly postprocessor pipeline.

    Usage with a LlamaIndex query engine:
        from llama_index.core import VectorStoreIndex

        index = VectorStoreIndex.from_documents(documents)
        pipeline = build_assembly_pipeline(token_budget=2000)

        query_engine = index.as_query_engine(
            node_postprocessors=pipeline,
            similarity_top_k=20,  # Over-retrieve
        )
        response = query_engine.query("How do I get a refund?")
    """
    return [
        TermOverlapReranker(
            original_weight=0.4,
            overlap_weight=0.6,
        ),
        DeduplicationPostprocessor(
            similarity_threshold=similarity_threshold,
        ),
        TokenBudgetPostprocessor(
            token_budget=token_budget,
        ),
        AttributionFormatter(),
    ]


def format_context_block(nodes: list[NodeWithScore]) -> str:
    """Format postprocessed nodes into a context block string."""
    if not nodes:
        return (
            "<retrieved_context>\n"
            "No relevant information found.\n"
            "</retrieved_context>"
        )

    sections: list[str] = []
    for node_with_score in nodes:
        citation = node_with_score.node.metadata.get("citation", "[Source]")
        content = node_with_score.node.get_content()
        sections.append(f"{citation}\n{content}")

    body = "\n\n---\n\n".join(sections)
    return f"<retrieved_context>\n{body}\n</retrieved_context>"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate the RAG assembly pipeline without requiring API keys."""

    # Create simulated retrieval results as LlamaIndex nodes
    raw_nodes = [
        NodeWithScore(
            node=TextNode(
                text=(
                    "Refunds are processed within 5-10 business days after approval. "
                    "Contact support with your order ID to initiate a refund request."
                ),
                metadata={"source": "billing-faq.md", "section": "Refund Policy"},
            ),
            score=0.89,
        ),
        NodeWithScore(
            node=TextNode(
                text=(
                    "Refund requests must include the order ID. Refunds take 5-10 "
                    "business days to process after the request is approved by support."
                ),
                metadata={"source": "support-guide.md", "section": "Processing Refunds"},
            ),
            score=0.87,
        ),
        NodeWithScore(
            node=TextNode(
                text=(
                    "Invoices are generated on the 1st of each month and sent to "
                    "the billing email on file."
                ),
                metadata={"source": "billing-faq.md", "section": "Invoice Schedule"},
            ),
            score=0.62,
        ),
        NodeWithScore(
            node=TextNode(
                text="Our office hours are Monday through Friday, 9 AM to 5 PM EST.",
                metadata={"source": "general-info.md", "section": "Contact Us"},
            ),
            score=0.25,
        ),
    ]

    # Build and run the assembly pipeline
    pipeline = build_assembly_pipeline(
        token_budget=2000,
        similarity_threshold=0.85,
        min_relevance_score=0.2,
    )

    query = QueryBundle(query_str="How do I get a refund?")
    processed = raw_nodes

    for postprocessor in pipeline:
        processed = postprocessor.postprocess_nodes(
            processed, query_bundle=query
        )

    print(f"Nodes used: {len(processed)}")
    print(
        f"Total tokens: {sum(len(n.node.get_content()) // 4 for n in processed)}"
    )
    print()
    print(format_context_block(processed))


if __name__ == "__main__":
    main()
