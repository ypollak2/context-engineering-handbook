# Semantic Tool Selection

> Dynamically select only the relevant tool descriptions to include in the prompt based on semantic similarity to the current query.

## Problem

When an agent has access to many tools (10-100+), including all tool descriptions in the system prompt becomes expensive or impossible. Each tool schema with its name, description, and parameter definitions can consume 100-500 tokens. At 50 tools, that is 5,000-25,000 tokens of static overhead on every single turn -- even when the user is asking a simple question that needs zero or one tool. This wastes budget, dilutes the model's attention across irrelevant tools, and increases the likelihood of the model hallucinating tool calls to tools that sound vaguely relevant.

## Solution

Semantic Tool Selection treats the tool registry like a searchable knowledge base. Each tool's description (and optionally its parameter schema) is embedded into a vector space at registration time. When a user query arrives, the query is embedded and compared against the tool embeddings using cosine similarity. Only the top-k most relevant tools are injected into the prompt for that turn.

This transforms the tool menu from a static list into a dynamic, query-aware selection. The model sees a focused set of 3-7 tools that are actually relevant to the current task, rather than scrolling through dozens of irrelevant options. For tools that are always needed (e.g., a "final answer" tool or a "delegate" tool), the pattern supports pinning specific tools so they are always included regardless of similarity score.

## How It Works

```
Tool Registration (one-time)
+------------------+
| Tool 1: desc     |---> embed() ---> vector store
| Tool 2: desc     |---> embed() ---> vector store
| ...              |
| Tool N: desc     |---> embed() ---> vector store
+------------------+

Per-Turn Selection
+------------------+
| User Query       |---> embed() ---> query vector
+------------------+
         |
         v
+------------------+
| Cosine Similarity|  Compare query vector against all tool vectors
| Top-K Selection  |  Select k most similar tools
+------------------+
         |
         v
+------------------+
| Pinned Tools     |  Always include essential tools
| + Top-K Results  |  regardless of similarity
+------------------+
         |
         v
+------------------+
| Filtered Tool    |  Only these schemas go into
| Descriptions     |  the system prompt
+------------------+
```

## Implementation

### Python

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToolDefinition:
    """A tool available to the agent."""
    name: str
    description: str
    parameters: dict[str, Any]
    pinned: bool = False  # Always include in selection


@dataclass(frozen=True)
class ScoredTool:
    """A tool with its relevance score for the current query."""
    tool: ToolDefinition
    score: float


class SemanticToolSelector:
    """
    Selects relevant tools for each turn using embedding similarity.

    In production, replace the _embed method with calls to an embedding
    API (OpenAI, Cohere, or a local model like all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.3,
        embedding_dim: int = 384,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.embedding_dim = embedding_dim
        self._tools: list[ToolDefinition] = []
        self._embeddings: np.ndarray | None = None

    def register_tools(self, tools: list[ToolDefinition]) -> None:
        """Register tools and pre-compute their embeddings."""
        self._tools = tools
        descriptions = [
            f"{t.name}: {t.description}" for t in tools
        ]
        self._embeddings = self._embed_batch(descriptions)

    def select(self, query: str) -> list[ScoredTool]:
        """Select the most relevant tools for the given query."""
        if not self._tools or self._embeddings is None:
            return []

        query_embedding = self._embed(query)

        # Cosine similarity against all tool embeddings
        similarities = self._cosine_similarity(
            query_embedding, self._embeddings
        )

        # Build scored results
        scored: list[ScoredTool] = []
        pinned: list[ScoredTool] = []

        for i, tool in enumerate(self._tools):
            score = float(similarities[i])
            scored_tool = ScoredTool(tool=tool, score=score)

            if tool.pinned:
                pinned.append(scored_tool)
            else:
                scored.append(scored_tool)

        # Sort non-pinned by score descending, filter by min_score, take top_k
        selected = sorted(scored, key=lambda s: s.score, reverse=True)
        selected = [s for s in selected if s.score >= self.min_score]
        selected = selected[: self.top_k]

        # Pinned tools always included, placed first
        return pinned + selected

    def format_tool_block(self, tools: list[ScoredTool]) -> str:
        """Format selected tools into a prompt-ready block."""
        if not tools:
            return "<available_tools>\nNo tools available.\n</available_tools>"

        sections: list[str] = []
        for scored in tools:
            t = scored.tool
            params = ", ".join(
                f"{name}: {info.get('type', 'any')}"
                + (" (required)" if info.get("required") else "")
                for name, info in t.parameters.items()
            )
            sections.append(
                f"- **{t.name}**: {t.description}\n"
                f"  Parameters: {params or 'none'}"
            )

        body = "\n\n".join(sections)
        return f"<available_tools>\n{body}\n</available_tools>"

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        PRODUCTION: Replace with your embedding model.
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return np.array(response.data[0].embedding)
        """
        # Deterministic pseudo-embedding for demonstration
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.array([self._embed(t) for t in texts])

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity between a query vector and a matrix of vectors."""
        query_norm = query / np.linalg.norm(query)
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix_norms @ query_norm


# --- Example usage ---

def main() -> None:
    tools = [
        ToolDefinition(
            name="search_knowledge_base",
            description="Search the company knowledge base for articles, FAQs, and documentation",
            parameters={
                "query": {"type": "string", "required": True},
                "category": {"type": "string", "required": False},
            },
        ),
        ToolDefinition(
            name="create_ticket",
            description="Create a support ticket for issues that need human agent follow-up",
            parameters={
                "title": {"type": "string", "required": True},
                "description": {"type": "string", "required": True},
                "priority": {"type": "string", "required": False},
            },
        ),
        ToolDefinition(
            name="get_order_status",
            description="Look up the current status of a customer order by order ID",
            parameters={
                "order_id": {"type": "string", "required": True},
            },
        ),
        ToolDefinition(
            name="process_refund",
            description="Initiate a refund for a specific order or transaction",
            parameters={
                "order_id": {"type": "string", "required": True},
                "reason": {"type": "string", "required": True},
                "amount": {"type": "number", "required": False},
            },
        ),
        ToolDefinition(
            name="update_account_email",
            description="Update the email address on a customer account",
            parameters={
                "customer_id": {"type": "string", "required": True},
                "new_email": {"type": "string", "required": True},
            },
        ),
        ToolDefinition(
            name="schedule_callback",
            description="Schedule a phone callback from a human agent at a specific time",
            parameters={
                "phone": {"type": "string", "required": True},
                "preferred_time": {"type": "string", "required": True},
            },
        ),
        ToolDefinition(
            name="send_invoice",
            description="Send or resend an invoice to the customer's billing email",
            parameters={
                "invoice_id": {"type": "string", "required": True},
            },
        ),
        ToolDefinition(
            name="final_answer",
            description="Provide the final response to the user's question",
            parameters={
                "answer": {"type": "string", "required": True},
            },
            pinned=True,  # Always available
        ),
    ]

    selector = SemanticToolSelector(top_k=3, min_score=0.1)
    selector.register_tools(tools)

    query = "I need a refund for order #12345"
    selected = selector.select(query)

    print(f"Query: {query}")
    print(f"Selected {len(selected)} tools (from {len(tools)} total):\n")

    for scored in selected:
        pin_label = " [PINNED]" if scored.tool.pinned else ""
        print(f"  {scored.tool.name} (score: {scored.score:.3f}){pin_label}")

    print()
    print(selector.format_tool_block(selected))


if __name__ == "__main__":
    main()
```

### TypeScript

```typescript
// semantic-tool-selection.ts

interface ToolDefinition {
  readonly name: string;
  readonly description: string;
  readonly parameters: Record<string, { type: string; required?: boolean }>;
  readonly pinned?: boolean;
}

interface ScoredTool {
  readonly tool: ToolDefinition;
  readonly score: number;
}

interface SelectorConfig {
  readonly topK: number;
  readonly minScore: number;
  readonly embeddingDim: number;
}

const DEFAULT_SELECTOR_CONFIG: SelectorConfig = {
  topK: 5,
  minScore: 0.3,
  embeddingDim: 384,
};

class SemanticToolSelector {
  private config: SelectorConfig;
  private tools: ToolDefinition[] = [];
  private embeddings: number[][] = [];

  constructor(config: Partial<SelectorConfig> = {}) {
    this.config = { ...DEFAULT_SELECTOR_CONFIG, ...config };
  }

  registerTools(tools: ToolDefinition[]): void {
    this.tools = tools;
    this.embeddings = tools.map((t) =>
      this.embed(`${t.name}: ${t.description}`)
    );
  }

  select(query: string): ScoredTool[] {
    if (this.tools.length === 0) return [];

    const queryEmbedding = this.embed(query);
    const similarities = this.embeddings.map((toolEmb) =>
      this.cosineSimilarity(queryEmbedding, toolEmb)
    );

    const pinned: ScoredTool[] = [];
    const scored: ScoredTool[] = [];

    for (let i = 0; i < this.tools.length; i++) {
      const tool = this.tools[i];
      const scoredTool: ScoredTool = { tool, score: similarities[i] };

      if (tool.pinned) {
        pinned.push(scoredTool);
      } else {
        scored.push(scoredTool);
      }
    }

    const selected = scored
      .sort((a, b) => b.score - a.score)
      .filter((s) => s.score >= this.config.minScore)
      .slice(0, this.config.topK);

    return [...pinned, ...selected];
  }

  formatToolBlock(tools: ScoredTool[]): string {
    if (tools.length === 0) {
      return "<available_tools>\nNo tools available.\n</available_tools>";
    }

    const sections = tools.map((scored) => {
      const t = scored.tool;
      const params = Object.entries(t.parameters)
        .map(
          ([name, info]) =>
            `${name}: ${info.type}${info.required ? " (required)" : ""}`
        )
        .join(", ");

      return `- **${t.name}**: ${t.description}\n  Parameters: ${params || "none"}`;
    });

    return `<available_tools>\n${sections.join("\n\n")}\n</available_tools>`;
  }

  /**
   * Pseudo-embedding for demonstration.
   *
   * PRODUCTION: Replace with a real embedding API call:
   *
   *   import OpenAI from "openai";
   *   const client = new OpenAI();
   *   const response = await client.embeddings.create({
   *     model: "text-embedding-3-small",
   *     input: text,
   *   });
   *   return response.data[0].embedding;
   */
  private embed(text: string): number[] {
    // Deterministic pseudo-embedding using a simple hash-based seed
    let seed = 0;
    for (let i = 0; i < text.length; i++) {
      seed = ((seed << 5) - seed + text.charCodeAt(i)) | 0;
    }

    const vec: number[] = [];
    for (let i = 0; i < this.config.embeddingDim; i++) {
      // Simple LCG pseudo-random number generator
      seed = (seed * 1664525 + 1013904223) | 0;
      vec.push((seed / 2147483647) * 2 - 1);
    }

    // Normalize
    const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return vec.map((v) => v / norm);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// --- Example usage ---

function main(): void {
  const tools: ToolDefinition[] = [
    {
      name: "search_knowledge_base",
      description:
        "Search the company knowledge base for articles, FAQs, and documentation",
      parameters: {
        query: { type: "string", required: true },
        category: { type: "string" },
      },
    },
    {
      name: "create_ticket",
      description:
        "Create a support ticket for issues that need human agent follow-up",
      parameters: {
        title: { type: "string", required: true },
        description: { type: "string", required: true },
        priority: { type: "string" },
      },
    },
    {
      name: "get_order_status",
      description:
        "Look up the current status of a customer order by order ID",
      parameters: {
        order_id: { type: "string", required: true },
      },
    },
    {
      name: "process_refund",
      description: "Initiate a refund for a specific order or transaction",
      parameters: {
        order_id: { type: "string", required: true },
        reason: { type: "string", required: true },
        amount: { type: "number" },
      },
    },
    {
      name: "update_account_email",
      description: "Update the email address on a customer account",
      parameters: {
        customer_id: { type: "string", required: true },
        new_email: { type: "string", required: true },
      },
    },
    {
      name: "schedule_callback",
      description:
        "Schedule a phone callback from a human agent at a specific time",
      parameters: {
        phone: { type: "string", required: true },
        preferred_time: { type: "string", required: true },
      },
    },
    {
      name: "send_invoice",
      description:
        "Send or resend an invoice to the customer's billing email",
      parameters: {
        invoice_id: { type: "string", required: true },
      },
    },
    {
      name: "final_answer",
      description: "Provide the final response to the user's question",
      parameters: {
        answer: { type: "string", required: true },
      },
      pinned: true,
    },
  ];

  const selector = new SemanticToolSelector({ topK: 3, minScore: 0.1 });
  selector.registerTools(tools);

  const query = "I need a refund for order #12345";
  const selected = selector.select(query);

  console.log(`Query: ${query}`);
  console.log(`Selected ${selected.length} tools (from ${tools.length} total):\n`);

  for (const scored of selected) {
    const pinLabel = scored.tool.pinned ? " [PINNED]" : "";
    console.log(
      `  ${scored.tool.name} (score: ${scored.score.toFixed(3)})${pinLabel}`
    );
  }

  console.log();
  console.log(selector.formatToolBlock(selected));
}

main();
```

## Trade-offs

| Pros | Cons |
|------|------|
| Enables agents with 50-100+ tools without blowing the context budget | Embedding quality directly affects tool selection accuracy |
| Reduces hallucinated tool calls by removing irrelevant options | Adds embedding computation latency per turn (~10-50ms) |
| Scales linearly -- adding new tools does not increase per-turn token cost | Requires maintaining a tool embedding index alongside the tool registry |
| Pinning mechanism ensures critical tools are never dropped | Semantic similarity can miss tools with indirect relevance (e.g., "change my password" might not match "update_security_settings") |
| Works with any embedding model and can be improved independently | Cold-start: new tools need embedding before they can be selected |

## When to Use

- Your agent has more than 10 tools and including all descriptions consumes significant token budget.
- You are building an MCP client that connects to multiple servers, each exposing their own tools.
- Your enterprise agent integrates with 50+ internal systems (CRM, ticketing, billing, HR, etc.).
- You notice the model making confused or hallucinated tool calls because it is overwhelmed by options.
- Tool usage patterns vary widely per query -- most queries need only 2-3 tools from the full catalog.

## When NOT to Use

- You have fewer than 10 tools and their total description size fits comfortably in the context window.
- Every tool is potentially relevant on every turn (e.g., a calculator agent where all operations matter).
- Your tools are so similar in description that semantic similarity cannot reliably distinguish them -- use explicit routing instead.
- You are in an environment where tool descriptions are enforced by the API (e.g., OpenAI function calling with a fixed tool list) and cannot be dynamically filtered.

## Related Patterns

- [Just-in-Time Retrieval](just-in-time-retrieval.md) -- Semantic Tool Selection is a specialized form of JIT retrieval applied to tool capabilities rather than knowledge.
- [RAG Context Assembly](rag-context-assembly.md) -- The ranking and budgeting stages share the same principles; you could use RAG Context Assembly to process tool descriptions as "chunks."
- [Progressive Disclosure](../construction/progressive-disclosure.md) -- An alternative approach where tool descriptions are revealed in stages based on conversation progression rather than semantic similarity.

## Real-World Examples

- **MCP (Model Context Protocol) ecosystems**: Agents that connect to multiple MCP servers can accumulate hundreds of tools. Production MCP clients like Claude Desktop and Cursor use tool selection strategies to avoid overwhelming the model with every available tool on every turn.
- **LangChain / CrewAI tool routing**: Both frameworks support dynamic tool selection based on task descriptions, allowing agents to operate over large tool catalogs without including every tool in the prompt.
- **GPT-4 with large plugin sets**: When OpenAI supported ChatGPT Plugins, the system used a selection layer to choose which plugin descriptions to include based on the user's message, since including all plugin schemas simultaneously was impractical.
- **Enterprise copilots (ServiceNow, Salesforce)**: Internal copilots that integrate with dozens of enterprise systems use semantic matching to select which integration tools to surface based on the employee's current question, rather than listing every available integration.
