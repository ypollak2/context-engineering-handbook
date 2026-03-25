# Few-Shot Curation

> Dynamically select the most relevant examples for each prompt instead of using a static set.

## Problem

Static few-shot examples -- hardcoded into a prompt template -- degrade over time and across use cases. An example that perfectly illustrates email classification is useless when the user asks about contract analysis. Worse, irrelevant examples actively confuse the model by establishing the wrong pattern. But maintaining separate prompt templates for every possible task is not scalable. You need examples that adapt to the input automatically.

## Solution

Maintain a library of labeled examples and select the best ones at runtime based on the current input. Selection can use embedding similarity (find examples whose inputs are semantically closest to the current query), task classification (match examples by category), recency (prefer recent examples that reflect current patterns), or a combination. The selected examples are injected into the prompt's few-shot section, replacing static examples with contextually relevant ones.

This decouples the example lifecycle from the prompt template lifecycle. The prompt template defines *where* examples go and *how many*. The curation system decides *which* examples fill those slots. New examples can be added to the library without modifying the prompt. Poorly performing examples can be retired based on outcome tracking.

## How It Works

```
                        +-------------------+
                        |   Example Library  |
                        |  (100s-1000s of   |
                        |   labeled pairs)   |
                        +---------+---------+
                                  |
                                  v
+------------+     +--------------------------+     +----------------+
|   Current   |---->|   Selector               |---->|  Top-K         |
|   Input     |     |                          |     |  Examples      |
+------------+     |  1. Embed current input   |     +-------+--------+
                   |  2. Score against library  |             |
                   |  3. Diversify selections   |             v
                   |  4. Apply token budget     |     +----------------+
                   +--------------------------+     |  Prompt with    |
                                                     |  curated shots  |
                                                     +----------------+

Selection strategies:
+----------------------------------------------------------+
| Embedding Similarity  | Best for semantic matching       |
| Task Classification   | Best for categorical routing     |
| Recency Weighting     | Best for evolving domains        |
| Outcome Scoring       | Best for optimizing quality      |
| Diversity Sampling    | Best for covering edge cases     |
+----------------------------------------------------------+
```

## Implementation

### Python

```python
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class SelectionStrategy(Enum):
    SIMILARITY = auto()
    DIVERSITY = auto()
    HYBRID = auto()


@dataclass(frozen=True)
class Example:
    """A single few-shot example with metadata."""
    id: str
    input_text: str
    output_text: str
    category: str
    embedding: tuple[float, ...] | None = None
    quality_score: float = 1.0  # 0.0 - 1.0, based on outcome tracking
    token_count: int = 0

    @staticmethod
    def create(
        input_text: str,
        output_text: str,
        category: str,
        embedding: list[float] | None = None,
        quality_score: float = 1.0,
    ) -> "Example":
        """Factory that auto-generates ID and estimates tokens."""
        content_hash = hashlib.sha256(
            f"{input_text}{output_text}".encode()
        ).hexdigest()[:12]
        token_est = (len(input_text) + len(output_text)) // 4
        return Example(
            id=content_hash,
            input_text=input_text,
            output_text=output_text,
            category=category,
            embedding=tuple(embedding) if embedding else None,
            quality_score=quality_score,
            token_count=token_est,
        )


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


@dataclass
class FewShotSelector:
    """Selects the most relevant examples from a library for a given input.

    Supports embedding-based similarity, diversity sampling, and hybrid
    approaches. Respects token budgets and quality thresholds.
    """
    examples: list[Example] = field(default_factory=list)
    max_examples: int = 3
    max_tokens: int = 2000
    min_quality: float = 0.5
    strategy: SelectionStrategy = SelectionStrategy.HYBRID
    diversity_weight: float = 0.3  # 0.0 = pure similarity, 1.0 = pure diversity

    # In production, replace with a real embedding function (OpenAI, Cohere, etc.)
    _embed_fn: object = None

    def add_example(self, example: Example) -> None:
        """Add an example to the library."""
        self.examples.append(example)

    def add_examples(self, examples: list[Example]) -> None:
        """Add multiple examples to the library."""
        self.examples.extend(examples)

    def select(
        self,
        query: str,
        query_embedding: list[float],
        category_filter: str | None = None,
        k: int | None = None,
    ) -> list[Example]:
        """Select the top-k most relevant examples for a query.

        Args:
            query: The current user input (used for logging/debugging).
            query_embedding: Pre-computed embedding of the query.
            category_filter: Optional category to restrict selection.
            k: Number of examples to return (defaults to self.max_examples).

        Returns:
            Ordered list of selected examples, best first.
        """
        k = k or self.max_examples
        query_emb = tuple(query_embedding)

        # Filter by category and quality
        candidates = [
            ex for ex in self.examples
            if ex.quality_score >= self.min_quality
            and ex.embedding is not None
            and (category_filter is None or ex.category == category_filter)
        ]

        if not candidates:
            return []

        if self.strategy == SelectionStrategy.SIMILARITY:
            return self._select_by_similarity(candidates, query_emb, k)
        elif self.strategy == SelectionStrategy.DIVERSITY:
            return self._select_by_diversity(candidates, query_emb, k)
        else:
            return self._select_hybrid(candidates, query_emb, k)

    def format_examples(self, examples: list[Example]) -> str:
        """Format selected examples into a prompt-ready string."""
        parts: list[str] = []
        for i, ex in enumerate(examples, 1):
            parts.append(
                f"Example {i}:\n"
                f"Input: {ex.input_text}\n"
                f"Output: {ex.output_text}"
            )
        return "\n\n".join(parts)

    def _select_by_similarity(
        self,
        candidates: list[Example],
        query_emb: tuple[float, ...],
        k: int,
    ) -> list[Example]:
        """Pure similarity-based selection."""
        scored = [
            (ex, cosine_similarity(query_emb, ex.embedding) * ex.quality_score)
            for ex in candidates
            if ex.embedding is not None
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self._apply_token_budget([ex for ex, _ in scored[:k]])

    def _select_by_diversity(
        self,
        candidates: list[Example],
        query_emb: tuple[float, ...],
        k: int,
    ) -> list[Example]:
        """Maximal Marginal Relevance (MMR) for diversity."""
        return self._mmr(candidates, query_emb, k, lambda_param=0.3)

    def _select_hybrid(
        self,
        candidates: list[Example],
        query_emb: tuple[float, ...],
        k: int,
    ) -> list[Example]:
        """Hybrid: MMR with tunable diversity weight."""
        lambda_param = 1.0 - self.diversity_weight
        return self._mmr(candidates, query_emb, k, lambda_param)

    def _mmr(
        self,
        candidates: list[Example],
        query_emb: tuple[float, ...],
        k: int,
        lambda_param: float,
    ) -> list[Example]:
        """Maximal Marginal Relevance selection.

        Balances relevance to the query with diversity among selected examples.
        lambda_param=1.0 is pure relevance, lambda_param=0.0 is pure diversity.
        """
        selected: list[Example] = []
        remaining = list(candidates)

        for _ in range(min(k, len(candidates))):
            best_score = -float("inf")
            best_example = None

            for ex in remaining:
                if ex.embedding is None:
                    continue
                relevance = cosine_similarity(query_emb, ex.embedding) * ex.quality_score

                # Max similarity to any already-selected example
                if selected:
                    max_sim = max(
                        cosine_similarity(ex.embedding, sel.embedding)
                        for sel in selected
                        if sel.embedding is not None
                    )
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_example = ex

            if best_example is None:
                break
            selected.append(best_example)
            remaining.remove(best_example)

        return self._apply_token_budget(selected)

    def _apply_token_budget(self, examples: list[Example]) -> list[Example]:
        """Trim examples to fit within token budget."""
        result: list[Example] = []
        total_tokens = 0
        for ex in examples:
            if total_tokens + ex.token_count > self.max_tokens:
                break
            result.append(ex)
            total_tokens += ex.token_count
        return result


# --- Usage ---

# Create an example library (embeddings would come from an API in production)
library = [
    Example.create(
        input_text="Classify: 'Your account has been suspended, click here to verify'",
        output_text='{"category": "phishing", "confidence": 0.95}',
        category="email_classification",
        embedding=[0.8, 0.1, 0.05, 0.02, 0.03],
        quality_score=0.95,
    ),
    Example.create(
        input_text="Classify: 'Meeting tomorrow at 3pm to discuss Q4 roadmap'",
        output_text='{"category": "meeting", "confidence": 0.92}',
        category="email_classification",
        embedding=[0.1, 0.8, 0.05, 0.03, 0.02],
        quality_score=0.90,
    ),
    Example.create(
        input_text="Classify: 'Invoice #4521 is attached for your review'",
        output_text='{"category": "finance", "confidence": 0.88}',
        category="email_classification",
        embedding=[0.05, 0.1, 0.8, 0.03, 0.02],
        quality_score=0.85,
    ),
    Example.create(
        input_text="Classify: 'Congratulations! You've won a free iPhone'",
        output_text='{"category": "spam", "confidence": 0.97}',
        category="email_classification",
        embedding=[0.75, 0.05, 0.1, 0.05, 0.05],
        quality_score=0.92,
    ),
    Example.create(
        input_text="Classify: 'The deployment pipeline failed on staging'",
        output_text='{"category": "engineering", "confidence": 0.91}',
        category="email_classification",
        embedding=[0.02, 0.15, 0.03, 0.7, 0.1],
        quality_score=0.88,
    ),
]

selector = FewShotSelector(
    max_examples=3,
    max_tokens=1000,
    strategy=SelectionStrategy.HYBRID,
    diversity_weight=0.3,
)
selector.add_examples(library)

# At runtime: select examples for a new query
query = "Classify: 'Urgent: verify your bank details immediately'"
query_embedding = [0.82, 0.08, 0.04, 0.03, 0.03]  # Similar to phishing/spam

selected = selector.select(
    query=query,
    query_embedding=query_embedding,
    category_filter="email_classification",
)

prompt_examples = selector.format_examples(selected)
print(f"Selected {len(selected)} examples:\n")
print(prompt_examples)
```

### TypeScript

```typescript
interface Example {
  readonly id: string;
  readonly inputText: string;
  readonly outputText: string;
  readonly category: string;
  readonly embedding: number[] | null;
  readonly qualityScore: number;
  readonly tokenCount: number;
}

type SelectionStrategy = "similarity" | "diversity" | "hybrid";

interface SelectorOptions {
  readonly maxExamples: number;
  readonly maxTokens: number;
  readonly minQuality: number;
  readonly strategy: SelectionStrategy;
  readonly diversityWeight: number;
}

const DEFAULT_SELECTOR_OPTIONS: SelectorOptions = {
  maxExamples: 3,
  maxTokens: 2000,
  minQuality: 0.5,
  strategy: "hybrid",
  diversityWeight: 0.3,
};

function createExample(params: {
  inputText: string;
  outputText: string;
  category: string;
  embedding?: number[];
  qualityScore?: number;
}): Example {
  const content = `${params.inputText}${params.outputText}`;
  // Simple hash for ID generation
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    hash = (hash << 5) - hash + content.charCodeAt(i);
    hash |= 0;
  }
  return {
    id: Math.abs(hash).toString(16).slice(0, 12),
    inputText: params.inputText,
    outputText: params.outputText,
    category: params.category,
    embedding: params.embedding ?? null,
    qualityScore: params.qualityScore ?? 1.0,
    tokenCount: Math.ceil((params.inputText.length + params.outputText.length) / 4),
  };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

class FewShotSelector {
  private readonly examples: Example[] = [];
  private readonly options: SelectorOptions;

  constructor(options: Partial<SelectorOptions> = {}) {
    this.options = { ...DEFAULT_SELECTOR_OPTIONS, ...options };
  }

  addExample(example: Example): void {
    this.examples.push(example);
  }

  addExamples(examples: Example[]): void {
    this.examples.push(...examples);
  }

  select(params: {
    query: string;
    queryEmbedding: number[];
    categoryFilter?: string;
    k?: number;
  }): Example[] {
    const k = params.k ?? this.options.maxExamples;

    const candidates = this.examples.filter(
      (ex) =>
        ex.qualityScore >= this.options.minQuality &&
        ex.embedding !== null &&
        (params.categoryFilter === undefined ||
          ex.category === params.categoryFilter)
    );

    if (candidates.length === 0) return [];

    switch (this.options.strategy) {
      case "similarity":
        return this.selectBySimilarity(candidates, params.queryEmbedding, k);
      case "diversity":
        return this.mmr(candidates, params.queryEmbedding, k, 0.3);
      case "hybrid":
        return this.mmr(
          candidates,
          params.queryEmbedding,
          k,
          1.0 - this.options.diversityWeight
        );
    }
  }

  formatExamples(examples: Example[]): string {
    return examples
      .map(
        (ex, i) =>
          `Example ${i + 1}:\nInput: ${ex.inputText}\nOutput: ${ex.outputText}`
      )
      .join("\n\n");
  }

  private selectBySimilarity(
    candidates: Example[],
    queryEmb: number[],
    k: number
  ): Example[] {
    const scored = candidates
      .filter((ex) => ex.embedding !== null)
      .map((ex) => ({
        example: ex,
        score:
          cosineSimilarity(queryEmb, ex.embedding!) * ex.qualityScore,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);

    return this.applyTokenBudget(scored.map((s) => s.example));
  }

  /**
   * Maximal Marginal Relevance: balances relevance with diversity.
   * lambdaParam=1.0 is pure relevance, 0.0 is pure diversity.
   */
  private mmr(
    candidates: Example[],
    queryEmb: number[],
    k: number,
    lambdaParam: number
  ): Example[] {
    const selected: Example[] = [];
    const remaining = [...candidates];

    for (let step = 0; step < Math.min(k, candidates.length); step++) {
      let bestScore = -Infinity;
      let bestIndex = -1;

      for (let i = 0; i < remaining.length; i++) {
        const ex = remaining[i];
        if (ex.embedding === null) continue;

        const relevance =
          cosineSimilarity(queryEmb, ex.embedding) * ex.qualityScore;

        let maxSim = 0;
        for (const sel of selected) {
          if (sel.embedding === null) continue;
          const sim = cosineSimilarity(ex.embedding, sel.embedding);
          if (sim > maxSim) maxSim = sim;
        }

        const mmrScore =
          lambdaParam * relevance - (1 - lambdaParam) * maxSim;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIndex = i;
        }
      }

      if (bestIndex === -1) break;
      selected.push(remaining[bestIndex]);
      remaining.splice(bestIndex, 1);
    }

    return this.applyTokenBudget(selected);
  }

  private applyTokenBudget(examples: Example[]): Example[] {
    const result: Example[] = [];
    let totalTokens = 0;

    for (const ex of examples) {
      if (totalTokens + ex.tokenCount > this.options.maxTokens) break;
      result.push(ex);
      totalTokens += ex.tokenCount;
    }

    return result;
  }
}

// --- Usage ---

const library: Example[] = [
  createExample({
    inputText:
      "Classify: 'Your account has been suspended, click here to verify'",
    outputText: '{"category": "phishing", "confidence": 0.95}',
    category: "email_classification",
    embedding: [0.8, 0.1, 0.05, 0.02, 0.03],
    qualityScore: 0.95,
  }),
  createExample({
    inputText:
      "Classify: 'Meeting tomorrow at 3pm to discuss Q4 roadmap'",
    outputText: '{"category": "meeting", "confidence": 0.92}',
    category: "email_classification",
    embedding: [0.1, 0.8, 0.05, 0.03, 0.02],
    qualityScore: 0.9,
  }),
  createExample({
    inputText: "Classify: 'Invoice #4521 is attached for your review'",
    outputText: '{"category": "finance", "confidence": 0.88}',
    category: "email_classification",
    embedding: [0.05, 0.1, 0.8, 0.03, 0.02],
    qualityScore: 0.85,
  }),
  createExample({
    inputText:
      "Classify: 'Congratulations! You've won a free iPhone'",
    outputText: '{"category": "spam", "confidence": 0.97}',
    category: "email_classification",
    embedding: [0.75, 0.05, 0.1, 0.05, 0.05],
    qualityScore: 0.92,
  }),
  createExample({
    inputText:
      "Classify: 'The deployment pipeline failed on staging'",
    outputText: '{"category": "engineering", "confidence": 0.91}',
    category: "email_classification",
    embedding: [0.02, 0.15, 0.03, 0.7, 0.1],
    qualityScore: 0.88,
  }),
];

const selector = new FewShotSelector({
  maxExamples: 3,
  maxTokens: 1000,
  strategy: "hybrid",
  diversityWeight: 0.3,
});

selector.addExamples(library);

// At runtime: select examples for a new query
const query = "Classify: 'Urgent: verify your bank details immediately'";
const queryEmbedding = [0.82, 0.08, 0.04, 0.03, 0.03];

const selected = selector.select({
  query,
  queryEmbedding,
  categoryFilter: "email_classification",
});

console.log(`Selected ${selected.length} examples:\n`);
console.log(selector.formatExamples(selected));
```

## Trade-offs

| Pros | Cons |
|------|------|
| Examples are always relevant to the current input | Requires an embedding model or classification system |
| Library can grow independently of prompt templates | Embedding computation adds latency per request |
| MMR diversity prevents redundant examples | Cold start: need enough examples before curation adds value |
| Quality scoring enables continuous improvement | Quality scoring requires outcome tracking infrastructure |
| Token budget is managed automatically | Maintaining example quality at scale requires curation effort |

## When to Use

- Classification systems where the category space is large and varied
- Customer support bots that need to match past resolutions to new tickets
- Code generation tools that benefit from similar code examples
- Any prompt with few-shot examples where the domain has more than ~10 distinct patterns
- Systems where example quality degrades over time (changing APIs, evolving terminology)

## When NOT to Use

- When you have fewer than 10 examples total -- static selection is fine
- Tasks where all examples are equally relevant regardless of input (e.g., formatting instructions)
- Latency-critical paths where embedding computation is too expensive
- When the output format is the same for all inputs and examples only demonstrate tone

## Related Patterns

- [System Prompt Architecture](system-prompt-architecture.md) -- defines the examples section that Few-Shot Curation populates
- [Progressive Disclosure](progressive-disclosure.md) -- may trigger example inclusion only when the model needs demonstration

## Real-World Examples

- **Production classification APIs** (e.g., at fintech companies) maintain libraries of thousands of labeled transaction descriptions. When a new transaction comes in, the most similar past examples are selected as few-shot context, dramatically improving accuracy on edge cases.
- **Customer support platforms** (Intercom, Zendesk AI) match incoming tickets against a library of resolved tickets using embedding similarity, then include the best matches as examples in the prompt so the model can follow established resolution patterns.
- **Code completion systems** use the current file and cursor context to select relevant code examples from the user's own codebase or from a curated library, providing the model with patterns that match the current coding context.
- **Legal document analysis tools** select example annotations from similar contract types (NDA vs. SLA vs. employment agreement) rather than using a generic set, improving extraction accuracy by 15-30% compared to static examples.
