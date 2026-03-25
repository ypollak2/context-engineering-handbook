# Context Engineering Benchmark Suite

A benchmark suite for measuring how well your context engineering strategies work in practice. Includes 5 benchmarks that test different aspects of context quality, available in both Python and TypeScript.

## Quick Start

### Python

```bash
cd python
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run a single benchmark
python runner.py --benchmark needle_in_haystack --model gpt-4

# Run all benchmarks
python runner.py --all --model claude-3-5-sonnet-20241022

# Output as JSON
python runner.py --all --model gpt-4 --output json

# Save to file
python runner.py --all --model gpt-4 --output json --output-file ../results/run.json
```

### TypeScript

```bash
cd typescript
npm install

# Run a single benchmark
npx ts-node --esm src/runner.ts --benchmark needle-in-haystack --model gpt-4

# Run all benchmarks
npx ts-node --esm src/runner.ts --all --model claude-3-5-sonnet-20241022
```

## The 5 Benchmarks

### 1. Needle in Haystack

**What it measures**: Can the LLM find a specific fact buried in a large context? Tests attention degradation as context size grows and as the position of key information changes.

**Why it matters**: If your context window is stuffed with information, the model may lose track of important details -- especially in the middle of long contexts (the "lost in the middle" effect). This benchmark quantifies exactly where and when that happens.

**Metrics**:
- Overall recall rate
- Recall by position (beginning / middle / end)
- Recall by context size (1K / 2K / 4K / 8K tokens)

**What good looks like**:
| Score | Interpretation |
|-------|---------------|
| > 95% | Excellent -- context engineering is working well |
| 80-95% | Good -- some positional degradation, consider reordering |
| 60-80% | Fair -- important info is getting lost in larger contexts |
| < 60% | Poor -- context is too noisy or too large for the model |

### 2. Instruction Adherence

**What it measures**: Given 10 specific behavioral rules in a system prompt (e.g., "respond in JSON", "never use the word 'simply'", "include confidence scores"), how consistently does the model follow them across 20 diverse queries?

**Why it matters**: System prompts are the primary mechanism for controlling LLM behavior. If rules are ignored, your application's output will be unpredictable. This benchmark tells you exactly which rules are fragile and which are robust.

**Metrics**:
- Overall compliance rate (all rules x all queries)
- Per-rule adherence rate
- Per-query compliance rate

**What good looks like**:
| Score | Interpretation |
|-------|---------------|
| > 90% | Excellent -- instructions are clear and well-structured |
| 70-90% | Good -- some rules may need reinforcement or rephrasing |
| 50-70% | Fair -- consider reducing rule count or simplifying rules |
| < 50% | Poor -- too many rules, conflicting instructions, or rules too complex |

### 3. Compression Fidelity

**What it measures**: Take a conversation with known facts and decisions, compress it using 4 strategies (LLM summarization, head truncation, tail truncation, key-point extraction), then test if the compressed version preserves the original information.

**Why it matters**: Long conversations must be compressed to fit context windows. This benchmark reveals which compression strategy loses the least information for your use case, and quantifies the tradeoff between compression ratio and fidelity.

**Metrics**:
- Fact retention rate per strategy
- Decision retention rate per strategy
- Compression ratio per strategy

**What good looks like**:
| Strategy | Good Fact Retention | Good Decision Retention |
|----------|-------------------|----------------------|
| LLM Summary | > 85% | > 80% |
| Key Points | > 80% | > 85% |
| Truncation (head) | > 50% | > 40% |
| Truncation (tail) | > 40% | > 50% |

### 4. Retrieval Relevance

**What it measures**: Given a question and a mix of relevant and irrelevant retrieved chunks, does the model use the right chunks and ignore the noise?

**Why it matters**: RAG systems retrieve chunks that are not always relevant. If the model cannot distinguish signal from noise in retrieved context, answers will be contaminated by irrelevant information or miss key details.

**Metrics**:
- Answer accuracy (does the response contain the correct answer?)
- Relevant chunk utilization rate (did it use the relevant chunks?)
- Irrelevant chunk contamination rate (did it use irrelevant chunks?)

**What good looks like**:
| Metric | Good | Concerning |
|--------|------|-----------|
| Answer accuracy | > 90% | < 70% |
| Utilization rate | > 80% | < 50% |
| Contamination rate | < 10% | > 30% |

### 5. Token Efficiency

**What it measures**: For each scenario, we know exactly which context sections are "signal" (needed for the answer) and which are "noise" (irrelevant). The benchmark measures the signal-to-noise ratio and whether removing noise changes answer quality.

**Why it matters**: Every token costs money and consumes context window space. If you can get the same answer quality with fewer tokens, your application is cheaper, faster, and more reliable. This benchmark shows you how much waste is in your context.

**Metrics**:
- Effective token ratio (signal tokens / total tokens)
- Accuracy with full context vs. signal-only context
- Total wasted tokens across scenarios

**What good looks like**:
| Metric | Good | Concerning |
|--------|------|-----------|
| Effective ratio | > 60% | < 30% |
| Full accuracy = Signal accuracy | Context engineering is tight | N/A |
| Full accuracy < Signal accuracy | Noise is hurting performance | Investigate |

## Mapping Benchmarks to Patterns

When benchmark scores are low, apply these context engineering patterns:

| Low Score In | Apply These Patterns |
|-------------|---------------------|
| Needle in Haystack | **Progressive Disclosure** -- layer information from most to least relevant. **Attention Focusing** -- place critical facts at the start or end of context, not the middle. |
| Instruction Adherence | **Rule Prioritization** -- fewer, clearer rules. **Instruction Reinforcement** -- repeat critical rules. **Structured Output** -- use schema enforcement (e.g., JSON mode). |
| Compression Fidelity | **Hierarchical Summarization** -- summarize in stages. **Fact Extraction First** -- extract key facts before general summarization. **Observation Masking** -- remove verbose tool outputs, keep conclusions. |
| Retrieval Relevance | **Chunk Reranking** -- rerank retrieved chunks before injection. **Relevance Filtering** -- set a similarity threshold and drop low-scoring chunks. **Source Attribution** -- ask the model to cite chunks, forcing selective attention. |
| Token Efficiency | **Context Pruning** -- remove sections with low relevance scores. **Dynamic Context** -- only include context sections relevant to the current query. **Semantic Caching** -- cache and reuse context for repeated query patterns. |

## Sample Output

```
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
| Benchmark                  | Primary Metric                 | Details                                            | Scale              |
+============================+================================+====================================================+====================+
| needle_in_haystack         | Overall Recall: 91.7%          | beginning: 100.0% | middle: 75.0% | end: 100.0%    | 12 trials          |
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
| instruction_adherence      | Overall Compliance: 73.5%      | System prompt: 312 tokens                          | 10 rules tested    |
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
| compression_fidelity       | See strategy breakdown         | llm_summary: fact=92%, decision=90% | truncat...   | 8 trials           |
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
| retrieval_relevance        | Accuracy: 100.0%               | Utilization: 87.5% | Contamination: 6.2%           | 4 scenarios        |
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
| token_efficiency           | Signal ratio: 44.2%            | Full accuracy: 100.0% | Signal-only: 100.0%        | Wasted: 1847 tokens|
+----------------------------+--------------------------------+----------------------------------------------------+--------------------+
```

## Interpreting Results

**Reading the table above**:

1. **Needle in Haystack**: 91.7% overall recall is good, but middle-position recall is only 75% -- the "lost in the middle" effect is present. Place critical information at context boundaries.

2. **Instruction Adherence**: 73.5% compliance means roughly 1 in 4 rule-checks fail. Review per-rule results to find which rules are most fragile and simplify or reinforce them.

3. **Compression Fidelity**: LLM summarization retains 92% of facts, far better than truncation strategies. Use LLM-based compression when possible, but note the extra API call cost.

4. **Retrieval Relevance**: 100% accuracy with 87.5% utilization and only 6.2% contamination is strong. The retrieval pipeline is well-tuned.

5. **Token Efficiency**: A 44.2% signal ratio means over half the context tokens are noise. Since accuracy is 100% with both full and signal-only context, the noise is not hurting yet but is wasting tokens and money.

## Cost Considerations

Each benchmark makes multiple LLM API calls:

| Benchmark | Approximate API Calls | Notes |
|-----------|----------------------|-------|
| Needle in Haystack | 12 | 4 sizes x 3 positions |
| Instruction Adherence | 20 | 20 queries |
| Compression Fidelity | 24+ | 2 scenarios x 4 strategies x (1 compress + ~10 verify) |
| Retrieval Relevance | 4 | 4 scenarios |
| Token Efficiency | 6 | 3 scenarios x 2 (full + signal-only) |

Running all benchmarks with GPT-4 typically costs $1-3 USD. Using smaller models (GPT-3.5-turbo, Claude Haiku) reduces cost significantly while still providing useful relative comparisons.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── python/
│   ├── requirements.txt
│   ├── runner.py                # CLI entry point
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── needle_in_haystack.py
│   │   ├── instruction_adherence.py
│   │   ├── compression_fidelity.py
│   │   ├── retrieval_relevance.py
│   │   └── token_efficiency.py
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py
│       └── metrics.py
├── typescript/
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── runner.ts
│   │   ├── benchmarks/
│   │   │   ├── needle-in-haystack.ts
│   │   │   ├── instruction-adherence.ts
│   │   │   ├── compression-fidelity.ts
│   │   │   ├── retrieval-relevance.ts
│   │   │   └── token-efficiency.ts
│   │   └── utils/
│   │       ├── llm-client.ts
│   │       └── metrics.ts
│   └── README.md
└── results/
    └── .gitkeep
```
