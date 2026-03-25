# Context Engineering Benchmarks - TypeScript

TypeScript implementation of the context engineering benchmark suite.

## Setup

```bash
npm install
```

## Usage

```bash
# Run a single benchmark
npx ts-node --esm src/runner.ts --benchmark needle-in-haystack --model gpt-4

# Run all benchmarks
npx ts-node --esm src/runner.ts --all --model claude-3-5-sonnet-20241022

# Output as JSON
npx ts-node --esm src/runner.ts --all --model gpt-4 --output json

# Save results to file
npx ts-node --esm src/runner.ts --all --model gpt-4 --output json --output-file ../results/run.json
```

## Environment Variables

Set one or both depending on which models you want to benchmark:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Available Benchmarks

- `needle-in-haystack` - Fact retrieval in large contexts
- `instruction-adherence` - System prompt rule compliance
- `compression-fidelity` - Info preservation after compaction
- `retrieval-relevance` - Relevant vs irrelevant chunk usage
- `token-efficiency` - Signal-to-noise ratio in context
