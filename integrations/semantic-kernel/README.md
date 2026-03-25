# Semantic Kernel Integration

Implements 3 Context Engineering Handbook patterns using Microsoft Semantic Kernel v1.0+ Python SDK, focused on enterprise AI scenarios.

## Patterns Covered

| Pattern | File | Why This Framework |
|---------|------|--------------------|
| [System Prompt Architecture](../../patterns/construction/system-prompt-architecture.md) | `system_prompt_architecture.py` | `KernelFunction` composition with conditional prompt sections |
| [Semantic Tool Selection](../../patterns/retrieval/semantic-tool-selection.md) | `semantic_tool_selection.py` | Custom plugin selector using semantic similarity to filter plugins |
| [KV-Cache Optimization](../../patterns/optimization/kv-cache-optimization.md) | `kv_cache_optimization.py` | Prompt template management with stable prefixes for cache hits |

## Why These Patterns

Semantic Kernel excels at enterprise AI orchestration. These three patterns target areas where SK's abstractions are strongest:

1. **System Prompt Architecture**: SK's `KernelFunction` and prompt template system are purpose-built for composable, parameterized prompts. The handbook's modular section approach maps directly to SK's template composition.

2. **Semantic Tool Selection**: SK's plugin architecture supports hundreds of functions across enterprise integrations. Dynamic selection by embedding similarity prevents overwhelming the model.

3. **KV-Cache Optimization**: SK's prompt rendering pipeline gives fine-grained control over message ordering. Separating frozen prefixes from dynamic suffixes maximizes cache reuse with Azure OpenAI.

## Prerequisites

- Python 3.11+
- An Azure OpenAI or OpenAI API key
- Semantic Kernel v1.0+ Python SDK

## Quick Start

```bash
cd integrations/semantic-kernel/python
pip install -r requirements.txt

# Set one of these:
export OPENAI_API_KEY="sk-..."
# or for Azure:
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."

python system_prompt_architecture.py
python semantic_tool_selection.py
python kv_cache_optimization.py
```

## Framework-Specific Notes

### How Semantic Kernel Differs from the Generic Patterns

1. **KernelFunction composition**: SK treats prompts as functions that can be composed, parameterized, and invoked. System prompt sections become KernelFunctions that render their content based on runtime conditions.

2. **Plugin architecture**: SK plugins are collections of functions (native or semantic). The tool selection pattern filters which plugin functions are visible to the planner, rather than filtering raw tool descriptions.

3. **Azure OpenAI integration**: SK has first-class support for Azure OpenAI's prompt caching. The KV-cache pattern leverages this with explicit cache control hints.

4. **Planner integration**: SK's planners (Handlebars, Stepwise) automatically select functions to call. Semantic tool selection acts as a pre-filter that limits what the planner sees.

5. **Enterprise patterns**: SK is designed for enterprise scenarios with multiple services. The patterns reflect this with support for feature flags, user tiers, and conditional section inclusion.
