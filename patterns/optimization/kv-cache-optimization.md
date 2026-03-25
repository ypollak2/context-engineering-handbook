# KV-Cache Optimization

> Structure your context to maximize KV-cache hit rates, dramatically reducing inference latency and cost by reusing cached key-value pairs across LLM calls.

## Problem

Every token in an LLM prompt requires computing key-value (KV) pairs during the attention mechanism. When you make repeated calls with similar prompts, the LLM provider recomputes these KV pairs from scratch each time -- even if 90% of the prompt is identical. This is wasteful:

- A system prompt of 2,000 tokens is reprocessed on every call
- Tool definitions (often 5,000+ tokens) are recomputed identically each time
- Agent loops with stable instructions pay the full compute cost per iteration

Without KV-cache optimization, you pay for the same computation repeatedly. With it, the provider recognizes the identical prefix and reuses cached KV pairs, charging only for the new tokens.

## Solution

Structure your context so that the **shared, stable content appears first** and **variable content is appended at the end**. Never insert new content into the middle of a stable prefix -- this invalidates the cache for everything after the insertion point. Think of your prompt as two zones:

1. **Frozen prefix**: System prompt, tool definitions, static instructions, few-shot examples
2. **Dynamic suffix**: User messages, retrieved context, current state

The frozen prefix hits the cache. The dynamic suffix is computed fresh. The larger the frozen prefix relative to total context, the greater the savings.

## How It Works

```
Without KV-Cache Optimization:
  Call 1: [System Prompt][Tools][History][User Msg 1]  --> compute ALL tokens
  Call 2: [System Prompt][Tools][History][User Msg 2]  --> compute ALL tokens again
  Call 3: [System Prompt][Tools][History][User Msg 3]  --> compute ALL tokens again

With KV-Cache Optimization:
  Call 1: [System Prompt][Tools][History][User Msg 1]  --> compute all, CACHE prefix
  Call 2: [System Prompt][Tools][History][User Msg 2]  --> REUSE cached prefix, compute only new
  Call 3: [System Prompt][Tools][History][User Msg 3]  --> REUSE cached prefix, compute only new
         |<--- frozen prefix --->|<-- dynamic -->|

Cost comparison (example: 8K token prefix, 500 token suffix):
  Without: 3 calls x 8,500 tokens = 25,500 tokens computed
  With:    8,500 + 500 + 500      = 9,500 tokens computed  (2.7x reduction)
```

Cache invalidation rules:
```
Prefix:  [A][B][C][D][E]
                        ^-- Appending [F] here: cache for [A-E] is valid

Prefix:  [A][B][X][D][E]
               ^-- Inserting/changing [X] here: cache for [D][E] is INVALID
```

## Implementation

### Python

```python
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MessageBlock:
    """An immutable message in the conversation."""
    role: str
    content: str
    cache_control: dict = field(default_factory=dict)


class StablePrefixManager:
    """
    Manages the frozen prefix portion of the context to maximize
    KV-cache hit rates. The prefix is assembled once and reused
    across calls. Only the dynamic suffix changes.
    """

    def __init__(self):
        self._system_prompt: str = ""
        self._tool_definitions: list[dict] = []
        self._static_examples: list[MessageBlock] = []
        self._prefix_hash: str = ""

    def set_system_prompt(self, prompt: str) -> "StablePrefixManager":
        """Set the system prompt. Returns new instance (immutable pattern)."""
        new = StablePrefixManager()
        new._system_prompt = prompt
        new._tool_definitions = list(self._tool_definitions)
        new._static_examples = list(self._static_examples)
        new._recompute_hash()
        return new

    def set_tool_definitions(self, tools: list[dict]) -> "StablePrefixManager":
        """Set tool definitions. Returns new instance."""
        new = StablePrefixManager()
        new._system_prompt = self._system_prompt
        new._tool_definitions = list(tools)
        new._static_examples = list(self._static_examples)
        new._recompute_hash()
        return new

    def set_static_examples(
        self, examples: list[MessageBlock]
    ) -> "StablePrefixManager":
        """Set few-shot examples. Returns new instance."""
        new = StablePrefixManager()
        new._system_prompt = self._system_prompt
        new._tool_definitions = list(self._tool_definitions)
        new._static_examples = list(examples)
        new._recompute_hash()
        return new

    @property
    def prefix_hash(self) -> str:
        return self._prefix_hash

    def build_prefix_messages(self) -> list[dict[str, Any]]:
        """
        Build the frozen prefix as a list of message dicts.
        Mark the last block with cache_control for providers that support it.
        """
        messages: list[dict[str, Any]] = []

        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt,
            })

        for example in self._static_examples:
            messages.append({
                "role": example.role,
                "content": example.content,
            })

        # Mark the last prefix message with cache_control breakpoint
        if messages:
            messages[-1]["cache_control"] = {"type": "ephemeral"}

        return messages

    def _recompute_hash(self) -> None:
        """Hash the prefix content for cache-hit tracking."""
        content = json.dumps({
            "system": self._system_prompt,
            "tools": self._tool_definitions,
            "examples": [
                {"role": e.role, "content": e.content}
                for e in self._static_examples
            ],
        }, sort_keys=True)
        self._prefix_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


class KVCacheOptimizedClient:
    """
    Wraps an LLM client to structure messages for maximum KV-cache reuse.

    Ensures the stable prefix is always identical across calls and that
    dynamic content is always appended, never inserted.
    """

    def __init__(self, llm_client, prefix_manager: StablePrefixManager):
        """
        Args:
            llm_client: Any LLM client with a chat() method that accepts
                        messages and tools parameters.
            prefix_manager: The stable prefix configuration.
        """
        self._client = llm_client
        self._prefix = prefix_manager
        self._cache_stats = {"hits": 0, "misses": 0, "total_calls": 0}

    def chat(
        self,
        dynamic_messages: list[dict[str, str]],
        **kwargs,
    ) -> Any:
        """
        Make an LLM call with optimized message ordering.

        Args:
            dynamic_messages: The variable portion (user messages, retrieved
                              context, conversation history).
            **kwargs: Additional arguments passed to the underlying client.
        """
        # Build the full message list: frozen prefix + dynamic suffix
        prefix_messages = self._prefix.build_prefix_messages()
        all_messages = prefix_messages + dynamic_messages

        # Track cache effectiveness
        self._cache_stats["total_calls"] += 1

        response = self._client.chat(
            messages=all_messages,
            tools=self._prefix._tool_definitions or None,
            **kwargs,
        )

        # If the provider returns cache info, track it
        usage = getattr(response, "usage", None)
        if usage:
            cached = getattr(usage, "cache_read_input_tokens", 0)
            if cached > 0:
                self._cache_stats["hits"] += 1
            else:
                self._cache_stats["misses"] += 1

        return response

    @property
    def cache_hit_rate(self) -> float:
        """Return the cache hit rate as a percentage."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total == 0:
            return 0.0
        return (self._cache_stats["hits"] / total) * 100

    @property
    def cache_stats(self) -> dict:
        """Return a copy of cache statistics."""
        return dict(self._cache_stats)


def build_append_only_history(
    existing_history: list[dict[str, str]],
    new_message: dict[str, str],
) -> list[dict[str, str]]:
    """
    Append a message to history without mutating the original list.
    This ensures the prefix portion of history remains cache-friendly.
    """
    return [*existing_history, new_message]
```

### TypeScript

```typescript
import { createHash } from "crypto";

interface Message {
  readonly role: string;
  readonly content: string;
  readonly cache_control?: { type: string };
}

interface LLMClient {
  chat(params: {
    messages: Message[];
    tools?: Record<string, unknown>[];
  }): Promise<{
    content: string;
    usage?: { cache_read_input_tokens?: number };
  }>;
}

interface CacheStats {
  hits: number;
  misses: number;
  totalCalls: number;
}

class StablePrefixManager {
  private readonly systemPrompt: string;
  private readonly toolDefinitions: readonly Record<string, unknown>[];
  private readonly staticExamples: readonly Message[];
  readonly prefixHash: string;

  constructor(params?: {
    systemPrompt?: string;
    toolDefinitions?: Record<string, unknown>[];
    staticExamples?: Message[];
  }) {
    this.systemPrompt = params?.systemPrompt ?? "";
    this.toolDefinitions = Object.freeze([...(params?.toolDefinitions ?? [])]);
    this.staticExamples = Object.freeze([...(params?.staticExamples ?? [])]);
    this.prefixHash = this.computeHash();
  }

  withSystemPrompt(prompt: string): StablePrefixManager {
    return new StablePrefixManager({
      systemPrompt: prompt,
      toolDefinitions: [...this.toolDefinitions],
      staticExamples: [...this.staticExamples],
    });
  }

  withToolDefinitions(
    tools: Record<string, unknown>[]
  ): StablePrefixManager {
    return new StablePrefixManager({
      systemPrompt: this.systemPrompt,
      toolDefinitions: tools,
      staticExamples: [...this.staticExamples],
    });
  }

  withStaticExamples(examples: Message[]): StablePrefixManager {
    return new StablePrefixManager({
      systemPrompt: this.systemPrompt,
      toolDefinitions: [...this.toolDefinitions],
      staticExamples: examples,
    });
  }

  buildPrefixMessages(): Message[] {
    const messages: Message[] = [];

    if (this.systemPrompt) {
      messages.push({ role: "system", content: this.systemPrompt });
    }

    for (const example of this.staticExamples) {
      messages.push({ role: example.role, content: example.content });
    }

    // Mark last prefix message with cache breakpoint
    if (messages.length > 0) {
      const last = messages[messages.length - 1];
      messages[messages.length - 1] = {
        ...last,
        cache_control: { type: "ephemeral" },
      };
    }

    return messages;
  }

  getToolDefinitions(): Record<string, unknown>[] {
    return [...this.toolDefinitions];
  }

  private computeHash(): string {
    const content = JSON.stringify({
      system: this.systemPrompt,
      tools: this.toolDefinitions,
      examples: this.staticExamples.map((e) => ({
        role: e.role,
        content: e.content,
      })),
    });
    return createHash("sha256").update(content).digest("hex").slice(0, 16);
  }
}

class KVCacheOptimizedClient {
  private readonly client: LLMClient;
  private readonly prefix: StablePrefixManager;
  private readonly stats: CacheStats = { hits: 0, misses: 0, totalCalls: 0 };

  constructor(client: LLMClient, prefix: StablePrefixManager) {
    this.client = client;
    this.prefix = prefix;
  }

  async chat(
    dynamicMessages: Message[]
  ): Promise<{ content: string; cacheHit: boolean }> {
    const prefixMessages = this.prefix.buildPrefixMessages();
    const allMessages = [...prefixMessages, ...dynamicMessages];

    this.stats.totalCalls += 1;

    const tools = this.prefix.getToolDefinitions();
    const response = await this.client.chat({
      messages: allMessages,
      tools: tools.length > 0 ? tools : undefined,
    });

    const cachedTokens = response.usage?.cache_read_input_tokens ?? 0;
    const cacheHit = cachedTokens > 0;

    if (cacheHit) {
      this.stats.hits += 1;
    } else {
      this.stats.misses += 1;
    }

    return { content: response.content, cacheHit };
  }

  get cacheHitRate(): number {
    const total = this.stats.hits + this.stats.misses;
    if (total === 0) return 0;
    return (this.stats.hits / total) * 100;
  }

  get cacheStats(): Readonly<CacheStats> {
    return { ...this.stats };
  }
}

function buildAppendOnlyHistory(
  existingHistory: readonly Message[],
  newMessage: Message
): Message[] {
  return [...existingHistory, newMessage];
}

export {
  StablePrefixManager,
  KVCacheOptimizedClient,
  buildAppendOnlyHistory,
  Message,
  CacheStats,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Up to 7x cost reduction on repeated calls | Requires discipline in message ordering -- one misplaced insertion invalidates the cache |
| Significant latency reduction (cached tokens process faster) | Not all providers support KV-cache reuse (or expose it) |
| Zero code change needed on the model side | Cache is ephemeral -- provider may evict at any time |
| Composable with other optimization techniques | Forces a "prefix-first" message architecture that can feel rigid |
| Provider-agnostic principle (works with any prefix-caching system) | Dynamic system prompts (e.g., with timestamps) defeat the cache |
| Observable via cache hit rate metrics | Cache window varies by provider (typically 5-10 minutes of inactivity) |

## When to Use

- Agent loops where the system prompt and tool definitions are identical across iterations
- High-volume applications making many similar LLM calls per second
- Multi-turn conversations where the prefix grows but the stable portion dominates
- Any system where you control the message ordering and can enforce prefix stability
- Batch processing where many inputs share the same instruction prefix

## When NOT to Use

- Single-shot LLM calls with no repeated prefix
- When your system prompt changes frequently (e.g., includes real-time data at the top)
- Low-volume applications where the cost savings are negligible
- When using providers that do not support prefix caching
- When message ordering is dictated by external constraints you cannot control

## Related Patterns

- **System Prompt Architecture** (Construction): Designing modular system prompts makes it easier to keep the prefix stable. Separate the static architecture from dynamic sections.
- **Progressive Disclosure** (Construction): Append conditionally-loaded context at the end of the prefix, not in the middle, to preserve cache validity.
- **Error Preservation** (Optimization): Errors should be appended to the dynamic suffix, never inserted into the stable prefix.

## Real-World Examples

- **Manus**: Achieves approximately 7x cost reduction by carefully structuring prompts so that system instructions and tool definitions form a stable, cacheable prefix. This is documented as one of their key architectural decisions.
- **Anthropic Prompt Caching**: Anthropic's API explicitly supports cache_control breakpoints that tell the system where the stable prefix ends. This allows deterministic cache hits rather than relying on automatic prefix detection.
- **OpenAI Predicted Outputs**: A related optimization where the provider caches not just the input prefix but also predicted output structure, further reducing computation.
- **High-Volume Agent Systems**: Any agent making 100+ LLM calls per task (code generation, research, data analysis) benefits enormously. The stable prefix (instructions + tools) typically represents 60-80% of total tokens.
