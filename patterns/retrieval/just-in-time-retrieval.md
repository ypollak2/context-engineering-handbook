# Just-in-Time Retrieval

> Fetch context only at the moment it's needed rather than pre-loading everything upfront.

## Problem

Many LLM applications eagerly load all potentially relevant context into the prompt before the model even sees the user's message. This wastes tokens on information the model may never need, pushes out content that matters, and serves stale data that could have been fetched fresh at the moment of use. The result is bloated prompts, higher costs, and worse model performance due to attention dilution.

## Solution

Just-in-Time (JIT) Retrieval inverts the loading strategy: instead of pre-fetching everything, you register retrieval hooks that trigger based on conversation state, detected user intent, or tool invocations. Context is fetched at the latest responsible moment -- when there is concrete evidence the model needs it.

This is analogous to lazy loading in software engineering. You define *what* can be retrieved and *when* it should be retrieved, but the actual fetch only happens if the trigger condition is met. The hooks can fire on intent classification signals (the user asked about billing), on tool calls (the agent is about to call a database tool and needs the schema), or on explicit model requests (the model asks for more context via a retrieval tool).

## How It Works

```
User Message
     |
     v
+------------------+
| Intent Detection |  <-- Classify what the user needs
+------------------+
     |
     v
+------------------+     +-------------------+
| Hook Registry    | --> | Matching Hooks    |
| (intent -> fetch)|     | fire in parallel  |
+------------------+     +-------------------+
                              |
                              v
                    +--------------------+
                    | Retrieved Context  |
                    | injected into      |
                    | prompt assembly    |
                    +--------------------+
                              |
                              v
                    +--------------------+
                    | LLM Call with      |
                    | minimal, fresh     |
                    | context            |
                    +--------------------+
```

1. **Register hooks** -- Each hook declares a trigger condition and a retrieval function.
2. **Detect intent** -- On each turn, classify the user's intent or parse tool-call signals.
3. **Fire matching hooks** -- Only hooks whose conditions match the current state execute.
4. **Assemble context** -- Retrieved results are injected into the prompt alongside the conversation.
5. **Call the model** -- The LLM sees only context that is demonstrably relevant to this turn.

## Implementation

### Python

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# A retrieval hook: a condition function and an async fetcher
@dataclass(frozen=True)
class RetrievalHook:
    """A hook that fires when its condition matches the current turn."""
    name: str
    condition: Callable[[TurnContext], bool]
    fetch: Callable[[TurnContext], Awaitable[list[str]]]
    priority: int = 0  # lower = higher priority


@dataclass(frozen=True)
class TurnContext:
    """Immutable snapshot of the current conversation turn."""
    user_message: str
    intent: str
    conversation_history: tuple[dict[str, str], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class JITContextManager:
    """Manages just-in-time retrieval hooks and fires them based on turn context."""

    def __init__(self) -> None:
        self._hooks: list[RetrievalHook] = []

    def register(self, hook: RetrievalHook) -> None:
        self._hooks = sorted(
            [*self._hooks, hook],
            key=lambda h: h.priority,
        )

    async def retrieve(self, turn: TurnContext) -> dict[str, list[str]]:
        """Evaluate all hooks against the current turn and fetch matching context."""
        matching = [h for h in self._hooks if h.condition(turn)]
        if not matching:
            return {}

        # Fire all matching hooks in parallel
        results = await asyncio.gather(
            *(hook.fetch(turn) for hook in matching),
            return_exceptions=True,
        )

        retrieved: dict[str, list[str]] = {}
        for hook, result in zip(matching, results):
            if isinstance(result, BaseException):
                # Graceful degradation: log and skip failed hooks
                print(f"Hook '{hook.name}' failed: {result}")
                continue
            retrieved[hook.name] = result

        return retrieved


# --- Example usage ---

async def fetch_billing_docs(turn: TurnContext) -> list[str]:
    """Simulate fetching billing documentation chunks."""
    # In production: query a vector store filtered by topic="billing"
    return [
        "Billing FAQ: Invoices are generated on the 1st of each month.",
        "Billing FAQ: Refunds take 5-10 business days to process.",
    ]


async def fetch_account_details(turn: TurnContext) -> list[str]:
    """Simulate fetching the user's account details."""
    user_id = turn.metadata.get("user_id", "unknown")
    # In production: query your account service
    return [
        f"Account {user_id}: Pro plan, active since 2024-01-15.",
        f"Account {user_id}: Payment method Visa ending 4242.",
    ]


async def main() -> None:
    manager = JITContextManager()

    manager.register(RetrievalHook(
        name="billing_docs",
        condition=lambda t: t.intent in ("billing", "invoice", "refund"),
        fetch=fetch_billing_docs,
        priority=0,
    ))

    manager.register(RetrievalHook(
        name="account_details",
        condition=lambda t: "account" in t.user_message.lower() or t.intent == "billing",
        fetch=fetch_account_details,
        priority=1,
    ))

    # Simulate a turn where the user asks about billing
    turn = TurnContext(
        user_message="Why was I charged twice this month?",
        intent="billing",
        conversation_history=(),
        metadata={"user_id": "usr_8832"},
    )

    context = await manager.retrieve(turn)

    # Build the prompt with only the retrieved context
    context_block = "\n".join(
        f"[{source}]\n" + "\n".join(chunks)
        for source, chunks in context.items()
    )
    print("--- Injected Context ---")
    print(context_block)


if __name__ == "__main__":
    asyncio.run(main())
```

### TypeScript

```typescript
// just-in-time-retrieval.ts

interface TurnContext {
  readonly userMessage: string;
  readonly intent: string;
  readonly conversationHistory: ReadonlyArray<{ role: string; content: string }>;
  readonly metadata: Readonly<Record<string, unknown>>;
}

interface RetrievalHook {
  readonly name: string;
  readonly condition: (turn: TurnContext) => boolean;
  readonly fetch: (turn: TurnContext) => Promise<string[]>;
  readonly priority: number;
}

class JITContextManager {
  private hooks: RetrievalHook[] = [];

  register(hook: RetrievalHook): void {
    this.hooks = [...this.hooks, hook].sort((a, b) => a.priority - b.priority);
  }

  async retrieve(turn: TurnContext): Promise<Record<string, string[]>> {
    const matching = this.hooks.filter((h) => h.condition(turn));
    if (matching.length === 0) return {};

    // Fire all matching hooks in parallel
    const settled = await Promise.allSettled(
      matching.map((hook) => hook.fetch(turn))
    );

    const retrieved: Record<string, string[]> = {};
    for (let i = 0; i < matching.length; i++) {
      const result = settled[i];
      if (result.status === "rejected") {
        console.error(`Hook '${matching[i].name}' failed:`, result.reason);
        continue;
      }
      retrieved[matching[i].name] = result.value;
    }

    return retrieved;
  }
}

// --- Example usage ---

async function fetchBillingDocs(_turn: TurnContext): Promise<string[]> {
  // In production: query a vector store filtered by topic
  return [
    "Billing FAQ: Invoices are generated on the 1st of each month.",
    "Billing FAQ: Refunds take 5-10 business days to process.",
  ];
}

async function fetchAccountDetails(turn: TurnContext): Promise<string[]> {
  const userId = (turn.metadata.user_id as string) ?? "unknown";
  return [
    `Account ${userId}: Pro plan, active since 2024-01-15.`,
    `Account ${userId}: Payment method Visa ending 4242.`,
  ];
}

async function main(): Promise<void> {
  const manager = new JITContextManager();

  manager.register({
    name: "billing_docs",
    condition: (t) => ["billing", "invoice", "refund"].includes(t.intent),
    fetch: fetchBillingDocs,
    priority: 0,
  });

  manager.register({
    name: "account_details",
    condition: (t) =>
      t.userMessage.toLowerCase().includes("account") || t.intent === "billing",
    fetch: fetchAccountDetails,
    priority: 1,
  });

  const turn: TurnContext = {
    userMessage: "Why was I charged twice this month?",
    intent: "billing",
    conversationHistory: [],
    metadata: { user_id: "usr_8832" },
  };

  const context = await manager.retrieve(turn);

  const contextBlock = Object.entries(context)
    .map(([source, chunks]) => `[${source}]\n${chunks.join("\n")}`)
    .join("\n\n");

  console.log("--- Injected Context ---");
  console.log(contextBlock);
}

main();
```

## Trade-offs

| Pros | Cons |
|------|------|
| Maximizes context freshness -- data is fetched at the moment of use | Adds latency to each turn (mitigated by parallel fetching) |
| Minimizes token waste -- only relevant context is loaded | Requires an intent detection or signal parsing layer |
| Scales well -- adding new hooks does not bloat the base prompt | Hook conditions can become complex to maintain over time |
| Graceful degradation -- failed hooks do not block the response | First-turn latency is higher than pre-loaded approaches |
| Decouples context sources from prompt construction | Debugging is harder when context injection is dynamic |

## When to Use

- Your application has many possible context sources but only a few are relevant per turn.
- Context freshness matters -- you are retrieving live data (account info, inventory, prices).
- You are building an agent that invokes tools and needs tool-specific context only when a tool is selected.
- Token budget is tight and you cannot afford to pre-load everything.
- You are integrating with MCP servers where tool schemas should be fetched on demand.

## When NOT to Use

- The context is small and static enough to always include (e.g., a short system prompt with 3 FAQ entries).
- Every turn always needs the same context -- pre-loading is simpler and avoids the hook machinery.
- Latency is the top priority and you cannot tolerate any per-turn fetch overhead.
- You lack a reliable signal for what context is needed (no intent classifier, no tool-call signals).

## Related Patterns

- [RAG Context Assembly](rag-context-assembly.md) -- Once JIT retrieval fires, use RAG Context Assembly to rank and budget the retrieved chunks.
- [Semantic Tool Selection](semantic-tool-selection.md) -- A specialized form of JIT retrieval for tool descriptions rather than knowledge.
- [Progressive Disclosure](../construction/progressive-disclosure.md) -- The construction-side complement: reveals prompt sections incrementally, while JIT retrieval fetches external data incrementally.

## Real-World Examples

- **MCP (Model Context Protocol) servers**: Tool schemas and resource contents are fetched only when the agent signals intent to use a specific tool or resource, rather than loading all available schemas at connection time.
- **Cursor / Cody / Continue (IDE assistants)**: File contents, symbol definitions, and documentation are retrieved only when the user references a file or the model requests more context about a symbol. The full codebase is never pre-loaded.
- **Intercom Fin**: Customer support context (account details, recent tickets, knowledge base articles) is fetched based on the detected topic of the customer's message rather than loading the entire customer profile upfront.
- **ChatGPT Plugins (legacy) / GPT Actions**: External API calls are made only when the model decides to invoke a specific action, not pre-fetched at conversation start.
