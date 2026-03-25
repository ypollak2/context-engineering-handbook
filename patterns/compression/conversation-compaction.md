# Conversation Compaction

> Periodically summarize older conversation turns into condensed fact blocks, enabling unbounded session length without losing critical context.

## Problem

Long-running agent sessions (debugging, multi-step feature development, iterative design) generate dozens or hundreds of conversation turns. Eventually the context window fills up, and the model either loses access to early decisions or the session must be restarted.

Without compaction:
- Early decisions, user preferences, and established facts get evicted from the context window.
- The model "forgets" what was already discussed and repeats questions or contradicts earlier choices.
- Users must manually re-explain context after every window overflow.
- Session continuity breaks down, destroying the value of long-running interactions.

The key challenge is deciding **what to preserve** (decisions made, facts established, user preferences, file paths modified) versus **what to discard** (reasoning chains, exploratory tangents, superseded information, verbose tool outputs).

## Solution

Monitor the token count of the conversation. When it crosses a threshold, invoke an LLM pass over the older turns to extract a structured summary. Replace the original turns with this summary, freeing tokens for new work while preserving the semantic core of the conversation.

The summary is not a generic "what happened" paragraph. It is a structured extraction of:
1. **Decisions made** -- what was agreed upon.
2. **Facts established** -- concrete information discovered during the session.
3. **Current state** -- what files were modified, what step we are on, what remains.
4. **User preferences** -- expressed constraints, style preferences, or requirements.

## How It Works

```
Turn 1-20 (original)     Turn 21-40 (original)     Turn 41+ (live)
+-----------------------+ +-----------------------+ +------------------+
| User: fix the auth... | | User: now add rate... | | User: let's also |
| Asst: I see the bug...| | Asst: I'll add a...  | | add logging...   |
| User: yes, use JWT... | | User: use Redis...   | |                  |
| ...                   | | ...                   | |                  |
+-----------------------+ +-----------------------+ +------------------+

        |                          |
        v                          v
  [Compaction LLM pass]     [Compaction LLM pass]
        |                          |
        v                          v
+-------------------------------------------------------+
| COMPACTED CONTEXT (structured summary)                |
| Decisions: JWT auth chosen, Redis for rate limiting   |
| Files modified: auth.py, middleware.py, redis.conf    |
| Current state: Auth + rate limiting complete          |
| User preferences: Prefers explicit error messages     |
+-------------------------------------------------------+

Total: ~200 tokens instead of ~4,000
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    token_count: int = 0


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    preserved_count: int
    removed_count: int
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass(frozen=True)
class ConversationCompactor:
    """Compacts older conversation turns into structured summaries.

    Immutable: every operation returns a new instance or new data.
    """

    max_context_tokens: int = 100_000
    compaction_threshold: float = 0.75  # Trigger at 75% capacity
    preserve_recent_turns: int = 10     # Always keep the last N turns
    summary_model: str = "claude-sonnet-4-20250514"

    EXTRACTION_PROMPT: str = field(default="""Analyze these conversation turns and extract a structured summary.

CONVERSATION TURNS:
{turns}

Extract the following categories. Be precise and factual. Include file paths,
variable names, and specific values -- not vague descriptions.

## Decisions Made
- List each decision as: "<what> -- <why, if stated>"

## Facts Established
- Concrete information discovered (error messages, file contents, API behavior)

## Current State
- Files modified and how
- Current step in the overall task
- What remains to be done

## User Preferences
- Expressed constraints, style preferences, or requirements

## Key Context
- Any other information the assistant would need to continue this conversation
  without re-reading the original turns

Keep the summary concise but complete. Prefer bullet points over prose.
Omit reasoning chains, exploratory tangents, and superseded information.""", repr=False)

    def should_compact(self, messages: list[Message]) -> bool:
        total_tokens = sum(m.token_count for m in messages)
        threshold = int(self.max_context_tokens * self.compaction_threshold)
        return total_tokens > threshold

    def select_turns_for_compaction(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
        """Split messages into (to_compact, to_preserve).

        System messages are always preserved. Recent turns are always preserved.
        """
        system_messages = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        if len(non_system) <= self.preserve_recent_turns:
            return [], messages

        split_point = len(non_system) - self.preserve_recent_turns
        to_compact = non_system[:split_point]
        to_preserve = system_messages + non_system[split_point:]

        return to_compact, to_preserve

    def build_extraction_prompt(self, turns: list[Message]) -> str:
        formatted = "\n".join(
            f"[{turn.role.value}]: {turn.content}" for turn in turns
        )
        return self.EXTRACTION_PROMPT.format(turns=formatted)

    async def compact(
        self,
        messages: list[Message],
        llm_client,  # Any client with async `complete(prompt) -> str`
    ) -> tuple[list[Message], CompactionResult]:
        """Compact older turns into a structured summary.

        Returns (new_messages, compaction_result). Does not mutate the input.
        """
        if not self.should_compact(messages):
            result = CompactionResult(
                summary="",
                preserved_count=len(messages),
                removed_count=0,
                tokens_before=sum(m.token_count for m in messages),
                tokens_after=sum(m.token_count for m in messages),
            )
            return list(messages), result

        to_compact, to_preserve = self.select_turns_for_compaction(messages)

        if not to_compact:
            result = CompactionResult(
                summary="",
                preserved_count=len(messages),
                removed_count=0,
                tokens_before=sum(m.token_count for m in messages),
                tokens_after=sum(m.token_count for m in messages),
            )
            return list(messages), result

        extraction_prompt = self.build_extraction_prompt(to_compact)
        summary = await llm_client.complete(extraction_prompt)

        summary_message = Message(
            role=Role.SYSTEM,
            content=f"[COMPACTED CONTEXT from {len(to_compact)} earlier turns]\n\n{summary}",
            token_count=len(summary.split()) * 2,  # Rough estimate; use tiktoken in production
        )

        new_messages = [summary_message] + to_preserve
        tokens_before = sum(m.token_count for m in messages)
        tokens_after = sum(m.token_count for m in new_messages)

        result = CompactionResult(
            summary=summary,
            preserved_count=len(to_preserve),
            removed_count=len(to_compact),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        return new_messages, result


# --- Usage Example ---

async def example_usage():
    """Demonstrates the compaction loop in a long-running agent session."""
    from some_llm_client import LLMClient  # Replace with your client

    client = LLMClient(model="claude-sonnet-4-20250514")
    compactor = ConversationCompactor(
        max_context_tokens=100_000,
        compaction_threshold=0.75,
        preserve_recent_turns=10,
    )

    messages: list[Message] = []

    while True:
        user_input = await get_user_input()
        messages.append(Message(role=Role.USER, content=user_input, token_count=len(user_input.split()) * 2))

        # Check if compaction is needed before generating a response
        if compactor.should_compact(messages):
            messages, result = await compactor.compact(messages, client)
            print(f"Compacted: saved {result.tokens_saved} tokens, "
                  f"removed {result.removed_count} turns")

        response = await client.complete(messages)
        messages.append(Message(role=Role.ASSISTANT, content=response, token_count=len(response.split()) * 2))
```

### TypeScript

```typescript
type Role = "user" | "assistant" | "system";

interface Message {
  readonly role: Role;
  readonly content: string;
  readonly tokenCount: number;
}

interface CompactionResult {
  readonly summary: string;
  readonly preservedCount: number;
  readonly removedCount: number;
  readonly tokensBefore: number;
  readonly tokensAfter: number;
  readonly tokensSaved: number;
}

interface LLMClient {
  complete(prompt: string): Promise<string>;
}

interface CompactorConfig {
  readonly maxContextTokens: number;
  readonly compactionThreshold: number; // 0-1, trigger compaction at this % of capacity
  readonly preserveRecentTurns: number;
  readonly extractionPrompt: string;
}

const DEFAULT_EXTRACTION_PROMPT = `Analyze these conversation turns and extract a structured summary.

CONVERSATION TURNS:
{turns}

Extract the following categories. Be precise and factual. Include file paths,
variable names, and specific values -- not vague descriptions.

## Decisions Made
- List each decision as: "<what> -- <why, if stated>"

## Facts Established
- Concrete information discovered (error messages, file contents, API behavior)

## Current State
- Files modified and how
- Current step in the overall task
- What remains to be done

## User Preferences
- Expressed constraints, style preferences, or requirements

## Key Context
- Any other information the assistant would need to continue this conversation
  without re-reading the original turns

Keep the summary concise but complete. Prefer bullet points over prose.
Omit reasoning chains, exploratory tangents, and superseded information.`;

const DEFAULT_CONFIG: CompactorConfig = {
  maxContextTokens: 100_000,
  compactionThreshold: 0.75,
  preserveRecentTurns: 10,
  extractionPrompt: DEFAULT_EXTRACTION_PROMPT,
};

function shouldCompact(
  messages: readonly Message[],
  config: CompactorConfig
): boolean {
  const totalTokens = messages.reduce((sum, m) => sum + m.tokenCount, 0);
  return totalTokens > config.maxContextTokens * config.compactionThreshold;
}

function selectTurnsForCompaction(
  messages: readonly Message[],
  config: CompactorConfig
): { toCompact: readonly Message[]; toPreserve: readonly Message[] } {
  const systemMessages = messages.filter((m) => m.role === "system");
  const nonSystem = messages.filter((m) => m.role !== "system");

  if (nonSystem.length <= config.preserveRecentTurns) {
    return { toCompact: [], toPreserve: [...messages] };
  }

  const splitPoint = nonSystem.length - config.preserveRecentTurns;
  return {
    toCompact: nonSystem.slice(0, splitPoint),
    toPreserve: [...systemMessages, ...nonSystem.slice(splitPoint)],
  };
}

function buildExtractionPrompt(
  turns: readonly Message[],
  config: CompactorConfig
): string {
  const formatted = turns
    .map((t) => `[${t.role}]: ${t.content}`)
    .join("\n");
  return config.extractionPrompt.replace("{turns}", formatted);
}

async function compactConversation(
  messages: readonly Message[],
  client: LLMClient,
  config: CompactorConfig = DEFAULT_CONFIG
): Promise<{ messages: readonly Message[]; result: CompactionResult }> {
  const tokensBefore = messages.reduce((sum, m) => sum + m.tokenCount, 0);

  if (!shouldCompact(messages, config)) {
    return {
      messages: [...messages],
      result: {
        summary: "",
        preservedCount: messages.length,
        removedCount: 0,
        tokensBefore,
        tokensAfter: tokensBefore,
        tokensSaved: 0,
      },
    };
  }

  const { toCompact, toPreserve } = selectTurnsForCompaction(messages, config);

  if (toCompact.length === 0) {
    return {
      messages: [...messages],
      result: {
        summary: "",
        preservedCount: messages.length,
        removedCount: 0,
        tokensBefore,
        tokensAfter: tokensBefore,
        tokensSaved: 0,
      },
    };
  }

  const prompt = buildExtractionPrompt(toCompact, config);
  const summary = await client.complete(prompt);

  const summaryMessage: Message = {
    role: "system",
    content: `[COMPACTED CONTEXT from ${toCompact.length} earlier turns]\n\n${summary}`,
    tokenCount: Math.ceil(summary.split(/\s+/).length * 1.5),
  };

  const newMessages = [summaryMessage, ...toPreserve];
  const tokensAfter = newMessages.reduce((sum, m) => sum + m.tokenCount, 0);

  return {
    messages: newMessages,
    result: {
      summary,
      preservedCount: toPreserve.length,
      removedCount: toCompact.length,
      tokensBefore,
      tokensAfter,
      tokensSaved: tokensBefore - tokensAfter,
    },
  };
}

// --- Usage Example ---

async function agentLoop(client: LLMClient): Promise<void> {
  let messages: readonly Message[] = [];
  const config: CompactorConfig = {
    ...DEFAULT_CONFIG,
    maxContextTokens: 100_000,
    compactionThreshold: 0.75,
    preserveRecentTurns: 10,
  };

  while (true) {
    const userInput = await getUserInput();
    messages = [
      ...messages,
      { role: "user", content: userInput, tokenCount: estimateTokens(userInput) },
    ];

    if (shouldCompact(messages, config)) {
      const compacted = await compactConversation(messages, client, config);
      messages = compacted.messages;
      console.log(
        `Compacted: saved ${compacted.result.tokensSaved} tokens, ` +
        `removed ${compacted.result.removedCount} turns`
      );
    }

    const response = await client.complete(formatMessages(messages));
    messages = [
      ...messages,
      { role: "assistant", content: response, tokenCount: estimateTokens(response) },
    ];
  }
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Enables unbounded session length | Summarization is lossy -- nuance can be lost |
| Preserves decisions and facts structurally | Requires an extra LLM call per compaction (cost + latency) |
| User does not need to re-explain context | Summary quality depends on the summarization model |
| Works with any LLM provider | Incorrect summaries can compound over multiple compactions |
| Tunable aggressiveness via threshold and preservation count | Adds complexity to the conversation management layer |

## When to Use

- Long-running agent sessions (debugging, multi-step feature development, pair programming).
- Chat applications where users have multi-hour conversations.
- Agent loops that iterate many times (research, planning, code generation cycles).
- Any system where conversation history grows faster than it is consumed.

## When NOT to Use

- Short, single-turn interactions (question-answer, classification, extraction).
- When the full conversation history is legally or contractually required to be preserved verbatim (use a separate audit log).
- When token cost is the primary concern and tool outputs are the main offender (use Observation Masking instead -- it is cheaper).
- When exact reproduction of earlier reasoning is critical (compaction is lossy by design).

## Related Patterns

- **[Observation Masking](observation-masking.md)** -- Complementary compression pattern that targets tool outputs rather than conversation turns. Apply observation masking first; it is cheaper and handles the most common bloat source.
- **[Sub-Agent Delegation](../isolation/sub-agent-delegation.md)** -- An alternative to compaction: instead of compressing a long session, offload sub-tasks to child agents with fresh contexts.
- **Progressive Disclosure** (construction) -- Controls what enters the context in the first place; compaction cleans up what is already there.

## Real-World Examples

1. **Claude Code's `/compact` command and automatic compaction** -- When the context window fills up, Claude Code summarizes the conversation history, preserving modified files, test commands used, and key decisions. Users can also trigger compaction manually.

2. **ChatGPT's conversation memory** -- OpenAI's memory feature extracts long-term facts from conversations and stores them outside the context window, injecting them into future sessions as needed.

3. **Cursor's conversation summarization** -- The IDE assistant periodically compresses earlier conversation turns to keep the working context focused on the current coding task.

4. **Custom agent frameworks** -- Any agent that runs for more than ~20 turns in a loop (e.g., a research agent iterating on search queries) needs compaction to avoid context window exhaustion.
