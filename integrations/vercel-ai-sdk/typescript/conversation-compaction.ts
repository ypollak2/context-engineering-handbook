/**
 * Conversation Compaction -- Vercel AI SDK Integration
 *
 * Implements structured fact extraction that hooks into the Vercel AI SDK's
 * onFinish callback. After each response, the compactor checks if the
 * conversation exceeds the token threshold and triggers compaction.
 *
 * Pattern: patterns/compression/conversation-compaction.md
 */

// ---------------------------------------------------------------------------
// Types (compatible with Vercel AI SDK's CoreMessage)
// ---------------------------------------------------------------------------

interface CoreMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content: string;
}

interface CompactionResult {
  readonly summary: string;
  readonly preservedCount: number;
  readonly removedCount: number;
  readonly tokensBefore: number;
  readonly tokensAfter: number;
  readonly tokensSaved: number;
}

interface CompactorConfig {
  readonly maxContextTokens: number;
  readonly compactionThreshold: number;
  readonly preserveRecentTurns: number;
}

const DEFAULT_CONFIG: CompactorConfig = {
  maxContextTokens: 100_000,
  compactionThreshold: 0.75,
  preserveRecentTurns: 10,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessageTokens(msg: CoreMessage): number {
  return estimateTokens(msg.content);
}

// ---------------------------------------------------------------------------
// ConversationCompactor
// ---------------------------------------------------------------------------

/**
 * Compacts conversation history using structured fact extraction.
 *
 * Designed to work with the Vercel AI SDK's onFinish callback:
 *
 *   import { streamText } from 'ai';
 *   import { openai } from '@ai-sdk/openai';
 *
 *   const compactor = createCompactor({ maxContextTokens: 100_000 });
 *   let messages: CoreMessage[] = [...];
 *
 *   const result = streamText({
 *     model: openai('gpt-4o'),
 *     messages,
 *     onFinish: async ({ text }) => {
 *       messages = [...messages, { role: 'assistant', content: text }];
 *       if (compactor.shouldCompact(messages)) {
 *         const compacted = await compactor.compact(messages);
 *         messages = compacted.messages;
 *         console.log(`Saved ${compacted.result.tokensSaved} tokens`);
 *       }
 *     },
 *   });
 */
class ConversationCompactor {
  private readonly config: CompactorConfig;
  private readonly extractFn:
    | ((turns: string) => Promise<string>)
    | null;

  constructor(
    config: Partial<CompactorConfig> = {},
    extractFn: ((turns: string) => Promise<string>) | null = null
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.extractFn = extractFn;
  }

  shouldCompact(messages: readonly CoreMessage[]): boolean {
    const totalTokens = messages.reduce(
      (sum, m) => sum + estimateMessageTokens(m),
      0
    );
    return (
      totalTokens >
      this.config.maxContextTokens * this.config.compactionThreshold
    );
  }

  async compact(
    messages: readonly CoreMessage[]
  ): Promise<{ messages: CoreMessage[]; result: CompactionResult }> {
    const tokensBefore = messages.reduce(
      (sum, m) => sum + estimateMessageTokens(m),
      0
    );

    if (!this.shouldCompact(messages)) {
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

    const { toCompact, toPreserve } = this.splitMessages(messages);

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

    // Extract structured summary
    const formattedTurns = toCompact
      .map((m) => `[${m.role}]: ${m.content}`)
      .join("\n");

    let summary: string;
    if (this.extractFn) {
      summary = await this.extractFn(formattedTurns);
    } else {
      summary = this.fallbackExtract(toCompact);
    }

    const summaryMessage: CoreMessage = {
      role: "system",
      content: `[COMPACTED CONTEXT from ${toCompact.length} earlier turns]\n\n${summary}`,
    };

    const newMessages: CoreMessage[] = [summaryMessage, ...toPreserve];
    const tokensAfter = newMessages.reduce(
      (sum, m) => sum + estimateMessageTokens(m),
      0
    );

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

  private splitMessages(
    messages: readonly CoreMessage[]
  ): { toCompact: CoreMessage[]; toPreserve: CoreMessage[] } {
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystem = messages.filter((m) => m.role !== "system");

    if (nonSystem.length <= this.config.preserveRecentTurns) {
      return { toCompact: [], toPreserve: [...messages] };
    }

    const splitPoint = nonSystem.length - this.config.preserveRecentTurns;
    return {
      toCompact: nonSystem.slice(0, splitPoint),
      toPreserve: [...systemMessages, ...nonSystem.slice(splitPoint)],
    };
  }

  private fallbackExtract(messages: readonly CoreMessage[]): string {
    const decisions: string[] = [];
    const facts: string[] = [];

    for (const msg of messages) {
      for (const line of msg.content.split("\n")) {
        const lower = line.trim().toLowerCase();
        if (lower.includes("decided") || lower.includes("decision:")) {
          decisions.push(`- ${line.trim()}`);
        } else if (
          ["error:", "file:", "path:", "version:", "using"].some((kw) =>
            lower.includes(kw)
          )
        ) {
          facts.push(`- ${line.trim()}`);
        }
      }
    }

    return [
      "## Decisions Made",
      ...(decisions.length > 0
        ? decisions
        : ["- No explicit decisions captured"]),
      "\n## Facts Established",
      ...(facts.length > 0 ? facts : ["- No explicit facts captured"]),
      `\n## Current State\n- Compacted ${messages.length} messages`,
    ].join("\n");
  }
}

/** Factory function for creating the compactor. */
function createCompactor(
  config: Partial<CompactorConfig> = {},
  extractFn: ((turns: string) => Promise<string>) | null = null
): ConversationCompactor {
  return new ConversationCompactor(config, extractFn);
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const compactor = createCompactor({
    maxContextTokens: 500,
    compactionThreshold: 0.5,
    preserveRecentTurns: 4,
  });

  const messages: CoreMessage[] = [
    { role: "system", content: "You are a coding assistant." },
    { role: "user", content: "Fix the auth bug in login.py" },
    {
      role: "assistant",
      content:
        "I see the bug. The token validation is missing. Decision: use JWT tokens.",
    },
    {
      role: "user",
      content: "Yes, use JWT. Also, the file path is src/auth/login.py",
    },
    {
      role: "assistant",
      content:
        "Updated src/auth/login.py with JWT validation. Using PyJWT version 2.8.",
    },
    { role: "user", content: "Now add rate limiting" },
    {
      role: "assistant",
      content:
        "Decision: decided to use Redis for rate limit storage. Adding middleware.",
    },
    { role: "user", content: "Use 5 req/min limit" },
    {
      role: "assistant",
      content: "Added rate limiting with 5 req/min using Redis.",
    },
    { role: "user", content: "Now add logging" },
    {
      role: "assistant",
      content: "I'll add structured logging to the auth module.",
    },
  ];

  console.log(`Messages: ${messages.length}`);
  console.log(`Should compact: ${compactor.shouldCompact(messages)}`);

  if (compactor.shouldCompact(messages)) {
    const { messages: newMessages, result } =
      await compactor.compact(messages);

    console.log(`\nCompaction result:`);
    console.log(`  Removed: ${result.removedCount} turns`);
    console.log(`  Preserved: ${result.preservedCount} turns`);
    console.log(`  Tokens saved: ${result.tokensSaved}`);
    console.log(`  New message count: ${newMessages.length}`);

    console.log(`\n=== Compacted messages ===`);
    for (const msg of newMessages) {
      console.log(`[${msg.role}]: ${msg.content.slice(0, 200)}...`);
      console.log();
    }
  }
}

main();
