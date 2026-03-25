/**
 * Conversation Compaction -- LangChain TypeScript Integration
 *
 * Structured fact extraction from older conversation turns using an LLM chain.
 * Replaces generic summarization with targeted extraction of decisions, facts,
 * current state, and user preferences.
 *
 * Pattern: patterns/compression/conversation-compaction.md
 */

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
// Extraction prompt
// ---------------------------------------------------------------------------

const EXTRACTION_PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a context compaction engine. Extract structured facts " +
      "from conversation turns. Be precise and factual. Include file " +
      "paths, variable names, and specific values -- not vague descriptions.\n\n" +
      "Output ONLY the structured summary in the format below.",
  ],
  [
    "human",
    "Analyze these conversation turns and extract a structured summary.\n\n" +
      "CONVERSATION TURNS:\n{turns}\n\n" +
      "Extract the following categories:\n\n" +
      "## Decisions Made\n" +
      '- List each decision as: "<what> -- <why, if stated>"\n\n' +
      "## Facts Established\n" +
      "- Concrete information discovered\n\n" +
      "## Current State\n" +
      "- Files modified, current step, what remains\n\n" +
      "## User Preferences\n" +
      "- Expressed constraints, style preferences\n\n" +
      "## Key Context\n" +
      "- Anything needed to continue without re-reading the original turns\n\n" +
      "Omit reasoning chains, exploratory tangents, and superseded information.",
  ],
]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessageTokens(msg: BaseMessage): number {
  const content =
    typeof msg.content === "string" ? msg.content : String(msg.content);
  return estimateTokens(content);
}

function formatTurns(messages: readonly BaseMessage[]): string {
  return messages
    .map((msg) => {
      const content =
        typeof msg.content === "string" ? msg.content : String(msg.content);
      return `[${msg._getType()}]: ${content}`;
    })
    .join("\n");
}

// ---------------------------------------------------------------------------
// ConversationCompactor
// ---------------------------------------------------------------------------

class ConversationCompactor {
  private readonly config: CompactorConfig;
  private readonly extractionChain: Runnable | null;

  constructor(
    llm: Runnable | null = null,
    config: Partial<CompactorConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.extractionChain = llm ? EXTRACTION_PROMPT.pipe(llm) : null;
  }

  shouldCompact(messages: readonly BaseMessage[]): boolean {
    const totalTokens = messages.reduce(
      (sum, m) => sum + estimateMessageTokens(m),
      0
    );
    return (
      totalTokens >
      this.config.maxContextTokens * this.config.compactionThreshold
    );
  }

  private splitMessages(
    messages: readonly BaseMessage[]
  ): { toCompact: BaseMessage[]; toPreserve: BaseMessage[] } {
    const systemMessages = messages.filter((m) => m instanceof SystemMessage);
    const nonSystem = messages.filter((m) => !(m instanceof SystemMessage));

    if (nonSystem.length <= this.config.preserveRecentTurns) {
      return { toCompact: [], toPreserve: [...messages] };
    }

    const splitPoint = nonSystem.length - this.config.preserveRecentTurns;
    return {
      toCompact: nonSystem.slice(0, splitPoint),
      toPreserve: [...systemMessages, ...nonSystem.slice(splitPoint)],
    };
  }

  async compact(
    messages: readonly BaseMessage[]
  ): Promise<{ messages: BaseMessage[]; result: CompactionResult }> {
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

    let summary: string;

    if (this.extractionChain) {
      const response = await this.extractionChain.invoke({
        turns: formatTurns(toCompact),
      });
      summary =
        typeof response.content === "string"
          ? response.content
          : String(response.content);
    } else {
      summary = this.fallbackExtract(toCompact);
    }

    const summaryMessage = new SystemMessage(
      `[COMPACTED CONTEXT from ${toCompact.length} earlier turns]\n\n${summary}`
    );

    const newMessages = [summaryMessage, ...toPreserve];
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

  private fallbackExtract(messages: readonly BaseMessage[]): string {
    const decisions: string[] = [];
    const facts: string[] = [];

    for (const msg of messages) {
      const content =
        typeof msg.content === "string" ? msg.content : String(msg.content);
      for (const line of content.split("\n")) {
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

    const parts = [
      "## Decisions Made",
      ...(decisions.length > 0
        ? decisions
        : ["- No explicit decisions captured"]),
      "\n## Facts Established",
      ...(facts.length > 0 ? facts : ["- No explicit facts captured"]),
      `\n## Current State\n- Compacted ${messages.length} messages`,
    ];
    return parts.join("\n");
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const compactor = new ConversationCompactor(null, {
    maxContextTokens: 500,
    compactionThreshold: 0.5,
    preserveRecentTurns: 4,
  });

  const messages: BaseMessage[] = [
    new SystemMessage("You are a coding assistant."),
    new HumanMessage("Fix the auth bug in login.py"),
    new AIMessage(
      "I see the bug. The token validation is missing. Decision: use JWT tokens."
    ),
    new HumanMessage("Yes, use JWT. Also, the file path is src/auth/login.py"),
    new AIMessage(
      "Updated src/auth/login.py with JWT validation. Using PyJWT version 2.8."
    ),
    new HumanMessage("Now add rate limiting to the login endpoint"),
    new AIMessage(
      "I'll add rate limiting. Decision: decided to use Redis for rate limit storage."
    ),
    new HumanMessage("Use a 5-request-per-minute limit"),
    new AIMessage(
      "Added rate limiting middleware with 5 req/min limit using Redis."
    ),
    new HumanMessage("Now let's add logging"),
    new AIMessage("I'll add structured logging to the auth module."),
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
    console.log(`\n=== Compacted context ===`);
    for (const msg of newMessages) {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : String(msg.content);
      console.log(`[${msg._getType()}]: ${content.slice(0, 200)}...`);
      console.log();
    }
  }
}

main();
