/**
 * Progressive Disclosure -- Vercel AI SDK Integration
 *
 * Implements staged context injection as middleware that modifies the
 * system prompt before each model call. The middleware tracks conversation
 * state and dynamically assembles context sections.
 *
 * Pattern: patterns/construction/progressive-disclosure.md
 */

// ---------------------------------------------------------------------------
// Types (compatible with Vercel AI SDK's CoreMessage)
// ---------------------------------------------------------------------------

interface CoreMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content: string;
}

enum DisclosureStage {
  BASELINE = 0,
  TASK_SCOPED = 1,
  DEEP_CONTEXT = 2,
  RESOLUTION = 3,
}

interface ContextBlock {
  readonly name: string;
  readonly content: string;
  readonly stage: DisclosureStage;
  readonly tokenEstimate: number;
  readonly ttlTurns?: number;
  readonly trigger?: (state: ConversationState) => boolean;
}

interface ConversationState {
  currentTurn: number;
  taskType: string | null;
  userMessages: string[];
}

// ---------------------------------------------------------------------------
// ProgressiveDisclosureMiddleware
// ---------------------------------------------------------------------------

/**
 * Middleware that stages context injection based on conversation state.
 *
 * In the Vercel AI SDK, middleware transforms messages before they reach
 * the model. This middleware:
 * 1. Tracks conversation state from incoming messages
 * 2. Evaluates triggers to activate context blocks
 * 3. Assembles the context and prepends it as a system message
 *
 * Usage with Vercel AI SDK:
 *
 *   import { generateText } from 'ai';
 *   import { openai } from '@ai-sdk/openai';
 *
 *   const disclosure = createProgressiveDisclosure(8000);
 *   disclosure.register({ name: 'role', ... });
 *
 *   // Use as a message transformer before calling the model
 *   const processedMessages = disclosure.process(messages);
 *   const result = await generateText({
 *     model: openai('gpt-4o'),
 *     messages: processedMessages,
 *   });
 */
class ProgressiveDisclosureMiddleware {
  private readonly registered: ContextBlock[] = [];
  private activeBlocks: Array<{ block: ContextBlock; addedAtTurn: number }> =
    [];
  private state: ConversationState = {
    currentTurn: 0,
    taskType: null,
    userMessages: [],
  };
  private readonly maxTokens: number;

  constructor(maxTokens: number = 16_000) {
    this.maxTokens = maxTokens;
  }

  register(block: ContextBlock): void {
    this.registered.push(block);
  }

  setTask(taskType: string): void {
    this.state.taskType = taskType;
    this.promoteStage(DisclosureStage.TASK_SCOPED);
  }

  addDeepContext(
    name: string,
    content: string,
    tokenEstimate: number,
    ttlTurns: number = 5
  ): void {
    const block: ContextBlock = {
      name,
      content,
      stage: DisclosureStage.DEEP_CONTEXT,
      tokenEstimate,
      ttlTurns,
    };
    this.activeBlocks.push({ block, addedAtTurn: this.state.currentTurn });
  }

  summarizeAndPrune(summary: string): void {
    this.activeBlocks = this.activeBlocks.filter(
      ({ block }) => block.stage !== DisclosureStage.DEEP_CONTEXT
    );
    this.activeBlocks.push({
      block: {
        name: "context_summary",
        content: summary,
        stage: DisclosureStage.RESOLUTION,
        tokenEstimate: Math.ceil(summary.length / 4),
      },
      addedAtTurn: this.state.currentTurn,
    });
  }

  /**
   * Process messages: advance state, build context, return modified messages.
   *
   * This is the middleware function. It takes the raw messages from the
   * application and returns modified messages with the assembled context
   * prepended as a system message.
   */
  process(messages: readonly CoreMessage[]): CoreMessage[] {
    // Advance state for each user message
    for (const msg of messages) {
      if (msg.role === "user") {
        this.state.currentTurn++;
        this.state.userMessages.push(msg.content);
      }
    }

    this.evaluateTriggers();
    this.expireStaleBlocks();

    // Build assembled context
    const contextStr = this.buildContext();
    const systemMsg: CoreMessage = { role: "system", content: contextStr };

    // Replace existing system messages
    const nonSystem = messages.filter((m) => m.role !== "system");
    return [systemMsg, ...nonSystem];
  }

  get tokenUsage(): number {
    const baselineTokens = this.registered
      .filter((b) => b.stage === DisclosureStage.BASELINE)
      .reduce((sum, b) => sum + b.tokenEstimate, 0);
    const activeTokens = this.activeBlocks.reduce(
      (sum, { block }) => sum + block.tokenEstimate,
      0
    );
    return baselineTokens + activeTokens;
  }

  // -- Internal ------------------------------------------------------------

  private buildContext(): string {
    const baseline = this.registered.filter(
      (b) => b.stage === DisclosureStage.BASELINE
    );
    const allBlocks = [
      ...baseline.map((block) => ({ block, addedAtTurn: 0 })),
      ...this.activeBlocks,
    ];
    allBlocks.sort((a, b) => a.block.stage - b.block.stage);

    const parts: string[] = [];
    let tokenTotal = 0;
    for (const { block } of allBlocks) {
      if (tokenTotal + block.tokenEstimate > this.maxTokens) continue;
      parts.push(`<!-- ${block.name} -->\n${block.content}`);
      tokenTotal += block.tokenEstimate;
    }
    return parts.join("\n\n");
  }

  private promoteStage(stage: DisclosureStage): void {
    for (const block of this.registered) {
      if (block.stage !== stage) continue;
      if (this.activeBlocks.some(({ block: b }) => b.name === block.name))
        continue;
      this.activeBlocks.push({ block, addedAtTurn: this.state.currentTurn });
    }
  }

  private evaluateTriggers(): void {
    for (const block of this.registered) {
      if (!block.trigger || !block.trigger(this.state)) continue;
      if (this.activeBlocks.some(({ block: b }) => b.name === block.name))
        continue;
      this.activeBlocks.push({ block, addedAtTurn: this.state.currentTurn });
    }
  }

  private expireStaleBlocks(): void {
    this.activeBlocks = this.activeBlocks.filter(({ block, addedAtTurn }) => {
      if (block.ttlTurns === undefined) return true;
      return this.state.currentTurn - addedAtTurn < block.ttlTurns;
    });
  }
}

/** Factory function for creating the middleware. */
function createProgressiveDisclosure(
  maxTokens: number = 16_000
): ProgressiveDisclosureMiddleware {
  return new ProgressiveDisclosureMiddleware(maxTokens);
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const disclosure = createProgressiveDisclosure(8000);

  // Baseline
  disclosure.register({
    name: "role",
    content: "You are a senior code reviewer. Be concise and precise.",
    stage: DisclosureStage.BASELINE,
    tokenEstimate: 20,
  });
  disclosure.register({
    name: "output_format",
    content: "Respond in markdown. Use code fences with language tags.",
    stage: DisclosureStage.BASELINE,
    tokenEstimate: 15,
  });

  // Task-scoped
  disclosure.register({
    name: "review_guidelines",
    content: [
      "When reviewing code:",
      "1. Check for correctness first",
      "2. Then readability",
      "3. Then performance",
      "Flag security issues immediately.",
    ].join("\n"),
    stage: DisclosureStage.TASK_SCOPED,
    tokenEstimate: 40,
  });

  // Trigger-based
  disclosure.register({
    name: "db_schema",
    content:
      "Schema: users(id, email, name, created_at), orders(id, user_id, total, status)",
    stage: DisclosureStage.TASK_SCOPED,
    tokenEstimate: 30,
    trigger: (state) =>
      state.userMessages.some((msg) => msg.toLowerCase().includes("database")),
  });

  // Turn 1
  console.log("=== Turn 1 ===");
  const messages1 = disclosure.process([
    { role: "user", content: "Please review this pull request" },
  ]);
  disclosure.setTask("code_review");
  console.log(`Token usage: ${disclosure.tokenUsage}`);
  console.log(`System prompt: ${messages1[0].content.slice(0, 120)}...`);
  console.log();

  // Turn 2 (triggers db_schema)
  console.log("=== Turn 2 ===");
  const messages2 = disclosure.process([
    { role: "user", content: "Here's the diff for the database migration" },
  ]);
  console.log(`Token usage: ${disclosure.tokenUsage}`);
  console.log();

  // Add and prune deep context
  disclosure.addDeepContext(
    "migration_file",
    "ALTER TABLE users ADD COLUMN last_login TIMESTAMP;",
    20
  );
  console.log(`After deep context: ${disclosure.tokenUsage} tokens`);

  disclosure.summarizeAndPrune(
    "Reviewed DB migration adding last_login to users table."
  );
  console.log(`After prune: ${disclosure.tokenUsage} tokens`);
  console.log();

  // Final turn
  const messages3 = disclosure.process([
    { role: "user", content: "What's the final verdict?" },
  ]);
  console.log("=== Final context ===");
  console.log(messages3[0].content);
}

main();
