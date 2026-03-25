/**
 * Progressive Disclosure -- LangChain TypeScript Integration
 *
 * Custom Runnable that stages context injection based on conversation state.
 * Plugs into any LCEL chain to dynamically assemble the system prompt.
 *
 * Pattern: patterns/construction/progressive-disclosure.md
 */

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { Runnable, RunnableConfig } from "@langchain/core/runnables";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

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
  filesReferenced: Set<string>;
  toolsUsed: Set<string>;
  userMessages: string[];
}

function createConversationState(): ConversationState {
  return {
    currentTurn: 0,
    taskType: null,
    filesReferenced: new Set(),
    toolsUsed: new Set(),
    userMessages: [],
  };
}

// ---------------------------------------------------------------------------
// ProgressiveDisclosureRunnable
// ---------------------------------------------------------------------------

class ProgressiveDisclosureRunnable extends Runnable<
  BaseMessage[],
  BaseMessage[]
> {
  lc_namespace = ["context_engineering", "progressive_disclosure"];

  private readonly registered: ContextBlock[] = [];
  private activeBlocks: Array<{ block: ContextBlock; addedAtTurn: number }> =
    [];
  private state: ConversationState = createConversationState();
  private readonly maxTokens: number;

  constructor(maxTokens: number = 16_000) {
    super();
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
    const summaryBlock: ContextBlock = {
      name: "context_summary",
      content: summary,
      stage: DisclosureStage.RESOLUTION,
      tokenEstimate: Math.ceil(summary.length / 4),
    };
    this.activeBlocks.push({
      block: summaryBlock,
      addedAtTurn: this.state.currentTurn,
    });
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

  // -- Runnable interface --------------------------------------------------

  async invoke(
    input: BaseMessage[],
    options?: Partial<RunnableConfig>
  ): Promise<BaseMessage[]> {
    // Advance state for each user message
    for (const msg of input) {
      if (msg instanceof HumanMessage) {
        this.state.currentTurn++;
        const content =
          typeof msg.content === "string" ? msg.content : String(msg.content);
        this.state.userMessages.push(content);
      }
    }

    this.evaluateTriggers();
    this.expireStaleBlocks();

    // Build assembled context
    const contextStr = this.buildContext();
    const systemMsg = new SystemMessage(contextStr);

    // Replace any existing system messages
    const nonSystem = input.filter((m) => !(m instanceof SystemMessage));
    return [systemMsg, ...nonSystem];
  }

  // -- Internal context assembly -------------------------------------------

  private buildContext(): string {
    const baseline = this.registered.filter(
      (b) => b.stage === DisclosureStage.BASELINE
    );
    const allBlocks: Array<{ block: ContextBlock; addedAtTurn: number }> = [
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
      const alreadyActive = this.activeBlocks.some(
        ({ block: b }) => b.name === block.name
      );
      if (!alreadyActive) {
        this.activeBlocks.push({ block, addedAtTurn: this.state.currentTurn });
      }
    }
  }

  private evaluateTriggers(): void {
    for (const block of this.registered) {
      if (!block.trigger || !block.trigger(this.state)) continue;
      const alreadyActive = this.activeBlocks.some(
        ({ block: b }) => b.name === block.name
      );
      if (!alreadyActive) {
        this.activeBlocks.push({ block, addedAtTurn: this.state.currentTurn });
      }
    }
  }

  private expireStaleBlocks(): void {
    this.activeBlocks = this.activeBlocks.filter(({ block, addedAtTurn }) => {
      if (block.ttlTurns === undefined) return true;
      return this.state.currentTurn - addedAtTurn < block.ttlTurns;
    });
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const disclosure = new ProgressiveDisclosureRunnable(8000);

  // Baseline -- always present
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

  // Simulate conversation
  console.log("=== Turn 1 ===");
  disclosure
    .invoke([new HumanMessage("Please review this pull request")])
    .then((result) => {
      disclosure.setTask("code_review");
      console.log(`Token usage: ${disclosure.tokenUsage}`);
      console.log(`Messages out: ${result.length}`);

      console.log("\n=== Turn 2 (triggers db_schema) ===");
      return disclosure.invoke([
        new HumanMessage("Here's the diff for the database migration"),
      ]);
    })
    .then(() => {
      console.log(`Token usage: ${disclosure.tokenUsage}`);

      disclosure.addDeepContext(
        "migration_file",
        "ALTER TABLE users ADD COLUMN last_login TIMESTAMP;",
        20
      );
      console.log(
        `\nToken usage after deep context: ${disclosure.tokenUsage}`
      );

      disclosure.summarizeAndPrune(
        "Previously reviewed a DB migration adding last_login to users table."
      );
      console.log(`Token usage after prune: ${disclosure.tokenUsage}`);

      return disclosure.invoke([
        new HumanMessage("What's the final verdict?"),
      ]);
    })
    .then((result) => {
      console.log("\n=== Final assembled context ===");
      const sysMsg = result.find((m) => m instanceof SystemMessage);
      if (sysMsg) {
        console.log(
          typeof sysMsg.content === "string"
            ? sysMsg.content
            : String(sysMsg.content)
        );
      }
    });
}

main();
