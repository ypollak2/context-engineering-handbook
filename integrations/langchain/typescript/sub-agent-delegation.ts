/**
 * Sub-Agent Delegation -- LangChain TypeScript Integration
 *
 * Parent-child agent composition with isolated contexts per sub-agent.
 * Each child runs in its own LLM call with a purpose-built system prompt,
 * and only the concise result is returned to the parent.
 *
 * Pattern: patterns/isolation/sub-agent-delegation.md
 */

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TaskStatus = "pending" | "running" | "completed" | "failed";

interface SubTask {
  readonly taskId: string;
  readonly description: string;
  readonly relevantFiles: readonly string[];
  readonly constraints: readonly string[];
  readonly parentDecisions: readonly string[];
  readonly outputFormat: string;
  readonly maxTokens: number;
  readonly timeoutMs: number;
}

interface SubTaskResult {
  readonly taskId: string;
  readonly status: TaskStatus;
  readonly result: string;
  readonly tokenCount: number;
  readonly error?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createSubTask(
  taskId: string,
  description: string,
  options: {
    relevantFiles?: string[];
    constraints?: string[];
    parentDecisions?: string[];
    outputFormat?: string;
    maxTokens?: number;
    timeoutMs?: number;
  } = {}
): SubTask {
  return {
    taskId,
    description,
    relevantFiles: options.relevantFiles ?? [],
    constraints: options.constraints ?? [],
    parentDecisions: options.parentDecisions ?? [],
    outputFormat:
      options.outputFormat ??
      "Provide a concise summary of findings (max 200 words).",
    maxTokens: options.maxTokens ?? 50_000,
    timeoutMs: options.timeoutMs ?? 120_000,
  };
}

function buildDelegationPrompt(task: SubTask): string {
  const sections = [`## Task\n${task.description}`];

  if (task.relevantFiles.length > 0) {
    sections.push(
      `## Relevant Files\n${task.relevantFiles.map((f) => `- ${f}`).join("\n")}`
    );
  }

  if (task.constraints.length > 0) {
    sections.push(
      `## Constraints\n${task.constraints.map((c) => `- ${c}`).join("\n")}`
    );
  }

  if (task.parentDecisions.length > 0) {
    sections.push(
      `## Prior Decisions (do not revisit)\n${task.parentDecisions.map((d) => `- ${d}`).join("\n")}`
    );
  }

  sections.push(`## Expected Output Format\n${task.outputFormat}`);
  return sections.join("\n\n");
}

// ---------------------------------------------------------------------------
// SubAgentOrchestrator
// ---------------------------------------------------------------------------

class SubAgentOrchestrator {
  private readonly llm: Runnable | null;
  private readonly results: Map<string, SubTaskResult> = new Map();

  constructor(llm: Runnable | null = null) {
    this.llm = llm;
  }

  async delegate(task: SubTask): Promise<SubTaskResult> {
    const systemPrompt = buildDelegationPrompt(task);

    try {
      let resultText: string;

      if (this.llm) {
        const messages = [
          new SystemMessage(systemPrompt),
          new HumanMessage("Execute the task described above."),
        ];
        const response = await this.llm.invoke(messages);
        resultText =
          typeof response.content === "string"
            ? response.content
            : String(response.content);
      } else {
        resultText = this.simulateChild(task);
      }

      const result: SubTaskResult = {
        taskId: task.taskId,
        status: "completed",
        result: resultText,
        tokenCount: Math.ceil(resultText.split(/\s+/).length * 1.5),
      };

      this.results.set(task.taskId, result);
      return result;
    } catch (err) {
      const result: SubTaskResult = {
        taskId: task.taskId,
        status: "failed",
        result: "",
        tokenCount: 0,
        error: err instanceof Error ? err.message : String(err),
      };

      this.results.set(task.taskId, result);
      return result;
    }
  }

  async delegateParallel(tasks: SubTask[]): Promise<SubTaskResult[]> {
    return Promise.all(tasks.map((task) => this.delegate(task)));
  }

  formatResultsForParent(results: readonly SubTaskResult[]): string {
    return results
      .map((r) => {
        if (r.status === "completed") {
          return `### Sub-task: ${r.taskId}\n**Status**: completed\n**Result**: ${r.result}`;
        }
        return `### Sub-task: ${r.taskId}\n**Status**: ${r.status}\n**Error**: ${r.error}`;
      })
      .join("\n\n");
  }

  get totalChildTokens(): number {
    let total = 0;
    for (const result of this.results.values()) {
      total += result.tokenCount;
    }
    return total;
  }

  private simulateChild(task: SubTask): string {
    return [
      `[Simulated child result for '${task.taskId}']`,
      `Analyzed: ${task.description}`,
      `Files checked: ${task.relevantFiles.join(", ") || "none specified"}`,
      `Recommendation: Based on analysis, proceed with the standard approach.`,
      `Confidence: high`,
    ].join("\n");
  }
}

/**
 * Create a LangChain prompt template for a child agent.
 * Use with create_tool_calling_agent for tool-using children.
 */
function createChildAgentPrompt(task: SubTask): ChatPromptTemplate {
  const systemPrompt = buildDelegationPrompt(task);
  return ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const orchestrator = new SubAgentOrchestrator(null);

  const researchTasks = [
    createSubTask(
      "research-approach",
      "Research implementation approaches for JWT authentication",
      {
        relevantFiles: ["src/auth/", "docs/architecture.md"],
        constraints: [
          "Do not modify any files",
          "Focus on existing codebase patterns",
        ],
        parentDecisions: ["Using Python/FastAPI stack"],
        outputFormat:
          "Recommend one approach with 3 bullet points of justification.",
      }
    ),
    createSubTask(
      "research-tests",
      "Analyze existing test patterns and coverage for the auth module",
      {
        relevantFiles: ["tests/auth/"],
        constraints: ["Read-only analysis"],
        outputFormat:
          "List existing test patterns and coverage gaps (max 150 words).",
      }
    ),
    createSubTask(
      "research-security",
      "Review security implications of JWT implementation",
      {
        relevantFiles: ["src/auth/", "src/middleware/"],
        constraints: [
          "Check for OWASP Top 10 relevance",
          "Do not implement fixes",
        ],
        outputFormat: "List security considerations as bullet points.",
      }
    ),
  ];

  const results = await orchestrator.delegateParallel(researchTasks);
  const summary = orchestrator.formatResultsForParent(results);

  console.log("=== Parent Agent Receives ===");
  console.log(`Total child tokens used: ${orchestrator.totalChildTokens}`);
  console.log(`Results count: ${results.length}`);
  console.log();
  console.log(summary);
  console.log();

  const parentContextTokens = Math.ceil(summary.length / 4);
  const estimatedChildWork = orchestrator.totalChildTokens * 5;
  console.log(`Parent context cost: ~${parentContextTokens} tokens`);
  console.log(`Estimated undelegated cost: ~${estimatedChildWork} tokens`);
  console.log(
    `Savings: ~${estimatedChildWork - parentContextTokens} tokens kept out of parent context`
  );
}

main();
