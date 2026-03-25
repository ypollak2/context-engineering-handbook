/**
 * Semantic Tool Selection -- LangChain TypeScript Integration
 *
 * Dynamic tool filtering by embedding similarity before agent execution.
 * Reduces the number of tool descriptions in the prompt to only the most
 * relevant ones for each turn.
 *
 * Pattern: patterns/retrieval/semantic-tool-selection.md
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ScoredTool {
  readonly tool: DynamicStructuredTool;
  readonly score: number;
}

interface SelectorConfig {
  readonly topK: number;
  readonly minScore: number;
  readonly embeddingDim: number;
}

const DEFAULT_CONFIG: SelectorConfig = {
  topK: 5,
  minScore: 0.3,
  embeddingDim: 384,
};

// ---------------------------------------------------------------------------
// Embedding helper (replace with real embedding model in production)
// ---------------------------------------------------------------------------

function pseudoEmbed(text: string, dim: number): number[] {
  let seed = 0;
  for (let i = 0; i < text.length; i++) {
    seed = ((seed << 5) - seed + text.charCodeAt(i)) | 0;
  }

  const vec: number[] = [];
  for (let i = 0; i < dim; i++) {
    seed = (seed * 1664525 + 1013904223) | 0;
    vec.push((seed / 2147483647) * 2 - 1);
  }

  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  return vec.map((v) => v / norm);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ---------------------------------------------------------------------------
// SemanticToolSelector
// ---------------------------------------------------------------------------

class SemanticToolSelector {
  private readonly config: SelectorConfig;
  private tools: DynamicStructuredTool[] = [];
  private embeddings: number[][] = [];
  private pinnedNames: Set<string>;

  constructor(
    config: Partial<SelectorConfig> = {},
    pinnedTools: string[] = []
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.pinnedNames = new Set(pinnedTools);
  }

  registerTools(tools: DynamicStructuredTool[]): void {
    this.tools = tools;
    this.embeddings = tools.map((t) =>
      pseudoEmbed(`${t.name}: ${t.description}`, this.config.embeddingDim)
    );
  }

  select(query: string): ScoredTool[] {
    if (this.tools.length === 0) return [];

    const queryEmbedding = pseudoEmbed(query, this.config.embeddingDim);

    const pinned: ScoredTool[] = [];
    const scored: ScoredTool[] = [];

    for (let i = 0; i < this.tools.length; i++) {
      const tool = this.tools[i];
      const similarity = cosineSimilarity(queryEmbedding, this.embeddings[i]);
      const st: ScoredTool = { tool, score: similarity };

      if (this.pinnedNames.has(tool.name)) {
        pinned.push(st);
      } else {
        scored.push(st);
      }
    }

    const selected = scored
      .sort((a, b) => b.score - a.score)
      .filter((s) => s.score >= this.config.minScore)
      .slice(0, this.config.topK);

    return [...pinned, ...selected];
  }

  selectTools(query: string): DynamicStructuredTool[] {
    return this.select(query).map((s) => s.tool);
  }

  formatSelectionReport(scored: ScoredTool[]): string {
    const lines = [
      `Selected ${scored.length} tools (from ${this.tools.length} registered):`,
    ];
    for (const st of scored) {
      const pin = this.pinnedNames.has(st.tool.name) ? " [PINNED]" : "";
      lines.push(`  ${st.tool.name} (score: ${st.score.toFixed(3)})${pin}`);
    }
    return lines.join("\n");
  }
}

// ---------------------------------------------------------------------------
// Demo tools
// ---------------------------------------------------------------------------

const searchKnowledgeBase = new DynamicStructuredTool({
  name: "search_knowledge_base",
  description:
    "Search the company knowledge base for articles, FAQs, and documentation",
  schema: z.object({
    query: z.string(),
    category: z.string().optional(),
  }),
  func: async ({ query }) => `Results for '${query}'`,
});

const createTicket = new DynamicStructuredTool({
  name: "create_ticket",
  description:
    "Create a support ticket for issues that need human agent follow-up",
  schema: z.object({
    title: z.string(),
    description: z.string(),
    priority: z.string().optional(),
  }),
  func: async ({ title }) => `Created ticket: ${title}`,
});

const getOrderStatus = new DynamicStructuredTool({
  name: "get_order_status",
  description: "Look up the current status of a customer order by order ID",
  schema: z.object({ order_id: z.string() }),
  func: async ({ order_id }) => `Order ${order_id}: shipped`,
});

const processRefund = new DynamicStructuredTool({
  name: "process_refund",
  description: "Initiate a refund for a specific order or transaction",
  schema: z.object({
    order_id: z.string(),
    reason: z.string(),
    amount: z.number().optional(),
  }),
  func: async ({ order_id }) => `Refund initiated for ${order_id}`,
});

const updateAccountEmail = new DynamicStructuredTool({
  name: "update_account_email",
  description: "Update the email address on a customer account",
  schema: z.object({
    customer_id: z.string(),
    new_email: z.string(),
  }),
  func: async ({ customer_id }) => `Email updated for ${customer_id}`,
});

const scheduleCallback = new DynamicStructuredTool({
  name: "schedule_callback",
  description:
    "Schedule a phone callback from a human agent at a specific time",
  schema: z.object({
    phone: z.string(),
    preferred_time: z.string(),
  }),
  func: async ({ phone, preferred_time }) =>
    `Callback scheduled for ${phone} at ${preferred_time}`,
});

const sendInvoice = new DynamicStructuredTool({
  name: "send_invoice",
  description: "Send or resend an invoice to the customer's billing email",
  schema: z.object({ invoice_id: z.string() }),
  func: async ({ invoice_id }) => `Invoice ${invoice_id} sent`,
});

const finalAnswer = new DynamicStructuredTool({
  name: "final_answer",
  description: "Provide the final response to the user's question",
  schema: z.object({ answer: z.string() }),
  func: async ({ answer }) => answer,
});

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const allTools = [
    searchKnowledgeBase,
    createTicket,
    getOrderStatus,
    processRefund,
    updateAccountEmail,
    scheduleCallback,
    sendInvoice,
    finalAnswer,
  ];

  const selector = new SemanticToolSelector(
    { topK: 3, minScore: 0.1 },
    ["final_answer"]
  );
  selector.registerTools(allTools);

  const queries = [
    "I need a refund for order #12345",
    "What's the status of my order?",
    "Can you update my email address?",
    "I need to speak with a human agent",
  ];

  for (const query of queries) {
    const scored = selector.select(query);
    console.log(`Query: ${query}`);
    console.log(selector.formatSelectionReport(scored));
    console.log();
  }
}

main();
