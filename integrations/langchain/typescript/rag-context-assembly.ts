/**
 * RAG Context Assembly -- LangChain TypeScript Integration
 *
 * Custom retriever chain with re-ranking, deduplication, and token budgeting.
 * Uses LangChain's Document type and Runnable interface for composability.
 *
 * Pattern: patterns/retrieval/rag-context-assembly.md
 */

import { Document } from "@langchain/core/documents";
import { RunnableLambda } from "@langchain/core/runnables";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AssemblyConfig {
  readonly tokenBudget: number;
  readonly similarityThreshold: number;
  readonly minRelevanceScore: number;
}

interface AssembledContext {
  readonly documents: readonly Document[];
  readonly totalTokens: number;
  readonly droppedCount: number;
  readonly contextBlock: string;
}

const DEFAULT_CONFIG: AssemblyConfig = {
  tokenBudget: 3000,
  similarityThreshold: 0.85,
  minRelevanceScore: 0.3,
};

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function shingle(text: string, n: number = 3): Set<string> {
  const words = text.toLowerCase().split(/\s+/);
  if (words.length < n) return new Set([text.toLowerCase()]);
  const shingles = new Set<string>();
  for (let i = 0; i <= words.length - n; i++) {
    shingles.add(words.slice(i, i + n).join(" "));
  }
  return shingles;
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  const intersection = new Set([...a].filter((x) => b.has(x)));
  const union = new Set([...a, ...b]);
  return union.size === 0 ? 0 : intersection.size / union.size;
}

function rerankDocuments(query: string, docs: Document[]): Document[] {
  /**
   * Re-rank documents by term overlap with the query.
   *
   * Production: Replace with a cross-encoder or LangChain's built-in reranker:
   *   import { CohereRerank } from "@langchain/cohere";
   */
  const queryTerms = new Set(query.toLowerCase().split(/\s+/));

  return docs
    .map((doc) => {
      const contentTerms = new Set(doc.pageContent.toLowerCase().split(/\s+/));
      const overlap =
        [...queryTerms].filter((t) => contentTerms.has(t)).length /
        Math.max(queryTerms.size, 1);
      const originalScore = (doc.metadata?.score as number) ?? 0.5;
      const newScore = originalScore * 0.4 + overlap * 0.6;

      return new Document({
        pageContent: doc.pageContent,
        metadata: { ...doc.metadata, score: newScore },
      });
    })
    .sort(
      (a, b) =>
        ((b.metadata?.score as number) ?? 0) -
        ((a.metadata?.score as number) ?? 0)
    );
}

function deduplicateDocuments(
  docs: Document[],
  threshold: number
): Document[] {
  if (docs.length === 0) return [];

  const sorted = [...docs].sort(
    (a, b) =>
      ((b.metadata?.score as number) ?? 0) -
      ((a.metadata?.score as number) ?? 0)
  );

  const kept: Document[] = [];
  const seenShingles: Set<string>[] = [];

  for (const doc of sorted) {
    const docShingles = shingle(doc.pageContent);
    const isDuplicate = seenShingles.some(
      (existing) => jaccardSimilarity(docShingles, existing) >= threshold
    );

    if (!isDuplicate) {
      kept.push(doc);
      seenShingles.push(docShingles);
    }
  }

  return kept;
}

function applyTokenBudget(
  docs: Document[],
  budget: number
): { selected: Document[]; totalTokens: number; dropped: number } {
  const sorted = [...docs].sort(
    (a, b) =>
      ((b.metadata?.score as number) ?? 0) -
      ((a.metadata?.score as number) ?? 0)
  );

  const selected: Document[] = [];
  let totalTokens = 0;
  let dropped = 0;

  for (const doc of sorted) {
    const cost = estimateTokens(doc.pageContent) + 20;
    if (totalTokens + cost > budget) {
      dropped++;
      continue;
    }
    selected.push(doc);
    totalTokens += cost;
  }

  return { selected, totalTokens, dropped };
}

function formatContextBlock(docs: readonly Document[]): string {
  if (docs.length === 0) {
    return "<retrieved_context>\nNo relevant information found.\n</retrieved_context>";
  }

  const sections = docs.map((doc, i) => {
    const source = doc.metadata?.source ?? "unknown";
    const section = doc.metadata?.section ?? "";
    let header = `[Source ${i + 1}: ${source}`;
    if (section) header += ` > ${section}`;
    header += "]";
    return `${header}\n${doc.pageContent}`;
  });

  return `<retrieved_context>\n${sections.join("\n\n---\n\n")}\n</retrieved_context>`;
}

// ---------------------------------------------------------------------------
// LCEL chain builder
// ---------------------------------------------------------------------------

type AssemblyInput = { query: string; documents: Document[] };

function buildRagAssemblyChain(config: Partial<AssemblyConfig> = {}) {
  const cfg: AssemblyConfig = { ...DEFAULT_CONFIG, ...config };

  return new RunnableLambda<AssemblyInput, AssembledContext>({
    func: async (input: AssemblyInput): Promise<AssembledContext> => {
      const { query, documents } = input;

      // Stage 1: Re-rank
      const reranked = rerankDocuments(query, documents);

      // Stage 2: Filter low-relevance
      const filtered = reranked.filter(
        (d) => ((d.metadata?.score as number) ?? 0) >= cfg.minRelevanceScore
      );

      // Stage 3: Deduplicate
      const deduped = deduplicateDocuments(filtered, cfg.similarityThreshold);

      // Stage 4: Apply token budget
      const { selected, totalTokens, dropped } = applyTokenBudget(
        deduped,
        cfg.tokenBudget
      );

      // Stage 5: Format
      const contextBlock = formatContextBlock(selected);

      return {
        documents: selected,
        totalTokens,
        droppedCount: dropped,
        contextBlock,
      };
    },
  });
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const rawDocs: Document[] = [
    new Document({
      pageContent:
        "Refunds are processed within 5-10 business days after approval. " +
        "Contact support with your order ID to initiate a refund request.",
      metadata: {
        source: "billing-faq.md",
        section: "Refund Policy",
        score: 0.89,
      },
    }),
    new Document({
      pageContent:
        "Refund requests must include the order ID. Refunds take 5-10 " +
        "business days to process after the request is approved by support.",
      metadata: {
        source: "support-guide.md",
        section: "Processing Refunds",
        score: 0.87,
      },
    }),
    new Document({
      pageContent:
        "Invoices are generated on the 1st of each month and sent to " +
        "the billing email on file.",
      metadata: {
        source: "billing-faq.md",
        section: "Invoice Schedule",
        score: 0.62,
      },
    }),
    new Document({
      pageContent:
        "Our office hours are Monday through Friday, 9 AM to 5 PM EST.",
      metadata: {
        source: "general-info.md",
        section: "Contact Us",
        score: 0.25,
      },
    }),
  ];

  const chain = buildRagAssemblyChain({
    tokenBudget: 2000,
    minRelevanceScore: 0.2,
  });

  const result = await chain.invoke({
    query: "How do I get a refund?",
    documents: rawDocs,
  });

  console.log(`Chunks used: ${result.documents.length}`);
  console.log(`Tokens used: ${result.totalTokens}`);
  console.log(`Chunks dropped: ${result.droppedCount}`);
  console.log();
  console.log(result.contextBlock);
}

main();
