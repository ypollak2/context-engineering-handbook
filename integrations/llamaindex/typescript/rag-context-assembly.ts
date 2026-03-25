/**
 * RAG Context Assembly -- LlamaIndex TypeScript Integration
 *
 * Implements the assembly pipeline as a series of node postprocessor
 * functions that mirror LlamaIndex's NodePostprocessor pattern.
 *
 * Pattern: patterns/retrieval/rag-context-assembly.md
 */

// ---------------------------------------------------------------------------
// Types (aligned with LlamaIndex's NodeWithScore)
// ---------------------------------------------------------------------------

interface NodeMetadata {
  readonly source?: string;
  readonly section?: string;
  readonly score?: number;
  readonly citation?: string;
  readonly [key: string]: unknown;
}

interface TextNode {
  readonly id: string;
  readonly text: string;
  readonly metadata: NodeMetadata;
}

interface NodeWithScore {
  readonly node: TextNode;
  readonly score: number;
}

interface AssemblyConfig {
  readonly tokenBudget: number;
  readonly similarityThreshold: number;
  readonly minRelevanceScore: number;
}

const DEFAULT_CONFIG: AssemblyConfig = {
  tokenBudget: 3000,
  similarityThreshold: 0.85,
  minRelevanceScore: 0.3,
};

// ---------------------------------------------------------------------------
// Postprocessor functions
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

/**
 * Stage 1: Re-rank nodes by term overlap with query.
 * In production, use LlamaIndex's CohereRerank or SentenceTransformerRerank.
 */
function rerankNodes(query: string, nodes: NodeWithScore[]): NodeWithScore[] {
  const queryTerms = new Set(query.toLowerCase().split(/\s+/));

  return nodes
    .map((nws) => {
      const contentTerms = new Set(nws.node.text.toLowerCase().split(/\s+/));
      const overlap =
        [...queryTerms].filter((t) => contentTerms.has(t)).length /
        Math.max(queryTerms.size, 1);
      const newScore = nws.score * 0.4 + overlap * 0.6;
      return { node: nws.node, score: newScore };
    })
    .sort((a, b) => b.score - a.score);
}

/** Stage 2: Remove near-duplicate nodes. */
function deduplicateNodes(
  nodes: NodeWithScore[],
  threshold: number
): NodeWithScore[] {
  const sorted = [...nodes].sort((a, b) => b.score - a.score);
  const kept: NodeWithScore[] = [];
  const seenShingles: Set<string>[] = [];

  for (const nws of sorted) {
    const nodeShingles = shingle(nws.node.text);
    const isDuplicate = seenShingles.some(
      (existing) => jaccardSimilarity(nodeShingles, existing) >= threshold
    );
    if (!isDuplicate) {
      kept.push(nws);
      seenShingles.push(nodeShingles);
    }
  }

  return kept;
}

/** Stage 3: Enforce token budget. */
function applyTokenBudget(
  nodes: NodeWithScore[],
  budget: number
): { selected: NodeWithScore[]; totalTokens: number; dropped: number } {
  const sorted = [...nodes].sort((a, b) => b.score - a.score);
  const selected: NodeWithScore[] = [];
  let totalTokens = 0;
  let dropped = 0;

  for (const nws of sorted) {
    const cost = estimateTokens(nws.node.text) + 20;
    if (totalTokens + cost > budget) {
      dropped++;
      continue;
    }
    selected.push(nws);
    totalTokens += cost;
  }

  return { selected, totalTokens, dropped };
}

/** Stage 4: Add source attribution to node metadata. */
function addAttribution(nodes: NodeWithScore[]): NodeWithScore[] {
  return nodes.map((nws, i) => {
    const source = nws.node.metadata.source ?? "unknown";
    const section = nws.node.metadata.section ?? "";
    let citation = `[Source ${i + 1}: ${source}`;
    if (section) citation += ` > ${section}`;
    citation += "]";

    return {
      node: {
        ...nws.node,
        metadata: { ...nws.node.metadata, citation, source_index: i + 1 },
      },
      score: nws.score,
    };
  });
}

/** Format processed nodes into a context block string. */
function formatContextBlock(nodes: readonly NodeWithScore[]): string {
  if (nodes.length === 0) {
    return "<retrieved_context>\nNo relevant information found.\n</retrieved_context>";
  }

  const sections = nodes.map((nws) => {
    const citation = nws.node.metadata.citation ?? "[Source]";
    return `${citation}\n${nws.node.text}`;
  });

  return `<retrieved_context>\n${sections.join("\n\n---\n\n")}\n</retrieved_context>`;
}

// ---------------------------------------------------------------------------
// Full assembly pipeline
// ---------------------------------------------------------------------------

interface AssembledContext {
  readonly nodes: readonly NodeWithScore[];
  readonly totalTokens: number;
  readonly droppedCount: number;
  readonly contextBlock: string;
}

function assembleContext(
  query: string,
  nodes: NodeWithScore[],
  config: Partial<AssemblyConfig> = {}
): AssembledContext {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Pipeline stages
  let processed = rerankNodes(query, nodes);
  processed = processed.filter((n) => n.score >= cfg.minRelevanceScore);
  processed = deduplicateNodes(processed, cfg.similarityThreshold);
  const { selected, totalTokens, dropped } = applyTokenBudget(
    processed,
    cfg.tokenBudget
  );
  const attributed = addAttribution(selected);
  const contextBlock = formatContextBlock(attributed);

  return {
    nodes: attributed,
    totalTokens,
    droppedCount: dropped,
    contextBlock,
  };
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const rawNodes: NodeWithScore[] = [
    {
      node: {
        id: "n1",
        text:
          "Refunds are processed within 5-10 business days after approval. " +
          "Contact support with your order ID to initiate a refund request.",
        metadata: { source: "billing-faq.md", section: "Refund Policy" },
      },
      score: 0.89,
    },
    {
      node: {
        id: "n2",
        text:
          "Refund requests must include the order ID. Refunds take 5-10 " +
          "business days to process after the request is approved by support.",
        metadata: {
          source: "support-guide.md",
          section: "Processing Refunds",
        },
      },
      score: 0.87,
    },
    {
      node: {
        id: "n3",
        text: "Invoices are generated on the 1st of each month and sent to the billing email on file.",
        metadata: { source: "billing-faq.md", section: "Invoice Schedule" },
      },
      score: 0.62,
    },
    {
      node: {
        id: "n4",
        text: "Our office hours are Monday through Friday, 9 AM to 5 PM EST.",
        metadata: { source: "general-info.md", section: "Contact Us" },
      },
      score: 0.25,
    },
  ];

  const result = assembleContext("How do I get a refund?", rawNodes, {
    tokenBudget: 2000,
    minRelevanceScore: 0.2,
  });

  console.log(`Nodes used: ${result.nodes.length}`);
  console.log(`Tokens used: ${result.totalTokens}`);
  console.log(`Nodes dropped: ${result.droppedCount}`);
  console.log();
  console.log(result.contextBlock);
}

main();
