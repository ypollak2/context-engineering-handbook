/**
 * Instruction Adherence Benchmark.
 *
 * Tests whether the LLM follows specific behavioral rules defined in the
 * system prompt. Measures compliance rate across diverse queries.
 */

import type { LLMClient } from "../utils/llm-client.js";
import { countTokens } from "../utils/metrics.js";

interface Rule {
  readonly id: string;
  readonly description: string;
  readonly instruction: string;
  readonly checker: (response: string) => boolean;
}

function checkJsonFormat(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    return typeof parsed === "object" && parsed !== null && "response" in parsed;
  } catch {
    return false;
  }
}

function checkNoSimply(response: string): boolean {
  return !response.toLowerCase().includes("simply");
}

function checkConfidenceScore(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    const conf = parsed.confidence;
    return typeof conf === "number" && conf >= 0 && conf <= 1;
  } catch {
    return false;
  }
}

function checkMaxSentences(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    const text: string = parsed.response ?? "";
    const sentences = text
      .split(/[.!?]+/)
      .filter((s: string) => s.trim().length > 0);
    return sentences.length <= 3;
  } catch {
    return false;
  }
}

function checkNoApology(response: string): boolean {
  const lower = response.toLowerCase();
  const phrases = ["i apologize", "sorry", "i'm sorry", "apologies"];
  return !phrases.some((p) => lower.includes(p));
}

function checkUppercaseTerms(response: string): boolean {
  const techTerms = [
    "python", "javascript", "typescript", "java", "http", "https",
    "tcp", "udp", "dns", "sql", "rest", "api", "json", "html", "css",
  ];
  const lower = response.toLowerCase();
  const found = techTerms.filter((t) => lower.includes(t));
  if (found.length === 0) return true;
  return found.some((t) => response.includes(t.toUpperCase()));
}

function checkNumberedSteps(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    const respText: string = String(parsed.response ?? "").toLowerCase();
    const processIndicators = [
      "first", "then", "next", "step", "process", "procedure",
    ];
    const hasProcessLanguage = processIndicators.some((i) =>
      respText.includes(i)
    );
    if (!hasProcessLanguage) return true;
    return "steps" in parsed && Array.isArray(parsed.steps);
  } catch {
    return false;
  }
}

function checkSourceCitation(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    const source = parsed.source;
    return ["training_data", "reasoning", "uncertain"].includes(source);
  } catch {
    return false;
  }
}

function checkNoMarkdown(response: string): boolean {
  const patterns = ["**", "##", "```", "- ", "* "];
  return !patterns.some((p) => response.includes(p));
}

function checkWordCount(response: string): boolean {
  try {
    const parsed = JSON.parse(response.trim());
    const claimed = parsed.word_count;
    if (typeof claimed !== "number") return false;
    const text: string = parsed.response ?? "";
    const actual = text.split(/\s+/).filter((w: string) => w.length > 0).length;
    return Math.abs(claimed - actual) <= Math.max(2, actual * 0.2);
  } catch {
    return false;
  }
}

const RULES: readonly Rule[] = [
  {
    id: "json_format",
    description: "Always respond in valid JSON",
    instruction:
      "Always format your entire response as valid JSON with a 'response' key.",
    checker: checkJsonFormat,
  },
  {
    id: "no_simply",
    description: "Never use the word 'simply'",
    instruction: "Never use the word 'simply' in your response.",
    checker: checkNoSimply,
  },
  {
    id: "confidence_score",
    description: "Include confidence scores",
    instruction:
      "Include a 'confidence' field (0.0-1.0) in your JSON response.",
    checker: checkConfidenceScore,
  },
  {
    id: "max_sentences",
    description: "Maximum 3 sentences in the response value",
    instruction: "Keep the 'response' value to 3 sentences or fewer.",
    checker: checkMaxSentences,
  },
  {
    id: "no_apology",
    description: "Never apologize",
    instruction:
      "Never begin with or include phrases like 'I apologize', 'Sorry', or 'I'm sorry'.",
    checker: checkNoApology,
  },
  {
    id: "uppercase_key_terms",
    description: "Key technical terms in uppercase",
    instruction:
      "Write key technical terms (programming languages, frameworks, protocols) in ALL CAPS.",
    checker: checkUppercaseTerms,
  },
  {
    id: "numbered_steps",
    description: "Use numbered steps for procedures",
    instruction:
      "When describing a process or procedure, use numbered steps in a 'steps' array.",
    checker: checkNumberedSteps,
  },
  {
    id: "source_citation",
    description: "Cite knowledge source",
    instruction:
      "Include a 'source' field indicating whether the answer is from 'training_data', 'reasoning', or 'uncertain'.",
    checker: checkSourceCitation,
  },
  {
    id: "no_markdown",
    description: "No markdown formatting",
    instruction:
      "Do not use any markdown formatting (no **, no ##, no `, no -).",
    checker: checkNoMarkdown,
  },
  {
    id: "word_count",
    description: "Include word count",
    instruction:
      "Include a 'word_count' field with the number of words in the 'response' value.",
    checker: checkWordCount,
  },
];

const QUERIES: readonly string[] = [
  "What is a REST API?",
  "How do I sort a list in Python?",
  "Explain the difference between TCP and UDP.",
  "What causes a stack overflow error?",
  "How does garbage collection work?",
  "What is the CAP theorem?",
  "Explain dependency injection.",
  "How do database indexes improve performance?",
  "What is the difference between authentication and authorization?",
  "How does HTTPS work?",
  "What is eventual consistency?",
  "Explain the observer design pattern.",
  "How do you handle race conditions?",
  "What is a memory leak?",
  "Explain the concept of immutability.",
  "How does DNS resolution work?",
  "What is the difference between threads and processes?",
  "Explain what a closure is in programming.",
  "How do load balancers distribute traffic?",
  "What is the purpose of a message queue?",
];

interface RuleResult {
  readonly ruleId: string;
  readonly description: string;
  readonly adherenceRate: number;
  readonly passed: number;
  readonly total: number;
}

export interface InstructionAdherenceResult {
  readonly benchmark: "instruction_adherence";
  readonly model: string;
  readonly overallCompliance: number;
  readonly systemPromptTokens: number;
  readonly perRule: readonly RuleResult[];
  readonly perQuerySummary: {
    readonly totalQueries: number;
    readonly avgCompliance: number;
  };
}

function buildSystemPrompt(): string {
  const instructions = RULES.map(
    (rule, i) => `${i + 1}. ${rule.instruction}`
  ).join("\n");
  return (
    "You are a helpful assistant. You MUST follow ALL of these rules " +
    "in every response:\n\n" +
    instructions +
    "\n\nFailure to follow any rule is considered a violation. " +
    "Every response must comply with ALL rules simultaneously."
  );
}

export async function runInstructionAdherence(
  client: LLMClient
): Promise<InstructionAdherenceResult> {
  const systemPrompt = buildSystemPrompt();
  const systemTokens = countTokens(systemPrompt);

  const rulePasses: Record<string, boolean[]> = {};
  for (const rule of RULES) {
    rulePasses[rule.id] = [];
  }

  const queryCompliances: number[] = [];

  for (const query of QUERIES) {
    const response = await client.complete(
      [{ role: "user", content: query }],
      systemPrompt
    );

    let rulesPassed = 0;
    for (const rule of RULES) {
      const passed = rule.checker(response.content);
      rulePasses[rule.id].push(passed);
      if (passed) rulesPassed++;
    }
    queryCompliances.push(rulesPassed / RULES.length);
  }

  const perRule: RuleResult[] = RULES.map((rule) => {
    const passes = rulePasses[rule.id];
    const passed = passes.filter(Boolean).length;
    return {
      ruleId: rule.id,
      description: rule.description,
      adherenceRate:
        Math.round((passed / passes.length) * 10000) / 10000,
      passed,
      total: passes.length,
    };
  });

  const totalChecks = RULES.length * QUERIES.length;
  const totalPassed = perRule.reduce((sum, r) => sum + r.passed, 0);

  return {
    benchmark: "instruction_adherence",
    model: client.model,
    overallCompliance:
      Math.round((totalPassed / totalChecks) * 10000) / 10000,
    systemPromptTokens: systemTokens,
    perRule,
    perQuerySummary: {
      totalQueries: QUERIES.length,
      avgCompliance:
        Math.round(
          (queryCompliances.reduce((a, b) => a + b, 0) /
            queryCompliances.length) *
            10000
        ) / 10000,
    },
  };
}
