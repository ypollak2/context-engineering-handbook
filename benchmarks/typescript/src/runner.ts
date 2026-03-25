#!/usr/bin/env node
/**
 * CLI runner for context engineering benchmarks.
 *
 * Usage:
 *   npx ts-node --esm src/runner.ts --benchmark needle-in-haystack --model gpt-4
 *   npx ts-node --esm src/runner.ts --all --model claude-3-5-sonnet-20241022 --output json
 */

import { Command } from "commander";
import { createLLMClient } from "./utils/llm-client.js";
import { runNeedleInHaystack } from "./benchmarks/needle-in-haystack.js";
import { runInstructionAdherence } from "./benchmarks/instruction-adherence.js";
import { runCompressionFidelity } from "./benchmarks/compression-fidelity.js";
import { runRetrievalRelevance } from "./benchmarks/retrieval-relevance.js";
import { runTokenEfficiency } from "./benchmarks/token-efficiency.js";

type OutputFormat = "table" | "json" | "csv";

interface RunConfig {
  readonly benchmarks: readonly string[];
  readonly model: string;
  readonly outputFormat: OutputFormat;
  readonly outputFile?: string;
}

const BENCHMARK_RUNNERS: Record<
  string,
  (client: ReturnType<typeof createLLMClient>) => Promise<any>
> = {
  "needle-in-haystack": runNeedleInHaystack,
  "instruction-adherence": runInstructionAdherence,
  "compression-fidelity": runCompressionFidelity,
  "retrieval-relevance": runRetrievalRelevance,
  "token-efficiency": runTokenEfficiency,
};

const ALL_BENCHMARK_NAMES = Object.keys(BENCHMARK_RUNNERS);

function formatTable(results: Record<string, any>): string {
  const rows: string[][] = [];
  const colWidths = [28, 30, 50, 20];

  const header = ["Benchmark", "Primary Metric", "Details", "Scale"];
  rows.push(header);
  rows.push(header.map((_, i) => "-".repeat(colWidths[i])));

  for (const [name, result] of Object.entries(results)) {
    if (name === "needle-in-haystack" || name === "needle_in_haystack") {
      const r = result as any;
      rows.push([
        name,
        `Recall: ${(r.overallRecall * 100).toFixed(1)}%`,
        Object.entries(r.recallByPosition ?? {})
          .map(([k, v]) => `${k}: ${((v as number) * 100).toFixed(1)}%`)
          .join(" | "),
        `${r.totalTrials} trials`,
      ]);
    } else if (name === "instruction-adherence" || name === "instruction_adherence") {
      const r = result as any;
      rows.push([
        name,
        `Compliance: ${(r.overallCompliance * 100).toFixed(1)}%`,
        `System prompt: ${r.systemPromptTokens} tokens`,
        `${r.perRule?.length ?? 0} rules tested`,
      ]);
    } else if (name === "compression-fidelity" || name === "compression_fidelity") {
      const r = result as any;
      const stratSummary = Object.entries(r.byStrategy ?? {})
        .map(
          ([k, v]: [string, any]) =>
            `${k}: fact=${((v.avgFactRetention ?? 0) * 100).toFixed(0)}%`
        )
        .join(" | ");
      rows.push([
        name,
        "See strategy breakdown",
        stratSummary.slice(0, 50),
        `${r.trials?.length ?? 0} trials`,
      ]);
    } else if (name === "retrieval-relevance" || name === "retrieval_relevance") {
      const r = result as any;
      rows.push([
        name,
        `Accuracy: ${(r.avgAnswerAccuracy * 100).toFixed(1)}%`,
        `Util: ${(r.avgUtilizationRate * 100).toFixed(1)}% | Contam: ${(r.avgContaminationRate * 100).toFixed(1)}%`,
        `${r.scenarios?.length ?? 0} scenarios`,
      ]);
    } else if (name === "token-efficiency" || name === "token_efficiency") {
      const r = result as any;
      rows.push([
        name,
        `Signal: ${(r.avgEffectiveRatio * 100).toFixed(1)}%`,
        `Full: ${(r.avgAccuracyFullContext * 100).toFixed(1)}% | Signal-only: ${(r.avgAccuracySignalOnly * 100).toFixed(1)}%`,
        `Wasted: ${r.totalWastedTokens} tok`,
      ]);
    }
  }

  return rows
    .map((row) =>
      row.map((cell, i) => cell.padEnd(colWidths[i])).join(" | ")
    )
    .join("\n");
}

function formatCsv(results: Record<string, any>): string {
  const lines = ["benchmark,metric,value"];
  for (const [name, result] of Object.entries(results)) {
    if (typeof result === "object" && result !== null) {
      for (const [key, value] of Object.entries(result)) {
        if (
          typeof value === "string" ||
          typeof value === "number" ||
          typeof value === "boolean"
        ) {
          lines.push(`${name},${key},${value}`);
        }
      }
    }
  }
  return lines.join("\n");
}

async function runBenchmarks(
  config: RunConfig
): Promise<Record<string, any>> {
  const client = createLLMClient({ model: config.model });
  const results: Record<string, any> = {};

  for (const name of config.benchmarks) {
    const runner = BENCHMARK_RUNNERS[name];
    if (!runner) {
      console.error(`Unknown benchmark: ${name}`);
      continue;
    }

    console.error(`Running ${name}...`);
    const start = performance.now();
    const result = await runner(client);
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    console.error(`  Completed in ${elapsed}s`);

    results[name] = result;
  }

  return results;
}

async function main(): Promise<void> {
  const program = new Command();

  program
    .name("context-benchmarks")
    .description("Context Engineering Benchmark Suite")
    .requiredOption("--model <model>", "Model to use (e.g., gpt-4, claude-3-5-sonnet-20241022)")
    .option(
      "--benchmark <name>",
      "Specific benchmark to run",
      undefined
    )
    .option("--all", "Run all benchmarks", false)
    .option(
      "--output <format>",
      "Output format: table, json, csv",
      "table"
    )
    .option("--output-file <path>", "Write results to file");

  program.parse();
  const opts = program.opts();

  if (!opts.benchmark && !opts.all) {
    console.error("Error: Either --benchmark <name> or --all is required");
    console.error(
      `Available benchmarks: ${ALL_BENCHMARK_NAMES.join(", ")}`
    );
    process.exit(1);
  }

  const benchmarks: readonly string[] = opts.all
    ? ALL_BENCHMARK_NAMES
    : [opts.benchmark];

  const config: RunConfig = {
    benchmarks,
    model: opts.model,
    outputFormat: opts.output as OutputFormat,
    outputFile: opts.outputFile,
  };

  console.error("Context Engineering Benchmark Suite");
  console.error(`Model: ${config.model}`);
  console.error(`Benchmarks: ${config.benchmarks.join(", ")}`);
  console.error("---");

  const results = await runBenchmarks(config);

  let formatted: string;
  switch (config.outputFormat) {
    case "json":
      formatted = JSON.stringify(results, null, 2);
      break;
    case "csv":
      formatted = formatCsv(results);
      break;
    default:
      formatted = formatTable(results);
  }

  if (config.outputFile) {
    const { writeFileSync } = await import("fs");
    writeFileSync(config.outputFile, formatted);
    console.error(`Results written to ${config.outputFile}`);
  } else {
    console.log(formatted);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
