#!/usr/bin/env python3
"""CLI runner for context engineering benchmarks.

Usage:
    python runner.py --benchmark needle_in_haystack --model gpt-4
    python runner.py --all --model claude-3-5-sonnet-20241022 --output json
    python runner.py --benchmark token_efficiency --model gpt-4 --output csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

# Add the parent directory to sys.path so benchmarks package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import ALL_BENCHMARKS
from benchmarks.utils.llm_client import LLMClient


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration for a benchmark run."""

    benchmarks: tuple[str, ...]
    model: str
    output_format: str
    output_file: str | None


def parse_args() -> RunConfig:
    """Parse command-line arguments into an immutable config."""
    parser = argparse.ArgumentParser(
        description="Context Engineering Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available benchmarks:\n"
            "  needle_in_haystack     - Test fact retrieval in large contexts\n"
            "  instruction_adherence  - Test system prompt rule compliance\n"
            "  compression_fidelity   - Test info preservation after compaction\n"
            "  retrieval_relevance    - Test relevant vs irrelevant chunk usage\n"
            "  token_efficiency       - Measure signal-to-noise in context\n"
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Name of a specific benchmark to run",
        choices=list(ALL_BENCHMARKS.keys()),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g., gpt-4, claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="table",
        choices=["table", "json", "csv"],
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Write results to file (default: stdout)",
    )

    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.error("Either --benchmark <name> or --all is required")

    if args.all:
        benchmark_names = tuple(ALL_BENCHMARKS.keys())
    else:
        benchmark_names = (args.benchmark,)

    return RunConfig(
        benchmarks=benchmark_names,
        model=args.model,
        output_format=args.output,
        output_file=args.output_file,
    )


def format_table(results: dict[str, Any]) -> str:
    """Format results as a human-readable table."""
    rows: list[list[str]] = []

    for bench_name, result in results.items():
        if bench_name == "needle_in_haystack":
            rows.append([
                bench_name,
                f"Overall Recall: {result['overall_recall']:.1%}",
                " | ".join(
                    f"{k}: {v:.1%}"
                    for k, v in result.get("recall_by_position", {}).items()
                ),
                f"{result['total_trials']} trials",
            ])
        elif bench_name == "instruction_adherence":
            rows.append([
                bench_name,
                f"Overall Compliance: {result['overall_compliance']:.1%}",
                f"System prompt: {result['system_prompt_tokens']} tokens",
                f"{len(result.get('per_rule', []))} rules tested",
            ])
        elif bench_name == "compression_fidelity":
            strategy_summary = " | ".join(
                f"{k}: fact={v.get('avg_fact_retention', 0):.1%}, decision={v.get('avg_decision_retention', 0):.1%}"
                for k, v in result.get("by_strategy", {}).items()
            )
            rows.append([
                bench_name,
                "See strategy breakdown",
                strategy_summary[:80] + "..." if len(strategy_summary) > 80 else strategy_summary,
                f"{len(result.get('trials', []))} trials",
            ])
        elif bench_name == "retrieval_relevance":
            rows.append([
                bench_name,
                f"Accuracy: {result['avg_answer_accuracy']:.1%}",
                f"Utilization: {result['avg_utilization_rate']:.1%} | Contamination: {result['avg_contamination_rate']:.1%}",
                f"{len(result.get('scenarios', []))} scenarios",
            ])
        elif bench_name == "token_efficiency":
            rows.append([
                bench_name,
                f"Signal ratio: {result['avg_effective_ratio']:.1%}",
                f"Full accuracy: {result['avg_accuracy_full_context']:.1%} | Signal-only: {result['avg_accuracy_signal_only']:.1%}",
                f"Wasted: {result['total_wasted_tokens']} tokens",
            ])
        else:
            rows.append([bench_name, "Complete", "", ""])

    headers = ["Benchmark", "Primary Metric", "Details", "Scale"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def format_csv(results: dict[str, Any]) -> str:
    """Format results as CSV."""
    lines = ["benchmark,metric,value"]

    for bench_name, result in results.items():
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"{bench_name},{key},{value}")

    return "\n".join(lines)


def run_benchmarks(config: RunConfig) -> dict[str, Any]:
    """Run the specified benchmarks and return results."""
    client = LLMClient(model=config.model)
    all_results: dict[str, Any] = {}

    for bench_name in config.benchmarks:
        bench_class = ALL_BENCHMARKS[bench_name]
        benchmark = bench_class()

        print(f"Running {bench_name}...", file=sys.stderr)
        start = time.monotonic()
        result = benchmark.run(client)
        elapsed = time.monotonic() - start
        print(f"  Completed in {elapsed:.1f}s", file=sys.stderr)

        all_results[bench_name] = result.to_dict()

    return all_results


def output_results(results: dict[str, Any], config: RunConfig) -> None:
    """Format and output the results."""
    if config.output_format == "json":
        formatted = json.dumps(results, indent=2)
    elif config.output_format == "csv":
        formatted = format_csv(results)
    else:
        formatted = format_table(results)

    if config.output_file:
        Path(config.output_file).write_text(formatted)
        print(f"Results written to {config.output_file}", file=sys.stderr)
    else:
        print(formatted)


def main() -> None:
    """Entry point."""
    config = parse_args()

    print(f"Context Engineering Benchmark Suite", file=sys.stderr)
    print(f"Model: {config.model}", file=sys.stderr)
    print(f"Benchmarks: {', '.join(config.benchmarks)}", file=sys.stderr)
    print(f"---", file=sys.stderr)

    results = run_benchmarks(config)
    output_results(results, config)


if __name__ == "__main__":
    main()
