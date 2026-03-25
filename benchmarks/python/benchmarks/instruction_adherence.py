"""Instruction Adherence Benchmark.

Tests whether the LLM follows specific behavioral rules defined in the system
prompt. Measures compliance rate across diverse queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from benchmarks.utils.llm_client import LLMClient, Message
from benchmarks.utils.metrics import count_tokens


@dataclass(frozen=True)
class Rule:
    """A behavioral rule to test adherence against."""

    id: str
    description: str
    instruction: str
    checker: str  # Name of the checker function to use


# 10 specific behavioral rules
RULES: tuple[Rule, ...] = (
    Rule(
        id="json_format",
        description="Always respond in valid JSON",
        instruction="Always format your entire response as valid JSON with a 'response' key.",
        checker="check_json_format",
    ),
    Rule(
        id="no_simply",
        description="Never use the word 'simply'",
        instruction="Never use the word 'simply' in your response.",
        checker="check_no_simply",
    ),
    Rule(
        id="confidence_score",
        description="Include confidence scores",
        instruction="Include a 'confidence' field (0.0-1.0) in your JSON response.",
        checker="check_confidence_score",
    ),
    Rule(
        id="max_sentences",
        description="Maximum 3 sentences in the response value",
        instruction="Keep the 'response' value to 3 sentences or fewer.",
        checker="check_max_sentences",
    ),
    Rule(
        id="no_apology",
        description="Never apologize",
        instruction="Never begin with or include phrases like 'I apologize', 'Sorry', or 'I'm sorry'.",
        checker="check_no_apology",
    ),
    Rule(
        id="uppercase_key_terms",
        description="Key technical terms in uppercase",
        instruction="Write key technical terms (programming languages, frameworks, protocols) in ALL CAPS.",
        checker="check_uppercase_terms",
    ),
    Rule(
        id="numbered_steps",
        description="Use numbered steps for procedures",
        instruction="When describing a process or procedure, use numbered steps in a 'steps' array.",
        checker="check_numbered_steps",
    ),
    Rule(
        id="source_citation",
        description="Cite knowledge source",
        instruction="Include a 'source' field indicating whether the answer is from 'training_data', 'reasoning', or 'uncertain'.",
        checker="check_source_citation",
    ),
    Rule(
        id="no_markdown",
        description="No markdown formatting",
        instruction="Do not use any markdown formatting (no **, no ##, no `, no -).",
        checker="check_no_markdown",
    ),
    Rule(
        id="word_count",
        description="Include word count",
        instruction="Include a 'word_count' field with the number of words in the 'response' value.",
        checker="check_word_count",
    ),
)

# 20 diverse test queries
QUERIES: tuple[str, ...] = (
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
)


def _check_json_format(response: str) -> bool:
    """Check if the response is valid JSON with a 'response' key."""
    try:
        parsed = json.loads(response.strip())
        return isinstance(parsed, dict) and "response" in parsed
    except (json.JSONDecodeError, ValueError):
        return False


def _check_no_simply(response: str) -> bool:
    """Check that the word 'simply' does not appear."""
    return "simply" not in response.lower()


def _check_confidence_score(response: str) -> bool:
    """Check for a confidence field with a float between 0 and 1."""
    try:
        parsed = json.loads(response.strip())
        conf = parsed.get("confidence")
        return isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _check_max_sentences(response: str) -> bool:
    """Check that the response value has 3 or fewer sentences."""
    try:
        parsed = json.loads(response.strip())
        text = parsed.get("response", "")
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return len(sentences) <= 3
    except (json.JSONDecodeError, ValueError):
        return False


def _check_no_apology(response: str) -> bool:
    """Check that no apology phrases appear."""
    lower = response.lower()
    apology_phrases = ("i apologize", "sorry", "i'm sorry", "apologies")
    return not any(phrase in lower for phrase in apology_phrases)


def _check_uppercase_terms(response: str) -> bool:
    """Check that at least some technical terms are uppercased.

    This is a soft check - we look for common tech terms and verify at least
    one appears in uppercase if any tech terms are present.
    """
    tech_terms = [
        "python", "javascript", "typescript", "java", "http", "https",
        "tcp", "udp", "dns", "sql", "rest", "api", "json", "html", "css",
    ]
    lower = response.lower()
    found_terms = [t for t in tech_terms if t in lower]
    if not found_terms:
        return True  # No tech terms to check
    return any(t.upper() in response for t in found_terms)


def _check_numbered_steps(response: str) -> bool:
    """Check for numbered steps when a process is described.

    If the response contains process-like language, verify it has a 'steps' array.
    """
    try:
        parsed = json.loads(response.strip())
        resp_text = str(parsed.get("response", "")).lower()
        process_indicators = ("first", "then", "next", "step", "process", "procedure")
        has_process_language = any(ind in resp_text for ind in process_indicators)
        if not has_process_language:
            return True  # No process to describe
        return "steps" in parsed and isinstance(parsed["steps"], list)
    except (json.JSONDecodeError, ValueError):
        return False


def _check_source_citation(response: str) -> bool:
    """Check for a source field with a valid value."""
    try:
        parsed = json.loads(response.strip())
        source = parsed.get("source", "")
        return source in ("training_data", "reasoning", "uncertain")
    except (json.JSONDecodeError, ValueError):
        return False


def _check_no_markdown(response: str) -> bool:
    """Check that no markdown formatting is used."""
    markdown_patterns = ["**", "##", "```", "- ", "* "]
    return not any(pattern in response for pattern in markdown_patterns)


def _check_word_count(response: str) -> bool:
    """Check that a word_count field exists and is approximately correct."""
    try:
        parsed = json.loads(response.strip())
        claimed_count = parsed.get("word_count")
        if not isinstance(claimed_count, int):
            return False
        actual_text = parsed.get("response", "")
        actual_count = len(actual_text.split())
        # Allow 20% tolerance
        return abs(claimed_count - actual_count) <= max(2, actual_count * 0.2)
    except (json.JSONDecodeError, ValueError):
        return False


_CHECKERS: dict[str, callable] = {
    "check_json_format": _check_json_format,
    "check_no_simply": _check_no_simply,
    "check_confidence_score": _check_confidence_score,
    "check_max_sentences": _check_max_sentences,
    "check_no_apology": _check_no_apology,
    "check_uppercase_terms": _check_uppercase_terms,
    "check_numbered_steps": _check_numbered_steps,
    "check_source_citation": _check_source_citation,
    "check_no_markdown": _check_no_markdown,
    "check_word_count": _check_word_count,
}


@dataclass(frozen=True)
class RuleResult:
    """Adherence result for a single rule across all queries."""

    rule_id: str
    rule_description: str
    passed: int
    total: int
    adherence_rate: float


@dataclass(frozen=True)
class QueryResult:
    """Result for a single query across all rules."""

    query: str
    rules_passed: int
    total_rules: int
    compliance_rate: float
    response: str


@dataclass(frozen=True)
class InstructionAdherenceResult:
    """Aggregate results for the instruction adherence benchmark."""

    rule_results: tuple[RuleResult, ...]
    query_results: tuple[QueryResult, ...]
    overall_compliance: float
    system_prompt_tokens: int
    model: str

    def to_dict(self) -> dict:
        return {
            "benchmark": "instruction_adherence",
            "model": self.model,
            "overall_compliance": round(self.overall_compliance, 4),
            "system_prompt_tokens": self.system_prompt_tokens,
            "per_rule": [
                {
                    "rule_id": r.rule_id,
                    "description": r.rule_description,
                    "adherence_rate": round(r.adherence_rate, 4),
                    "passed": r.passed,
                    "total": r.total,
                }
                for r in self.rule_results
            ],
            "per_query_summary": {
                "total_queries": len(self.query_results),
                "avg_compliance": round(
                    sum(q.compliance_rate for q in self.query_results) / len(self.query_results),
                    4,
                ) if self.query_results else 0.0,
            },
        }


@dataclass(frozen=True)
class InstructionAdherenceBenchmark:
    """Benchmark: Does the LLM follow specific behavioral rules?"""

    rules: tuple[Rule, ...] = RULES
    queries: tuple[str, ...] = QUERIES

    def run(self, client: LLMClient) -> InstructionAdherenceResult:
        """Run the instruction adherence benchmark."""
        system_prompt = self._build_system_prompt()
        system_tokens = count_tokens(system_prompt, client.model)

        # Track results: rule_id -> list of bools
        rule_passes: dict[str, list[bool]] = {r.id: [] for r in self.rules}
        query_results: list[QueryResult] = []

        for query in self.queries:
            response = client.complete(
                messages=[Message(role="user", content=query)],
                system=system_prompt,
            )

            rules_passed = 0
            for rule in self.rules:
                checker = _CHECKERS[rule.checker]
                passed = checker(response.content)
                rule_passes[rule.id].append(passed)
                if passed:
                    rules_passed += 1

            query_results.append(
                QueryResult(
                    query=query,
                    rules_passed=rules_passed,
                    total_rules=len(self.rules),
                    compliance_rate=rules_passed / len(self.rules),
                    response=response.content,
                )
            )

        rule_results = tuple(
            RuleResult(
                rule_id=rule.id,
                rule_description=rule.description,
                passed=sum(1 for p in rule_passes[rule.id] if p),
                total=len(rule_passes[rule.id]),
                adherence_rate=sum(1 for p in rule_passes[rule.id] if p) / len(rule_passes[rule.id])
                if rule_passes[rule.id] else 0.0,
            )
            for rule in self.rules
        )

        total_checks = len(self.rules) * len(self.queries)
        total_passed = sum(r.passed for r in rule_results)
        overall = total_passed / total_checks if total_checks > 0 else 0.0

        return InstructionAdherenceResult(
            rule_results=rule_results,
            query_results=tuple(query_results),
            overall_compliance=overall,
            system_prompt_tokens=system_tokens,
            model=client.model,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt with all rules."""
        instructions = "\n".join(
            f"{i + 1}. {rule.instruction}" for i, rule in enumerate(self.rules)
        )
        return (
            "You are a helpful assistant. You MUST follow ALL of these rules "
            "in every response:\n\n"
            f"{instructions}\n\n"
            "Failure to follow any rule is considered a violation. "
            "Every response must comply with ALL rules simultaneously."
        )
