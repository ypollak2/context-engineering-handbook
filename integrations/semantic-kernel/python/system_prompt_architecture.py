"""
System Prompt Architecture -- Semantic Kernel Integration

Implements modular, composable system prompt sections using Semantic Kernel's
KernelFunction and prompt template system. Each section is independently
authored, conditionally included based on runtime context, and assembled
into a final prompt at invocation time.

Pattern: https://github.com/context-engineering-handbook/patterns/construction/system-prompt-architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Domain types (from the handbook pattern)
# ---------------------------------------------------------------------------


class SectionPriority(Enum):
    """Controls ordering in the assembled prompt. Lower value = higher priority."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True)
class PromptSection:
    """An immutable, composable section of a system prompt.

    In Semantic Kernel, each section can be thought of as a mini
    KernelFunction that renders its content with variable substitution.
    """

    name: str
    content: str
    priority: SectionPriority = SectionPriority.MEDIUM
    required: bool = False
    token_estimate: int = 0
    condition: str | None = None  # Feature flag or condition name

    def render(self, variables: dict[str, str] | None = None) -> str:
        """Render the section, substituting template variables.

        Uses Semantic Kernel's {{$variable}} syntax for compatibility.
        """
        text = self.content
        if variables:
            for key, value in variables.items():
                text = text.replace(f"{{{{${key}}}}}", value)
        return text


# ---------------------------------------------------------------------------
# SystemPromptBuilder -- SK-compatible prompt composition
# ---------------------------------------------------------------------------


class SystemPromptBuilder:
    """Composes a system prompt from modular sections.

    Designed to work with Semantic Kernel's ChatCompletionService by
    producing the system message content. Sections can be conditionally
    included based on feature flags, user tiers, or runtime state.

    Usage with Semantic Kernel:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

        kernel = Kernel()
        kernel.add_service(AzureChatCompletion(...))

        builder = SystemPromptBuilder(max_tokens=8000)
        builder = builder.add_section(PromptSection(
            name="role",
            content="You are a senior code reviewer.",
            priority=SectionPriority.CRITICAL,
            required=True,
        ))

        # Build and use as system message
        system_prompt = builder.build(
            variables={"user_name": "Alice"},
            active_conditions={"premium_tier", "code_review_mode"},
        )

        chat = kernel.get_service(type=AzureChatCompletion)
        response = await chat.get_chat_message_content(
            ChatHistory(system_message=system_prompt),
            settings,
        )
    """

    def __init__(
        self,
        sections: list[PromptSection] | None = None,
        section_separator: str = "\n\n---\n\n",
        max_tokens: int = 8000,
    ) -> None:
        self._sections: list[PromptSection] = list(sections or [])
        self._section_separator = section_separator
        self._max_tokens = max_tokens

    def add_section(self, section: PromptSection) -> SystemPromptBuilder:
        """Add a section. Returns a new builder (immutable pattern)."""
        return SystemPromptBuilder(
            sections=[*self._sections, section],
            section_separator=self._section_separator,
            max_tokens=self._max_tokens,
        )

    def build(
        self,
        variables: dict[str, str] | None = None,
        active_conditions: set[str] | None = None,
    ) -> str:
        """Assemble the final prompt from eligible sections.

        Args:
            variables: Template variable substitutions.
            active_conditions: Set of active feature flags or conditions.
                Sections with a `condition` are only included if their
                condition is in this set.
        """
        conditions = active_conditions or set()

        # Filter sections by conditions
        eligible: list[PromptSection] = []
        for section in self._sections:
            if section.condition is not None and section.condition not in conditions:
                continue
            eligible.append(section)

        # Sort by priority (CRITICAL first)
        eligible.sort(key=lambda s: s.priority.value)

        # Render within token budget
        parts: list[str] = []
        total_tokens = 0

        for section in eligible:
            rendered = section.render(variables)
            if total_tokens + section.token_estimate > self._max_tokens:
                if section.required:
                    # Required sections are always included
                    parts.append(rendered)
                    total_tokens += section.token_estimate
                continue
            parts.append(rendered)
            total_tokens += section.token_estimate

        return self._section_separator.join(parts)

    @property
    def section_count(self) -> int:
        return len(self._sections)

    @property
    def total_token_estimate(self) -> int:
        return sum(s.token_estimate for s in self._sections)


# ---------------------------------------------------------------------------
# Semantic Kernel prompt function builder
# ---------------------------------------------------------------------------


def build_sk_prompt_template(builder: SystemPromptBuilder) -> str:
    """Convert a SystemPromptBuilder into a Semantic Kernel prompt template.

    This generates a template string that uses SK's {{$variable}} syntax
    and can be registered as a KernelFunction.

    Usage:
        from semantic_kernel import Kernel
        from semantic_kernel.functions import KernelFunctionFromPrompt

        template = build_sk_prompt_template(builder)
        function = KernelFunctionFromPrompt(
            function_name="system_prompt",
            prompt=template,
        )
        kernel.add_function("prompts", function)
    """
    sections = [
        (
            f"<!-- Section: {s.name} (priority: {s.priority.name}) -->\n"
            f"{s.content}"
        )
        for s in sorted(builder._sections, key=lambda s: s.priority.value)
    ]
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate System Prompt Architecture with Semantic Kernel patterns."""

    # Build a modular system prompt
    builder = SystemPromptBuilder(max_tokens=8000)

    # Critical: Always included
    builder = builder.add_section(
        PromptSection(
            name="role",
            content=(
                "You are a senior software engineer at {{$company_name}}. "
                "You help developers write, review, and debug code."
            ),
            priority=SectionPriority.CRITICAL,
            required=True,
            token_estimate=30,
        )
    )

    # Critical: Behavioral constraints
    builder = builder.add_section(
        PromptSection(
            name="constraints",
            content=(
                "## Constraints\n"
                "- Never execute code directly\n"
                "- Always explain your reasoning\n"
                "- Flag security issues immediately\n"
                "- Prefer immutable data patterns"
            ),
            priority=SectionPriority.CRITICAL,
            required=True,
            token_estimate=40,
        )
    )

    # High: Output format
    builder = builder.add_section(
        PromptSection(
            name="output_format",
            content=(
                "## Output Format\n"
                "Respond in markdown. Use code fences with language tags. "
                "Structure reviews with: Summary, Issues, Suggestions."
            ),
            priority=SectionPriority.HIGH,
            token_estimate=30,
        )
    )

    # Conditional: Only for code review mode
    builder = builder.add_section(
        PromptSection(
            name="review_guidelines",
            content=(
                "## Code Review Guidelines\n"
                "1. Check for correctness first\n"
                "2. Then readability and naming\n"
                "3. Then performance implications\n"
                "4. Check for test coverage\n"
                "5. Verify error handling completeness"
            ),
            priority=SectionPriority.HIGH,
            condition="code_review_mode",
            token_estimate=50,
        )
    )

    # Conditional: Only for premium users
    builder = builder.add_section(
        PromptSection(
            name="premium_features",
            content=(
                "## Premium Features\n"
                "You have access to advanced analysis tools:\n"
                "- Architecture diagram generation\n"
                "- Performance profiling suggestions\n"
                "- Security vulnerability scanning"
            ),
            priority=SectionPriority.MEDIUM,
            condition="premium_tier",
            token_estimate=40,
        )
    )

    # Low priority: Domain knowledge
    builder = builder.add_section(
        PromptSection(
            name="tech_stack",
            content=(
                "## Tech Stack\n"
                "The codebase uses: Python 3.12, FastAPI, PostgreSQL, Redis, "
                "Docker, Kubernetes. Testing: pytest with 80% coverage target."
            ),
            priority=SectionPriority.LOW,
            token_estimate=30,
        )
    )

    print(f"Registered {builder.section_count} sections")
    print(f"Total token estimate: {builder.total_token_estimate}")
    print()

    # Scenario 1: Free tier, general assistance
    print("=== Scenario 1: Free tier, general assistance ===")
    prompt1 = builder.build(
        variables={"company_name": "Acme Corp"},
        active_conditions=set(),  # No special conditions
    )
    print(prompt1)
    print()

    # Scenario 2: Premium tier, code review mode
    print("=== Scenario 2: Premium tier, code review ===")
    prompt2 = builder.build(
        variables={"company_name": "Acme Corp"},
        active_conditions={"code_review_mode", "premium_tier"},
    )
    print(prompt2)
    print()

    # Show the SK-compatible template
    print("=== SK Prompt Template ===")
    template = build_sk_prompt_template(builder)
    print(template[:500] + "...")


if __name__ == "__main__":
    main()
