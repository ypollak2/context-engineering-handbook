# System Prompt Architecture

> Structure system prompts into modular, composable sections instead of monolithic text blocks.

## Problem

System prompts in production applications grow organically into thousands of lines of unstructured text. Different teams add instructions, examples, and constraints with no clear organization. The result is a prompt that is hard to debug, impossible to A/B test at the section level, and brittle to modify -- changing one paragraph introduces regressions elsewhere. Without structure, you also cannot conditionally include sections based on user tier, feature flags, or task type.

## Solution

Treat the system prompt like source code: decompose it into discrete, named sections with clear responsibilities. Each section -- role definition, behavioral constraints, output format, domain knowledge, tool instructions -- is an independent unit that can be authored, tested, and versioned separately. A builder or template engine composes the final prompt at runtime by selecting and ordering the relevant sections.

This is the same principle behind component-based UI or microservice architecture, applied to prompt engineering. Sections declare their own dependencies (e.g., "tool instructions" requires "output format" to define how tool results are reported), and the builder resolves the final assembly.

## How It Works

```
+--------------------------------------------------+
|              System Prompt (assembled)            |
|                                                   |
|  +--------------------------------------------+  |
|  | 1. Role Definition                         |  |
|  |    "You are a senior code reviewer..."     |  |
|  +--------------------------------------------+  |
|  | 2. Behavioral Constraints                  |  |
|  |    "Never execute code. Always explain..." |  |
|  +--------------------------------------------+  |
|  | 3. Domain Knowledge          [conditional] |  |
|  |    "The codebase uses React 19 with..."    |  |
|  +--------------------------------------------+  |
|  | 4. Output Format                           |  |
|  |    "Respond in markdown with sections..."  |  |
|  +--------------------------------------------+  |
|  | 5. Tool Instructions         [conditional] |  |
|  |    "You have access to: search, edit..."   |  |
|  +--------------------------------------------+  |
|  | 6. Examples                   [conditional] |  |
|  |    "Here is an example review..."          |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+

Sections are:
- Independently versioned and tested
- Conditionally included based on context
- Ordered by priority (most important first)
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum


class SectionPriority(Enum):
    """Controls ordering in the assembled prompt. Lower value = higher priority."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True)
class PromptSection:
    """An immutable, composable section of a system prompt."""
    name: str
    content: str
    priority: SectionPriority = SectionPriority.MEDIUM
    required: bool = False
    token_estimate: int = 0

    def render(self, variables: dict[str, str] | None = None) -> str:
        """Render the section, substituting any template variables."""
        text = self.content
        if variables:
            for key, value in variables.items():
                text = text.replace(f"{{{{{key}}}}}", value)
        return text


@dataclass
class SystemPromptBuilder:
    """Composes a system prompt from modular sections.

    Sections are added independently and assembled at build time.
    Conditional sections are included only when their predicates pass.
    """
    _sections: list[PromptSection] = field(default_factory=list)
    _section_separator: str = "\n\n---\n\n"
    _max_tokens: int = 8000

    def add_section(self, section: PromptSection) -> "SystemPromptBuilder":
        """Add a section to the prompt. Returns a new builder (immutable pattern)."""
        new_builder = SystemPromptBuilder(
            _sections=[*self._sections, section],
            _section_separator=self._section_separator,
            _max_tokens=self._max_tokens,
        )
        return new_builder

    def add_section_if(
        self, section: PromptSection, condition: bool
    ) -> "SystemPromptBuilder":
        """Conditionally add a section."""
        if condition:
            return self.add_section(section)
        return self

    def build(self, variables: dict[str, str] | None = None) -> str:
        """Assemble the final system prompt.

        Sections are sorted by priority, then joined with separators.
        Raises if token budget is exceeded.
        """
        sorted_sections = sorted(self._sections, key=lambda s: s.priority.value)

        total_tokens = sum(s.token_estimate for s in sorted_sections)
        if total_tokens > self._max_tokens:
            # Drop lowest-priority non-required sections until we fit
            sorted_sections = self._trim_to_budget(sorted_sections)

        rendered = [section.render(variables) for section in sorted_sections]
        return self._section_separator.join(rendered)

    def _trim_to_budget(
        self, sections: list[PromptSection]
    ) -> list[PromptSection]:
        """Remove lowest-priority optional sections until within token budget."""
        # Work from the end (lowest priority) backward
        kept: list[PromptSection] = list(sections)
        total = sum(s.token_estimate for s in kept)

        for section in reversed(sections):
            if total <= self._max_tokens:
                break
            if not section.required:
                kept.remove(section)
                total -= section.token_estimate

        return kept


# --- Usage ---

role = PromptSection(
    name="role",
    content=(
        "You are a senior code reviewer specializing in {{language}}. "
        "You provide thorough, constructive feedback focused on correctness, "
        "readability, and maintainability."
    ),
    priority=SectionPriority.CRITICAL,
    required=True,
    token_estimate=40,
)

constraints = PromptSection(
    name="constraints",
    content=(
        "Rules:\n"
        "- Never execute or run code. Only analyze it.\n"
        "- Always explain the reasoning behind each suggestion.\n"
        "- Flag security issues with [SECURITY] prefix.\n"
        "- Limit response to the 5 most important findings."
    ),
    priority=SectionPriority.CRITICAL,
    required=True,
    token_estimate=60,
)

output_format = PromptSection(
    name="output_format",
    content=(
        "Respond using this structure:\n"
        "## Summary\nOne paragraph overview.\n"
        "## Issues\nNumbered list, severity in brackets: [HIGH], [MEDIUM], [LOW].\n"
        "## Suggestions\nActionable improvements with code snippets."
    ),
    priority=SectionPriority.HIGH,
    required=True,
    token_estimate=50,
)

react_knowledge = PromptSection(
    name="react_domain",
    content=(
        "Domain context: This codebase uses React 19 with Server Components. "
        "Prefer 'use client' only when necessary. Check for proper Suspense "
        "boundaries and error boundaries around async components."
    ),
    priority=SectionPriority.MEDIUM,
    token_estimate=45,
)

tool_instructions = PromptSection(
    name="tools",
    content=(
        "You have access to the following tools:\n"
        "- `search(query)`: Search the codebase for symbols or patterns.\n"
        "- `read_file(path)`: Read a file's contents.\n"
        "Use tools to gather context before making review comments."
    ),
    priority=SectionPriority.HIGH,
    token_estimate=55,
)


# Build with conditional sections
has_tools = True
is_react_project = True

prompt = (
    SystemPromptBuilder()
    .add_section(role)
    .add_section(constraints)
    .add_section(output_format)
    .add_section_if(react_knowledge, condition=is_react_project)
    .add_section_if(tool_instructions, condition=has_tools)
    .build(variables={"language": "TypeScript"})
)

print(prompt)
```

### TypeScript

```typescript
enum SectionPriority {
  CRITICAL = 0,
  HIGH = 1,
  MEDIUM = 2,
  LOW = 3,
}

interface PromptSection {
  readonly name: string;
  readonly content: string;
  readonly priority: SectionPriority;
  readonly required: boolean;
  readonly tokenEstimate: number;
}

interface BuilderOptions {
  readonly sectionSeparator: string;
  readonly maxTokens: number;
}

const DEFAULT_OPTIONS: BuilderOptions = {
  sectionSeparator: "\n\n---\n\n",
  maxTokens: 8000,
};

function createSection(
  params: Partial<PromptSection> & Pick<PromptSection, "name" | "content">
): PromptSection {
  return {
    priority: SectionPriority.MEDIUM,
    required: false,
    tokenEstimate: 0,
    ...params,
  };
}

function renderSection(
  section: PromptSection,
  variables?: Record<string, string>
): string {
  let text = section.content;
  if (variables) {
    for (const [key, value] of Object.entries(variables)) {
      text = text.replaceAll(`{{${key}}}`, value);
    }
  }
  return text;
}

class SystemPromptBuilder {
  private readonly sections: readonly PromptSection[];
  private readonly options: BuilderOptions;

  constructor(
    sections: readonly PromptSection[] = [],
    options: Partial<BuilderOptions> = {}
  ) {
    this.sections = sections;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /** Add a section. Returns a new builder instance (immutable). */
  addSection(section: PromptSection): SystemPromptBuilder {
    return new SystemPromptBuilder(
      [...this.sections, section],
      this.options
    );
  }

  /** Conditionally add a section. */
  addSectionIf(
    section: PromptSection,
    condition: boolean
  ): SystemPromptBuilder {
    return condition ? this.addSection(section) : this;
  }

  /** Assemble the final system prompt string. */
  build(variables?: Record<string, string>): string {
    const sorted = [...this.sections].sort(
      (a, b) => a.priority - b.priority
    );

    const trimmed = this.trimToBudget(sorted);

    return trimmed
      .map((section) => renderSection(section, variables))
      .join(this.options.sectionSeparator);
  }

  private trimToBudget(sections: PromptSection[]): PromptSection[] {
    let total = sections.reduce((sum, s) => sum + s.tokenEstimate, 0);
    if (total <= this.options.maxTokens) return sections;

    const kept = [...sections];
    for (let i = kept.length - 1; i >= 0; i--) {
      if (total <= this.options.maxTokens) break;
      if (!kept[i].required) {
        total -= kept[i].tokenEstimate;
        kept.splice(i, 1);
      }
    }
    return kept;
  }
}

// --- Usage ---

const role = createSection({
  name: "role",
  content:
    "You are a senior code reviewer specializing in {{language}}. " +
    "You provide thorough, constructive feedback focused on correctness, " +
    "readability, and maintainability.",
  priority: SectionPriority.CRITICAL,
  required: true,
  tokenEstimate: 40,
});

const constraints = createSection({
  name: "constraints",
  content: [
    "Rules:",
    "- Never execute or run code. Only analyze it.",
    "- Always explain the reasoning behind each suggestion.",
    "- Flag security issues with [SECURITY] prefix.",
    "- Limit response to the 5 most important findings.",
  ].join("\n"),
  priority: SectionPriority.CRITICAL,
  required: true,
  tokenEstimate: 60,
});

const outputFormat = createSection({
  name: "output_format",
  content: [
    "Respond using this structure:",
    "## Summary",
    "One paragraph overview.",
    "## Issues",
    "Numbered list, severity in brackets: [HIGH], [MEDIUM], [LOW].",
    "## Suggestions",
    "Actionable improvements with code snippets.",
  ].join("\n"),
  priority: SectionPriority.HIGH,
  required: true,
  tokenEstimate: 50,
});

const reactKnowledge = createSection({
  name: "react_domain",
  content:
    "Domain context: This codebase uses React 19 with Server Components. " +
    "Prefer 'use client' only when necessary. Check for proper Suspense " +
    "boundaries and error boundaries around async components.",
  priority: SectionPriority.MEDIUM,
  tokenEstimate: 45,
});

const toolInstructions = createSection({
  name: "tools",
  content: [
    "You have access to the following tools:",
    "- `search(query)`: Search the codebase for symbols or patterns.",
    "- `read_file(path)`: Read a file's contents.",
    "Use tools to gather context before making review comments.",
  ].join("\n"),
  priority: SectionPriority.HIGH,
  tokenEstimate: 55,
});

const hasTools = true;
const isReactProject = true;

const prompt = new SystemPromptBuilder()
  .addSection(role)
  .addSection(constraints)
  .addSection(outputFormat)
  .addSectionIf(reactKnowledge, isReactProject)
  .addSectionIf(toolInstructions, hasTools)
  .build({ language: "TypeScript" });

console.log(prompt);
```

## Trade-offs

| Pros | Cons |
|------|------|
| Sections can be tested independently | Adds abstraction over plain strings |
| Conditional inclusion enables feature flags and A/B tests | Section interactions can be subtle (ordering, references between sections) |
| Token budget management is explicit | Requires upfront design effort for section boundaries |
| Multiple teams can own different sections | Template variable syntax needs escaping discipline |
| Version control diffs are meaningful per-section | Over-decomposition can fragment coherent instructions |

## When to Use

- Your system prompt exceeds ~500 tokens and is maintained by more than one person
- You need conditional prompt behavior based on user tier, feature flags, or task type
- You want to A/B test specific prompt sections independently
- You manage multiple AI-powered features that share common instructions (e.g., tone, safety rules)
- Your prompt needs to respect a strict token budget and gracefully degrade

## When NOT to Use

- Simple, single-purpose prompts under 200 tokens -- the overhead is not justified
- Rapid prototyping where the prompt is still changing every hour
- One-off scripts or notebooks where prompt maintainability is irrelevant
- When all prompt content is always included unconditionally and authored by one person

## Related Patterns

- [Progressive Disclosure](progressive-disclosure.md) -- decides *when* each section is included over a conversation
- [Few-Shot Curation](few-shot-curation.md) -- dynamically populates the examples section of your prompt architecture

## Real-World Examples

- **Anthropic's Claude system prompts** use layered sections: identity, capabilities, behavioral guidelines, tool definitions, and formatting rules are distinct blocks assembled per deployment context.
- **OpenAI's GPT Builder** stores system prompts as structured JSON with separate fields for instructions, conversation starters, and knowledge references -- effectively a UI over this pattern.
- **Production template engines**: Teams at Stripe, Notion, and similar companies use Jinja2 (Python) or Handlebars (JS) to compose system prompts from template partials, with conditionals for feature flags and user segments.
- **LangChain's ChatPromptTemplate** implements a version of this pattern where `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`, and others are composed into a pipeline with variable substitution.
