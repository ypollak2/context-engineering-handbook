# Context Engineering Anti-Patterns

> Learn from others' mistakes. These are the most common ways context engineering goes wrong in production.

---

## Anti-Pattern: The Kitchen Sink

**Symptoms**: System prompts that are thousands of tokens long. The model ignores critical instructions, follows formatting rules inconsistently, and produces unfocused outputs. Token costs are inexplicably high.

**Root Cause**: Developers treat the system prompt as a dumping ground for every instruction, preference, example, and edge case they can think of. No prioritization or layering strategy exists.

**Impact**: Attention dilution causes the model to miss important instructions buried deep in the prompt. Cost scales linearly with prompt size on every request. Contradictory instructions create unpredictable behavior.

**Fix**: Apply [System Prompt Architecture](../patterns/construction/system-prompt-architecture.md) to separate always-on instructions from situationally relevant ones. Use [Progressive Disclosure](../patterns/construction/progressive-disclosure.md) to inject context only when the conversation reaches the relevant topic. Ruthlessly prune instructions that don't measurably improve output quality.

---

## Anti-Pattern: Context Amnesia

**Symptoms**: After a long conversation, the agent "forgets" decisions made earlier. Compacted summaries drop critical details. The user has to repeat themselves. Established constraints are violated in later turns.

**Root Cause**: Summarization and compaction treat all context as equally important. Key decisions, user preferences, and established facts are compressed away alongside genuinely low-value turns. No mechanism marks certain context as "must preserve."

**Impact**: Users lose trust when the agent contradicts earlier agreements. Subtle bugs appear when architectural decisions made 20 turns ago are forgotten during implementation. Debugging becomes impossible because the causal chain is broken.

**Fix**: Apply [Conversation Compaction](../patterns/compression/conversation-compaction.md) with explicit preservation rules to tag high-value context that must survive compaction. Maintain a structured scratchpad of decisions, constraints, and facts that persists independently of conversation history. Use [Filesystem-as-Memory](../patterns/persistence/filesystem-as-memory.md) for decisions that must never be lost.

---

## Anti-Pattern: The Echo Chamber

**Symptoms**: Agent outputs become repetitive and formulaic over long sessions. Creative tasks produce variations of the same idea. The agent converges on a single approach and cannot explore alternatives even when asked.

**Root Cause**: Accumulated context creates a strong prior that biases generation. Previous outputs in the conversation history act as few-shot examples that the model imitates. No mechanism exists to inject diversity or reset creative direction.

**Impact**: Users receive diminishing value from extended sessions. Brainstorming tasks produce an illusion of variety without substantive differences. The agent becomes unable to course-correct because every new generation reinforces the existing pattern.

**Fix**: Apply [Sub-Agent Delegation](../patterns/isolation/sub-agent-delegation.md) to run exploratory work in isolated contexts that don't inherit prior bias. Inject explicit diversity prompts ("propose an approach fundamentally different from the above"). Use [Conversation Compaction](../patterns/compression/conversation-compaction.md) to summarize conclusions rather than carrying full conversation history forward.

---

## Anti-Pattern: Stale Context Poisoning

**Symptoms**: The agent confidently states outdated information. RAG-retrieved documents reference deprecated APIs, old pricing, or superseded policies. Users follow the agent's advice and encounter errors because the information was silently out of date.

**Root Cause**: Retrieved context from vector stores, memory systems, or cached documents carries no freshness metadata. The model has no way to distinguish a document indexed yesterday from one indexed two years ago. Embedding similarity does not correlate with temporal relevance.

**Impact**: Incorrect answers delivered with high confidence. Users build on outdated assumptions. In regulated domains, stale context can cause compliance violations. Trust erosion is severe because the agent appears authoritative while being wrong.

**Fix**: Apply [RAG Context Assembly](../patterns/retrieval/rag-context-assembly.md) with freshness scoring to attach timestamps and TTLs to all stored context. Include retrieval dates in the context presented to the model with explicit instructions to flag potentially outdated information. Use [Context Rot Detection](../patterns/evaluation/context-rot-detection.md) to monitor for staleness. Periodically re-index or expire stale entries.

---

## Anti-Pattern: Tool Schema Overload

**Symptoms**: Tool selection accuracy degrades as more tools are added. The agent calls the wrong tool or invents parameters. Token usage is high even for simple queries because dozens of tool schemas are included in every request.

**Root Cause**: All available tool schemas are injected into every conversation regardless of the user's current task. A system with 50+ tools wastes thousands of tokens on schema definitions the model must parse but will never use in that turn.

**Impact**: Increased latency and cost on every request. Tool selection errors cause cascading failures in agentic workflows. The model becomes confused between tools with similar names or overlapping functionality, leading to subtle misuse.

**Fix**: Apply [Semantic Tool Selection](../patterns/retrieval/semantic-tool-selection.md) to include only the tools relevant to the current task or conversation phase. Group tools by domain and load groups on demand. Use a lightweight classifier or keyword match to determine which tool groups to activate. Start with a minimal tool set and expand only when the agent explicitly requests additional capabilities.

---

## Anti-Pattern: The Infinite Loop

**Symptoms**: An agent retries a failing operation repeatedly. Error messages accumulate in the context window. Token usage spikes with no progress. The agent eventually hits a context limit or token budget and fails catastrophically.

**Root Cause**: The retry logic feeds the error back into the context but adds no new information that would change the outcome. The agent lacks a strategy for diagnosing root causes or escalating. The same prompt plus the same error produces the same failing action.

**Impact**: Wasted compute and API costs. Context window fills with repetitive error traces, crowding out useful information. Users wait for results that will never arrive. In agentic systems, one stuck loop can block an entire pipeline.

**Fix**: Apply [Error Preservation](../patterns/optimization/error-preservation.md) with a retry budget (e.g., max 2 retries with the same approach). After the budget is exhausted, require the agent to change strategy: try an alternative tool, simplify the request, or escalate to the user. Summarize repeated errors into a single diagnostic block rather than preserving every failed attempt. Use [Observation Masking](../patterns/compression/observation-masking.md) to collapse redundant error traces.

---

## Anti-Pattern: Context Isolation Neglect

**Symptoms**: The main conversation becomes cluttered with exploratory work, dead ends, and verbose tool outputs. The agent loses track of the primary task. Context window fills up rapidly during complex multi-step operations.

**Root Cause**: All work happens in a single context window. Exploratory searches, code analysis, research tangents, and sub-task execution all compete for the same limited space. No delegation mechanism separates concerns.

**Impact**: Primary task context is diluted by secondary work. Compaction becomes aggressive and lossy because so much low-value exploratory content must be compressed. The agent's effective "working memory" for the main task shrinks as sub-tasks consume context.

**Fix**: Apply [Sub-Agent Delegation](../patterns/isolation/sub-agent-delegation.md) to run exploratory and investigative work in isolated context windows. The main agent delegates a focused question to a sub-agent, receives a concise summary of findings, and incorporates only the relevant conclusions. This keeps the main context clean and focused. Use the main thread for orchestration and decision-making; use sub-agents for research, analysis, and execution of independent sub-tasks.
