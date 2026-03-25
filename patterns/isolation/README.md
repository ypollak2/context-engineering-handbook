# Context Isolation Patterns

Context isolation is the practice of **preventing context pollution between tasks** by giving each unit of work its own bounded context window. Instead of cramming everything into a single growing context, isolation patterns partition work across separate agent instances, each with only the information it needs.

These patterns answer the question: *How do I stop one task's context from degrading another task's performance?*

## Decision Tree

```
Start here: What is your multi-task challenge?
|
|-- "My agent's context gets polluted by exploratory sub-tasks"
|     --> Sub-Agent Delegation
|
|-- "I have multiple specialized agents that need to collaborate"
|     --> Multi-Agent Context Orchestration
|
|-- "I need parallel execution of independent research tasks"
|     --> Sub-Agent Delegation (supports concurrent child agents)
|
|-- "I need to define strict context contracts between team members"
|     --> Multi-Agent Context Orchestration
|
|-- "I have a single complex task that involves both research and action"
|     --> Sub-Agent Delegation for research, parent agent for action
|
|-- "I have a pipeline where Agent A's output feeds Agent B's input"
|     --> Multi-Agent Context Orchestration with sequential routing
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Sub-Agent Delegation](sub-agent-delegation.md) | Spawn child agents with isolated contexts for sub-tasks | Context pollution prevention and parallel execution |
| [Multi-Agent Context Orchestration](multi-agent-context-orchestration.md) | Manage context flow between collaborating agents via contracts | Scalable multi-agent collaboration without context explosion |

## How They Compose

These two patterns address isolation at different scales:

- **Sub-Agent Delegation** is a parent-child relationship. A single orchestrator spawns focused workers, gives them minimal context, and collects their results. The parent's context stays clean because exploratory work happens in disposable child contexts.
- **Multi-Agent Context Orchestration** is a peer-to-peer (or pipeline) relationship. Multiple specialized agents collaborate, and an orchestration layer controls what context flows between them, preventing any single agent from accumulating the full context of all agents.

In practice, they nest naturally: a multi-agent orchestrator coordinates specialized agents, and each of those agents may internally use sub-agent delegation for its own sub-tasks. The orchestration layer defines the macro context flow; sub-agent delegation handles the micro context isolation.
