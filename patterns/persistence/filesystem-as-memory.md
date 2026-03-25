# Filesystem-as-Memory

> Use the filesystem as a persistent, structured memory store -- write markdown files, JSON, or structured notes that agents can read back in future sessions.

## Problem

Most approaches to agent memory involve vector databases, embedding pipelines, and specialized infrastructure. But for many use cases -- especially coding agents and developer tools -- this is over-engineered. You need memory that is:

- **Human-readable**: Developers should be able to open, read, and edit memory files directly
- **Versionable**: Memory should participate in git workflows (diff, blame, review)
- **Tooling-compatible**: Existing tools (grep, editors, CI) should work with memory files
- **Simple to implement**: No additional infrastructure beyond the filesystem

Without a filesystem-based approach, you end up with opaque vector stores that developers cannot inspect, debug, or override.

## Solution

Store agent memory as structured files on disk. Use directories for organization (by topic, by date, by project), frontmatter for metadata, and markdown or JSON for content. The filesystem provides natural affordances: directories are categories, filenames are identifiers, file modification times are timestamps, and git provides versioning for free.

The key insight: **the most successful coding agents (Claude Code, Cursor, Devin) all use this pattern**. It is not a compromise -- it is the optimal design for tools that operate within a developer workflow.

## How It Works

```
Project Root
|
+-- .agent-memory/
|   |
|   +-- decisions/
|   |   +-- 2024-01-15-auth-architecture.md
|   |   +-- 2024-01-20-database-choice.md
|   |
|   +-- learnings/
|   |   +-- testing-patterns.md
|   |   +-- deployment-gotchas.md
|   |
|   +-- preferences/
|   |   +-- code-style.md
|   |   +-- review-checklist.md
|   |
|   +-- index.json        <-- Optional: structured index for fast lookup
|
+-- CLAUDE.md              <-- Top-level memory file (Claude Code pattern)
+-- .cursorrules           <-- Top-level memory file (Cursor pattern)
```

Memory lifecycle:
```
Agent learns something
        |
        v
+-------------------+
| Classify & Route  |  <-- What kind of memory? Decision? Learning? Preference?
+-------------------+
        |
        v
+-------------------+
| Write to File     |  <-- Structured markdown with frontmatter
+-------------------+
        |
        v
+-------------------+
| Git Tracks It     |  <-- Versioned, diffable, reviewable
+-------------------+

        ...later...

New session starts
        |
        v
+-------------------+
| Read Key Files    |  <-- Always load: top-level file, recent decisions
+-------------------+
        |
        v
+-------------------+
| Search on Demand  |  <-- Grep/glob for specific topics as needed
+-------------------+
        |
        v
Agent proceeds with persistent context
```

## Implementation

### Python

```python
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory file with metadata and content."""
    category: str
    title: str
    content: str
    tags: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""

    def to_frontmatter(self) -> str:
        """Serialize metadata as YAML frontmatter."""
        lines = [
            "---",
            f"title: {self.title}",
            f"category: {self.category}",
            f"tags: [{', '.join(self.tags)}]",
            f"created_at: {self.created_at}",
            f"updated_at: {self.updated_at}",
            "---",
        ]
        return "\n".join(lines)

    def to_file_content(self) -> str:
        """Full file content with frontmatter and body."""
        return f"{self.to_frontmatter()}\n\n{self.content}\n"


class FilesystemMemory:
    """
    A persistent memory store backed by structured files on disk.

    Organizes memory into categories (directories), with each memory
    stored as a markdown file with YAML frontmatter metadata.
    """

    def __init__(self, base_path: str | Path):
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_path / "index.json"

    def store(
        self,
        category: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> Path:
        """Write a memory entry to the filesystem."""
        now = datetime.now(timezone.utc).isoformat()
        slug = self._slugify(title)
        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        category_dir = self._base_path / category
        category_dir.mkdir(parents=True, exist_ok=True)

        file_path = category_dir / f"{date_prefix}-{slug}.md"

        # If file exists, preserve created_at
        created_at = now
        if file_path.exists():
            existing = self._read_frontmatter(file_path)
            created_at = existing.get("created_at", now)

        entry = MemoryEntry(
            category=category,
            title=title,
            content=content,
            tags=tuple(tags or []),
            created_at=created_at,
            updated_at=now,
        )

        file_path.write_text(entry.to_file_content(), encoding="utf-8")
        self._update_index(category, title, str(file_path), tags or [])
        return file_path

    def recall(self, category: str | None = None) -> list[MemoryEntry]:
        """Read all memory entries, optionally filtered by category."""
        entries = []
        search_path = (
            self._base_path / category if category else self._base_path
        )

        for md_file in sorted(search_path.rglob("*.md")):
            entry = self._read_entry(md_file)
            if entry is not None:
                entries.append(entry)

        return entries

    def search(self, query: str) -> list[MemoryEntry]:
        """Simple text search across all memory files."""
        query_lower = query.lower()
        results = []

        for md_file in self._base_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if query_lower in content.lower():
                entry = self._read_entry(md_file)
                if entry is not None:
                    results.append(entry)

        return results

    def build_context_block(
        self,
        categories: list[str] | None = None,
        max_entries: int = 10,
    ) -> str:
        """Build a context block from stored memories for LLM injection."""
        all_entries = []

        if categories:
            for category in categories:
                all_entries.extend(self.recall(category))
        else:
            all_entries = self.recall()

        # Sort by updated_at descending (most recent first)
        all_entries.sort(key=lambda e: e.updated_at, reverse=True)
        selected = all_entries[:max_entries]

        if not selected:
            return ""

        blocks = ["# Agent Memory\n"]
        for entry in selected:
            blocks.append(
                f"## [{entry.category}] {entry.title}\n"
                f"*Updated: {entry.updated_at}*\n\n"
                f"{entry.content}\n"
            )

        return "\n---\n\n".join(blocks)

    def _slugify(self, text: str) -> str:
        """Convert title to filesystem-safe slug."""
        return (
            text.lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(":", "")
            .replace(".", "")
            .strip("-")
        )

    def _read_entry(self, file_path: Path) -> MemoryEntry | None:
        """Parse a memory file into a MemoryEntry."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return None

        meta = self._read_frontmatter(file_path)
        # Strip frontmatter from content
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()

        return MemoryEntry(
            category=meta.get("category", file_path.parent.name),
            title=meta.get("title", file_path.stem),
            content=body,
            tags=tuple(meta.get("tags", [])),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
        )

    def _read_frontmatter(self, file_path: Path) -> dict:
        """Extract frontmatter metadata from a markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return {}

        if not content.startswith("---"):
            return {}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}

        meta = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()
                # Parse simple list syntax: [a, b, c]
                if value.startswith("[") and value.endswith("]"):
                    value = [
                        v.strip() for v in value[1:-1].split(",") if v.strip()
                    ]
                meta[key.strip()] = value

        return meta

    def _update_index(
        self,
        category: str,
        title: str,
        file_path: str,
        tags: list[str],
    ) -> None:
        """Update the JSON index for fast lookups."""
        index = {}
        if self._index_path.exists():
            try:
                index = json.loads(
                    self._index_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                index = {}

        entries = index.get("entries", [])
        # Remove existing entry for same path
        entries = [e for e in entries if e.get("path") != file_path]
        entries.append({
            "category": category,
            "title": title,
            "path": file_path,
            "tags": tags,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        index["entries"] = entries
        index["last_updated"] = datetime.now(timezone.utc).isoformat()

        self._index_path.write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
```

### TypeScript

```typescript
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { basename, dirname, join } from "path";
import { globSync } from "glob";

interface MemoryEntry {
  readonly category: string;
  readonly title: string;
  readonly content: string;
  readonly tags: readonly string[];
  readonly createdAt: string;
  readonly updatedAt: string;
}

function entryToFileContent(entry: MemoryEntry): string {
  const frontmatter = [
    "---",
    `title: ${entry.title}`,
    `category: ${entry.category}`,
    `tags: [${entry.tags.join(", ")}]`,
    `created_at: ${entry.createdAt}`,
    `updated_at: ${entry.updatedAt}`,
    "---",
  ].join("\n");

  return `${frontmatter}\n\n${entry.content}\n`;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\s/]/g, "-")
    .replace(/[:.]/g, "")
    .replace(/^-+|-+$/g, "");
}

function parseFrontmatter(content: string): Record<string, string | string[]> {
  if (!content.startsWith("---")) return {};
  const parts = content.split("---", 3);
  if (parts.length < 3) return {};

  const meta: Record<string, string | string[]> = {};
  for (const line of parts[1].trim().split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    let value = line.slice(colonIdx + 1).trim();
    if (value.startsWith("[") && value.endsWith("]")) {
      meta[key] = value
        .slice(1, -1)
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean);
    } else {
      meta[key] = value;
    }
  }
  return meta;
}

class FilesystemMemory {
  private readonly basePath: string;
  private readonly indexPath: string;

  constructor(basePath: string) {
    this.basePath = basePath;
    this.indexPath = join(basePath, "index.json");
    mkdirSync(basePath, { recursive: true });
  }

  store(params: {
    category: string;
    title: string;
    content: string;
    tags?: string[];
  }): string {
    const now = new Date().toISOString();
    const slug = slugify(params.title);
    const datePrefix = now.slice(0, 10);

    const categoryDir = join(this.basePath, params.category);
    mkdirSync(categoryDir, { recursive: true });

    const filePath = join(categoryDir, `${datePrefix}-${slug}.md`);

    let createdAt = now;
    if (existsSync(filePath)) {
      const existing = parseFrontmatter(readFileSync(filePath, "utf-8"));
      if (typeof existing.created_at === "string") {
        createdAt = existing.created_at;
      }
    }

    const entry: MemoryEntry = {
      category: params.category,
      title: params.title,
      content: params.content,
      tags: Object.freeze([...(params.tags ?? [])]),
      createdAt,
      updatedAt: now,
    };

    writeFileSync(filePath, entryToFileContent(entry), "utf-8");
    this.updateIndex(params.category, params.title, filePath, params.tags ?? []);
    return filePath;
  }

  recall(category?: string): MemoryEntry[] {
    const searchPath = category ? join(this.basePath, category) : this.basePath;
    const files = globSync(join(searchPath, "**/*.md"));

    return files
      .sort()
      .map((f) => this.readEntry(f))
      .filter((e): e is MemoryEntry => e !== null);
  }

  search(query: string): MemoryEntry[] {
    const queryLower = query.toLowerCase();
    const files = globSync(join(this.basePath, "**/*.md"));

    return files
      .filter((f) => {
        try {
          return readFileSync(f, "utf-8").toLowerCase().includes(queryLower);
        } catch {
          return false;
        }
      })
      .map((f) => this.readEntry(f))
      .filter((e): e is MemoryEntry => e !== null);
  }

  buildContextBlock(categories?: string[], maxEntries = 10): string {
    let entries: MemoryEntry[] = [];

    if (categories) {
      for (const cat of categories) {
        entries.push(...this.recall(cat));
      }
    } else {
      entries = this.recall();
    }

    entries.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    const selected = entries.slice(0, maxEntries);

    if (selected.length === 0) return "";

    const blocks = selected.map(
      (entry) =>
        `## [${entry.category}] ${entry.title}\n` +
        `*Updated: ${entry.updatedAt}*\n\n` +
        `${entry.content}\n`
    );

    return "# Agent Memory\n\n" + blocks.join("\n---\n\n");
  }

  private readEntry(filePath: string): MemoryEntry | null {
    try {
      const content = readFileSync(filePath, "utf-8");
      const meta = parseFrontmatter(content);

      let body = content;
      if (content.startsWith("---")) {
        const parts = content.split("---", 3);
        if (parts.length >= 3) body = parts[2].trim();
      }

      const tags = Array.isArray(meta.tags) ? meta.tags : [];

      return {
        category: (meta.category as string) ?? basename(dirname(filePath)),
        title: (meta.title as string) ?? basename(filePath, ".md"),
        content: body,
        tags: Object.freeze(tags),
        createdAt: (meta.created_at as string) ?? "",
        updatedAt: (meta.updated_at as string) ?? "",
      };
    } catch {
      return null;
    }
  }

  private updateIndex(
    category: string,
    title: string,
    filePath: string,
    tags: string[]
  ): void {
    let index: { entries: Array<Record<string, unknown>>; last_updated: string } = {
      entries: [],
      last_updated: "",
    };

    if (existsSync(this.indexPath)) {
      try {
        index = JSON.parse(readFileSync(this.indexPath, "utf-8"));
      } catch {
        // Reset on corruption
      }
    }

    index.entries = index.entries.filter((e) => e.path !== filePath);
    index.entries.push({
      category,
      title,
      path: filePath,
      tags,
      updated_at: new Date().toISOString(),
    });
    index.last_updated = new Date().toISOString();

    writeFileSync(this.indexPath, JSON.stringify(index, null, 2), "utf-8");
  }
}

export { FilesystemMemory, MemoryEntry };
```

## Trade-offs

| Pros | Cons |
|------|------|
| Human-readable -- developers can inspect and edit memory directly | No semantic search without adding an embedding layer |
| Works with git -- memory is versioned, diffable, and reviewable | Text search is limited compared to vector similarity |
| Zero infrastructure -- no databases, no APIs, no services | Does not scale well beyond thousands of memory files |
| Tool-compatible -- grep, editors, CI all work out of the box | File organization must be designed upfront; hard to reorganize later |
| Transparent -- no opaque embeddings or black-box retrieval | Concurrent writes from multiple agents need coordination |
| Debuggable -- you can always see exactly what the agent "knows" | Frontmatter parsing is fragile without a proper YAML library |

## When to Use

- Coding agents and developer tools where transparency matters
- Projects already using git for version control
- Teams that want to review and approve agent memory changes
- Single-agent systems where concurrent access is not an issue
- Early-stage projects that need memory without infrastructure investment

## When NOT to Use

- High-volume systems with thousands of memory writes per hour
- When you need semantic similarity search over memory
- Multi-agent systems with concurrent memory access from many writers
- When memory content is sensitive and should not be on disk in plaintext
- When the volume of memories makes directory browsing impractical (use Episodic Memory instead)

## Related Patterns

- **Episodic Memory**: Add a semantic index layer over filesystem memory when text search becomes insufficient.
- **System Prompt Architecture** (Construction): The top-level memory file (CLAUDE.md, .cursorrules) is effectively a modular system prompt that persists on disk.
- **Context Rot Detection** (Evaluation): Memory files can become stale. Use context rot detection to identify and clean up outdated memories.

## Real-World Examples

- **Claude Code (CLAUDE.md)**: Stores project context, coding standards, and agent instructions in a markdown file at the project root. The agent reads this file at session start and writes back learnings. This is the canonical example of filesystem-as-memory.
- **Cursor (.cursorrules)**: Stores editor-specific agent instructions in a dotfile. Simpler than CLAUDE.md but the same pattern.
- **Devin (knowledge base)**: Maintains a directory of knowledge files that the agent reads and updates as it works on tasks.
- **Aider (.aider.conf.yml + conventions)**: Uses configuration files and convention documents as persistent memory for coding context.
- **CLAUDE.md hierarchy**: Claude Code uses a layered filesystem memory -- global (~/.claude/CLAUDE.md), project (./CLAUDE.md), and directory-level files. This hierarchy mirrors the progressive disclosure pattern applied to persistent memory.
