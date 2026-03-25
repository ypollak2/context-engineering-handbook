# Interactive Pattern Finder

A single-page decision tree that helps you choose the right context engineering pattern for your problem.

## Usage

Open `index.html` directly in any modern browser -- no build step, no server, no installation required.

```
open index.html
```

Or serve it locally if you prefer:

```
python3 -m http.server 8000
# then visit http://localhost:8000/interactive/
```

## How It Works

1. Answer a series of questions about your current challenge.
2. The tool narrows down to a recommended pattern with a description and link to the full guide.
3. Each result also suggests related patterns you might want to explore.
4. Use the **Back** button to revise an answer or **Start Over** to begin fresh.

## Features

- Dark mode support (follows system preference)
- Fully responsive (works on mobile and desktop)
- Zero external dependencies
- Color-coded categories for quick visual identification
- Animated card transitions

## Categories Covered

| Category | Color | Patterns |
|----------|-------|----------|
| Compression | Red | Conversation Compaction, Observation Masking |
| Retrieval | Blue | RAG Context Assembly, Semantic Tool Selection, Just-in-Time Retrieval |
| Construction | Purple | System Prompt Architecture, Progressive Disclosure, Few-Shot Curation |
| Isolation | Amber | Sub-Agent Delegation, Multi-Agent Context Orchestration |
| Persistence | Green | Episodic Memory, Filesystem-as-Memory |
| Optimization | Pink | KV-Cache Optimization, Error Preservation |
| Evaluation | Cyan | Context Rot Detection |
