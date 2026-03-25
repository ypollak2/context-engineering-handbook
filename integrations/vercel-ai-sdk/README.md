# Vercel AI SDK Integration

Implements 3 Context Engineering Handbook patterns using the Vercel AI SDK (`ai` package v4+), focused on edge deployment, streaming, and Next.js applications.

## Patterns Covered

| Pattern | File | Why This Framework |
|---------|------|--------------------|
| [Progressive Disclosure](../../patterns/construction/progressive-disclosure.md) | `progressive-disclosure.ts` | Custom middleware that stages system prompt sections based on conversation state |
| [Conversation Compaction](../../patterns/compression/conversation-compaction.md) | `conversation-compaction.ts` | Custom `onFinish` handler that compacts history after each response |
| [Error Preservation](../../patterns/optimization/error-preservation.md) | `error-preservation.ts` | Custom error handler that preserves full context for retry |

## Why These Patterns

The Vercel AI SDK is designed for streaming-first, edge-deployed AI applications. These three patterns target the areas where the SDK's middleware and streaming architecture are strongest:

1. **Progressive Disclosure**: The SDK's middleware system enables intercepting and modifying messages before they reach the model. Progressive disclosure fits naturally as middleware that dynamically adjusts the system prompt.

2. **Conversation Compaction**: The SDK's `onFinish` callback provides a hook to process conversations after each response. Compaction triggers here to manage history before the next turn.

3. **Error Preservation**: The SDK's error handling and retry mechanisms benefit from structured error context. Preserving full error details enables smarter retries without losing the debugging signal.

## Prerequisites

- Node.js 20+
- An OpenAI API key (set `OPENAI_API_KEY` environment variable)
- `ai` package v4+

## Quick Start

```bash
cd integrations/vercel-ai-sdk/typescript
npm install
export OPENAI_API_KEY="sk-..."

npx tsx progressive-disclosure.ts
npx tsx conversation-compaction.ts
npx tsx error-preservation.ts
```

## Framework-Specific Notes

### How Vercel AI SDK Differs from the Generic Patterns

1. **Middleware architecture**: The SDK uses middleware to transform messages before and after model calls. Patterns are implemented as middleware functions rather than wrapper classes.

2. **Streaming-first**: All responses are streamed by default. The patterns account for streaming behavior -- for example, compaction happens in `onFinish` after the full response is available.

3. **Edge runtime**: The SDK is designed for edge deployment (Vercel Edge Functions, Cloudflare Workers). The patterns avoid heavy dependencies and keep processing lightweight.

4. **Core message format**: The SDK uses its own `CoreMessage` type with `role` and `content` fields. The patterns work with this format directly.

5. **Provider abstraction**: The SDK supports multiple providers (OpenAI, Anthropic, Google) through a unified interface. The patterns are provider-agnostic.
