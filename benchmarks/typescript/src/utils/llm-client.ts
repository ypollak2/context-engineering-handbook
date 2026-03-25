/**
 * Unified LLM client supporting OpenAI and Anthropic APIs.
 */

import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type Provider = "openai" | "anthropic";

export interface LLMResponse {
  readonly content: string;
  readonly model: string;
  readonly provider: Provider;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly latencyMs: number;
  readonly totalTokens: number;
}

export interface Message {
  readonly role: "user" | "assistant";
  readonly content: string;
}

export interface LLMClientConfig {
  readonly model: string;
  readonly maxTokens?: number;
  readonly temperature?: number;
}

function detectProvider(model: string): Provider {
  if (model.startsWith("claude")) {
    return "anthropic";
  }
  return "openai";
}

function validateApiKey(provider: Provider): string {
  if (provider === "openai") {
    const key = process.env.OPENAI_API_KEY ?? "";
    if (!key) {
      throw new Error(
        "OPENAI_API_KEY environment variable is required for OpenAI models."
      );
    }
    return key;
  }
  const key = process.env.ANTHROPIC_API_KEY ?? "";
  if (!key) {
    throw new Error(
      "ANTHROPIC_API_KEY environment variable is required for Anthropic models."
    );
  }
  return key;
}

function createResponse(params: {
  content: string;
  model: string;
  provider: Provider;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
}): LLMResponse {
  return {
    ...params,
    totalTokens: params.inputTokens + params.outputTokens,
  };
}

async function openaiComplete(
  config: LLMClientConfig,
  messages: readonly Message[],
  system?: string
): Promise<LLMResponse> {
  const client = new OpenAI();
  const formatted: Array<{ role: string; content: string }> = [];

  if (system) {
    formatted.push({ role: "system", content: system });
  }
  for (const msg of messages) {
    formatted.push({ role: msg.role, content: msg.content });
  }

  const start = performance.now();
  const response = await client.chat.completions.create({
    model: config.model,
    messages: formatted as any,
    max_tokens: config.maxTokens ?? 4096,
    temperature: config.temperature ?? 0,
  });
  const latencyMs = performance.now() - start;

  const choice = response.choices[0];
  return createResponse({
    content: choice?.message?.content ?? "",
    model: config.model,
    provider: "openai",
    inputTokens: response.usage?.prompt_tokens ?? 0,
    outputTokens: response.usage?.completion_tokens ?? 0,
    latencyMs,
  });
}

async function anthropicComplete(
  config: LLMClientConfig,
  messages: readonly Message[],
  system?: string
): Promise<LLMResponse> {
  const client = new Anthropic();
  const formatted = messages.map((msg) => ({
    role: msg.role as "user" | "assistant",
    content: msg.content,
  }));

  const params: any = {
    model: config.model,
    messages: formatted,
    max_tokens: config.maxTokens ?? 4096,
    temperature: config.temperature ?? 0,
  };
  if (system) {
    params.system = system;
  }

  const start = performance.now();
  const response = await client.messages.create(params);
  const latencyMs = performance.now() - start;

  let content = "";
  for (const block of response.content) {
    if ("text" in block) {
      content += block.text;
    }
  }

  return createResponse({
    content,
    model: config.model,
    provider: "anthropic",
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
    latencyMs,
  });
}

export interface LLMClient {
  readonly model: string;
  complete(messages: readonly Message[], system?: string): Promise<LLMResponse>;
}

export function createLLMClient(config: LLMClientConfig): LLMClient {
  const provider = detectProvider(config.model);
  validateApiKey(provider);

  return {
    model: config.model,
    async complete(
      messages: readonly Message[],
      system?: string
    ): Promise<LLMResponse> {
      if (provider === "openai") {
        return openaiComplete(config, messages, system);
      }
      return anthropicComplete(config, messages, system);
    },
  };
}
