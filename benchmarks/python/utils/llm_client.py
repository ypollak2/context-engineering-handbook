"""Unified LLM client supporting OpenAI and Anthropic APIs."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import anthropic
import openai


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class LLMResponse:
    """Immutable response from an LLM call."""

    content: str
    model: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Message:
    """Immutable chat message."""

    role: str
    content: str


def _detect_provider(model: str) -> Provider:
    """Detect the provider from the model name."""
    anthropic_prefixes = ("claude",)
    if model.startswith(anthropic_prefixes):
        return Provider.ANTHROPIC
    return Provider.OPENAI


def _validate_api_key(provider: Provider) -> str:
    """Validate and return the API key for a provider."""
    if provider == Provider.OPENAI:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI models."
            )
        return key
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic models."
        )
    return key


@dataclass(frozen=True)
class LLMClient:
    """Unified client for OpenAI and Anthropic LLMs.

    Immutable after construction. Each call creates fresh API client instances
    to avoid shared mutable state.
    """

    model: str
    max_tokens: int = 4096
    temperature: float = 0.0

    def _provider(self) -> Provider:
        return _detect_provider(self.model)

    def complete(
        self,
        messages: Sequence[Message],
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request and return an immutable response."""
        provider = self._provider()
        _validate_api_key(provider)

        if provider == Provider.OPENAI:
            return self._openai_complete(messages, system)
        return self._anthropic_complete(messages, system)

    def _openai_complete(
        self,
        messages: Sequence[Message],
        system: str | None,
    ) -> LLMResponse:
        client = openai.OpenAI()
        formatted: list[dict[str, str]] = []
        if system:
            formatted.append({"role": "system", "content": system})
        for msg in messages:
            formatted.append({"role": msg.role, "content": msg.content})

        start = time.monotonic()
        response = client.chat.completions.create(
            model=self.model,
            messages=formatted,  # type: ignore[arg-type]
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider=Provider.OPENAI,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    def _anthropic_complete(
        self,
        messages: Sequence[Message],
        system: str | None,
    ) -> LLMResponse:
        client = anthropic.Anthropic()
        formatted = [{"role": msg.role, "content": msg.content} for msg in messages]

        kwargs: dict = {
            "model": self.model,
            "messages": formatted,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system

        start = time.monotonic()
        response = client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=self.model,
            provider=Provider.ANTHROPIC,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
