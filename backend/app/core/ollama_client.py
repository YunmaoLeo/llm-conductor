"""Ollama API client for MIDI-LLM generation."""

import re
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

import httpx

from app.config import settings

# Regex to extract MIDI token IDs from Ollama response text
# Matches patterns like <|midi_token_12345|>
MIDI_TOKEN_PATTERN = re.compile(r"<\|midi_token_(\d+)\|>")

# BOS token separating text description from MIDI tokens
MIDI_BOS_TOKEN_ID = 55026


@dataclass
class GenerationResult:
    """Result from a single MIDI-LLM generation."""

    text_description: str
    midi_token_ids: list[int]
    raw_response: str
    generation_time: float
    done_reason: str = ""
    token_count: int = 0

    def __post_init__(self):
        self.token_count = len(self.midi_token_ids)


@dataclass
class StreamChunk:
    """A single chunk from streaming generation."""

    text: str
    done: bool
    done_reason: str = ""


def extract_midi_tokens(response_text: str) -> tuple[str, list[int]]:
    """Extract MIDI token IDs from Ollama response text.

    The model outputs a text description followed by MIDI tokens in the format
    <|midi_token_XXXXX|>. Token 55026 is the BOS separator and is skipped.

    The XXXXX values are already in AMT/GPT2 token space (0-55025).
    No vocabulary shift is needed (unlike HuggingFace reference code).

    AMT tokenization uses:
    - Time tokens: 0-9999
    - Duration tokens: 10000-10999
    - Note tokens: 11000+
    - Separator: 55025
    - BOS: 55026

    Args:
        response_text: Raw response string from Ollama.

    Returns:
        Tuple of (text_description, list_of_token_ids).
    """
    matches = MIDI_TOKEN_PATTERN.findall(response_text)
    raw_ids = [int(m) for m in matches]

    # Remove ALL BOS tokens (55026) from the sequence
    token_ids = [t for t in raw_ids if t != MIDI_BOS_TOKEN_ID]

    # Find the start of valid MIDI triples. The first token of a valid triple
    # should be a time token (< 10000). Skip any leading non-time tokens.
    start_idx = 0
    for i, t in enumerate(token_ids):
        if t < 10000:  # Time token range
            start_idx = i
            break
    token_ids = token_ids[start_idx:]

    # Text description is everything before the first MIDI token
    first_token_pos = response_text.find("<|midi_token_")
    if first_token_pos >= 0:
        text_desc = response_text[:first_token_pos].strip()
    else:
        text_desc = response_text.strip()

    return text_desc, token_ids


class OllamaClient:
    """Async client for communicating with Ollama's MIDI-LLM model."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            data = resp.json()
            model_names = [m["name"] for m in data.get("models", [])]
            return any(self.model in name for name in model_names)
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Non-streaming generation. Sends prompt and waits for full response.

        Args:
            prompt: User prompt text (appended after system prompt).
            system: System prompt override.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with extracted MIDI tokens.
        """
        system = system or settings.system_prompt
        temperature = temperature if temperature is not None else settings.default_temperature
        top_p = top_p if top_p is not None else settings.default_top_p
        max_tokens = max_tokens or settings.default_max_tokens

        # Build the full prompt: system + user + trailing space (matching training format)
        full_prompt = system + prompt + " "

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        start_time = time.monotonic()
        resp = await self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        generation_time = time.monotonic() - start_time

        data = resp.json()
        raw_response = data.get("response", "")
        done_reason = data.get("done_reason", "")

        text_desc, token_ids = extract_midi_tokens(raw_response)

        return GenerationResult(
            text_description=text_desc,
            midi_token_ids=token_ids,
            raw_response=raw_response,
            generation_time=generation_time,
            done_reason=done_reason,
        )

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming generation. Yields chunks as they arrive.

        IMPORTANT: MIDI tokens may be fragmented across chunks.
        Callers must accumulate all chunks and parse tokens from the
        complete response after the stream ends.

        Args:
            prompt: User prompt text.
            system: System prompt override.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.

        Yields:
            StreamChunk with partial response text.
        """
        system = system or settings.system_prompt
        temperature = temperature if temperature is not None else settings.default_temperature
        top_p = top_p if top_p is not None else settings.default_top_p
        max_tokens = max_tokens or settings.default_max_tokens

        full_prompt = system + prompt + " "

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        async with self._client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                import json

                chunk_data = json.loads(line)
                yield StreamChunk(
                    text=chunk_data.get("response", ""),
                    done=chunk_data.get("done", False),
                    done_reason=chunk_data.get("done_reason", ""),
                )

    async def generate_stream_full(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Streaming generation that accumulates into a full result.

        Uses streaming internally but returns the same GenerationResult
        as non-streaming generate(). Useful for progress tracking while
        still getting the complete parsed result.
        """
        start_time = time.monotonic()
        buffer = []

        async for chunk in self.generate_stream(
            prompt=prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            buffer.append(chunk.text)
            done_reason = chunk.done_reason

        generation_time = time.monotonic() - start_time
        raw_response = "".join(buffer)
        text_desc, token_ids = extract_midi_tokens(raw_response)

        return GenerationResult(
            text_description=text_desc,
            midi_token_ids=token_ids,
            raw_response=raw_response,
            generation_time=generation_time,
            done_reason=done_reason,
        )
