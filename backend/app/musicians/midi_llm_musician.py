"""MIDI-LLM Musician: Generates MIDI tokens via Ollama."""

import re
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings


# MIDI token extraction pattern
MIDI_TOKEN_PATTERN = re.compile(r"<\|midi_token_(\d+)\|>")
MIDI_BOS_TOKEN_ID = 55026  # BOS separator token


@dataclass
class MusicianGenerationResult:
    """Result from MIDI-LLM generation."""

    instruction: str  # The prompt used
    midi_token_ids: list[int]  # Extracted MIDI tokens
    generation_time_ms: float
    raw_response: str  # Full text response


class MIDILLMMusician:
    """Single MIDI-LLM musician instance for generating one track."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize MIDI-LLM musician.

        Args:
            base_url: Ollama API base URL (default from settings)
            model: Model name (default from settings)
            timeout: Request timeout in seconds (default from settings)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def generate(
        self,
        instruction: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> MusicianGenerationResult:
        """Generate MIDI tokens from instruction.

        Args:
            instruction: Natural language music description
            temperature: Sampling temperature (default from settings)
            top_p: Nucleus sampling parameter (default from settings)
            max_tokens: Maximum tokens to generate (default from settings)

        Returns:
            MusicianGenerationResult with extracted MIDI tokens

        Raises:
            httpx.HTTPError: If Ollama API fails
            ValueError: If no valid MIDI tokens extracted
        """
        # Build prompt with system template
        full_prompt = settings.system_prompt + instruction

        # Request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.default_temperature,
                "top_p": top_p or settings.default_top_p,
                "num_predict": max_tokens or settings.default_max_tokens,
            },
        }

        # Call Ollama API
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        response_text = data.get("response", "")
        generation_time_ms = data.get("total_duration", 0) / 1_000_000  # ns → ms

        # Extract MIDI tokens
        midi_token_ids = self._extract_midi_tokens(response_text)

        if not midi_token_ids:
            raise ValueError(f"No MIDI tokens found in response: {response_text[:200]}")

        return MusicianGenerationResult(
            instruction=instruction,
            midi_token_ids=midi_token_ids,
            generation_time_ms=generation_time_ms,
            raw_response=response_text,
        )

    def _extract_midi_tokens(self, response_text: str) -> list[int]:
        """Extract and clean MIDI token IDs from Ollama response.

        Critical cleaning steps:
        1. Extract all <|midi_token_XXXXX|> patterns
        2. Remove ALL BOS tokens (55026) - not just the first
        3. Skip leading non-time tokens to align triples

        Args:
            response_text: Raw response from Ollama

        Returns:
            List of cleaned MIDI token IDs
        """
        # Extract all token IDs
        matches = MIDI_TOKEN_PATTERN.findall(response_text)
        if not matches:
            return []

        raw_ids = [int(m) for m in matches]

        # Remove ALL BOS tokens (55026) from the sequence
        token_ids = [t for t in raw_ids if t != MIDI_BOS_TOKEN_ID]

        # Find start of valid MIDI triples (first time token < 10000)
        # Time tokens: 0-9999
        # Duration tokens: 10000-10999
        # Note tokens: 11000+
        start_idx = 0
        for i, t in enumerate(token_ids):
            if t < 10000:  # Found first time token
                start_idx = i
                break

        # Skip leading non-time tokens
        token_ids = token_ids[start_idx:]

        return token_ids

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
