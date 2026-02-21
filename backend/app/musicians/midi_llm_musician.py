"""MIDI-LLM Musician: Generates MIDI tokens via Ollama."""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# MIDI token extraction pattern
MIDI_TOKEN_PATTERN = re.compile(r"<\|midi_token_(\d+)\|>")
MIDI_BOS_TOKEN_ID = 55026  # BOS separator token

# Role-specific musical hints prepended to generation prompts
ROLE_HINTS = {
    "melody": (
        "Generate a clear single-voice melodic line. "
        "Use stepwise motion (scale runs, neighbor notes) as the foundation, "
        "with occasional interval leaps (3rds, 5ths) for interest. "
        "Include rests between phrases for natural breathing — aim for 4-bar phrases. "
        "Stay in the middle-to-upper register (C4-C6, MIDI 60-96). "
        "Use moderate density (2-5 notes/sec). "
        "Vary velocity for expressive dynamics. "
        "Avoid: dense chords, very low notes, or monotonous repeated pitches. "
    ),
    "harmony": (
        "Generate chordal accompaniment or arpeggiated patterns that support the melody. "
        "Use smooth voice leading — move each voice to the nearest chord tone. "
        "Common patterns: block chords on beats, arpeggiated broken chords, or sustained pads. "
        "Stay in the middle register (C3-C5, MIDI 48-84). "
        "Use low-to-moderate density (1-4 notes/sec). "
        "Follow standard harmonic rhythm: change chords every 1-2 bars. "
        "Avoid: competing with melody register, random dissonances, or overly busy patterns. "
    ),
    "bass": (
        "Generate a bass line that provides harmonic foundation. "
        "Stay in low register (E1-G3, MIDI 28-55). "
        "Use root notes of chords on strong beats, with passing tones and approach notes. "
        "Common patterns: walking bass (quarter notes), pedal tones, or root-fifth alternation. "
        "Maintain steady rhythmic pulse — bass anchors the tempo. "
        "Use low density (1-3 notes/sec). "
        "Avoid: high register notes, dense chords, or rhythmically erratic patterns. "
    ),
    "rhythm": (
        "Generate a rhythmic pattern with consistent, repeating pulse. "
        "Establish a clear groove within the first 2 bars and maintain it. "
        "Use percussive articulation: short note durations, strong velocity on downbeats. "
        "Common patterns: straight eighths, syncopated grooves, or shuffle feel. "
        "Keep density moderate-to-high (3-8 notes/sec) but REGULAR. "
        "Repeat rhythmic cells — consistency is more important than variety. "
        "Avoid: melodic movement, long sustained notes, or irregular/random rhythms. "
    ),
}

# Role-specific hard constraints (things to NEVER do)
ROLE_NEGATIVE_CONSTRAINTS = {
    "melody": (
        "NEVER use more than 2 simultaneous notes. "
        "NEVER go below MIDI pitch 55 (G3). "
        "DO NOT stay on a single repeated pitch for more than 4 beats. "
    ),
    "harmony": (
        "NEVER play in the same octave as the melody (stay below C5/MIDI 72). "
        "NEVER use more than 4 simultaneous notes. "
        "DO NOT use rapid single-note runs — chords and arpeggios only. "
    ),
    "bass": (
        "NEVER go above MIDI pitch 60 (C4). "
        "NEVER use chords — single notes only. "
        "DO NOT use fast ornamental runs. "
    ),
    "rhythm": (
        "NEVER use sustained notes longer than 1 beat. "
        "NEVER create melodic lines — pitch movement should be minimal. "
        "DO NOT vary the pattern significantly after establishing the groove. "
    ),
}


def build_dynamic_constraints(
    role: str,
    features: "MusicFeatures | None" = None,
) -> str:
    """Generate context-aware negative constraints based on role and features.

    Args:
        role: Track role (melody, harmony, bass, rhythm)
        features: Features from a previous failed generation (for adaptive constraints)

    Returns:
        Constraint string to append to the prompt
    """
    parts = []

    # Static role constraints
    static = ROLE_NEGATIVE_CONSTRAINTS.get(role, "")
    if static:
        parts.append(static)

    # Adaptive constraints from previous generation features
    if features is not None:
        if features.note_density > 12:
            parts.append(
                f"CRITICAL: Previous attempt had {features.note_density:.1f} notes/sec (TOO DENSE). "
                f"Use HALF as many notes. Leave space between phrases. "
            )
        elif features.note_density < 0.5:
            parts.append(
                f"CRITICAL: Previous attempt had only {features.note_density:.1f} notes/sec (TOO SPARSE). "
                f"Generate more notes — aim for at least 2 notes/sec. "
            )

        if hasattr(features, 'in_key_ratio') and features.in_key_ratio < 0.6:
            key_info = ""
            if features.estimated_key:
                key_info = f" ({features.estimated_key[0]} {features.estimated_key[1]})"
            parts.append(
                f"CRITICAL: Only {features.in_key_ratio:.0%} of notes were in key{key_info}. "
                f"Stay strictly on scale degrees. Avoid chromatic notes. "
            )

        if hasattr(features, 'silence_ratio') and features.silence_ratio > 0.4:
            parts.append(
                f"CRITICAL: {features.silence_ratio:.0%} silence detected. "
                f"Fill gaps with sustained notes or passing tones. No long empty sections. "
            )

        if hasattr(features, 'velocity_std') and features.velocity_std < 3:
            parts.append(
                "Dynamics are flat — vary note velocity between 50-100 for expression. "
            )

        # Register violations
        pitch_mid = (features.pitch_range[0] + features.pitch_range[1]) / 2
        if role == "bass" and pitch_mid > 55:
            parts.append(
                f"CRITICAL: Bass register too high (mean pitch {pitch_mid:.0f}). "
                f"Stay below MIDI 55 (G3). Use notes in the E1-G3 range. "
            )
        elif role == "melody" and pitch_mid < 55:
            parts.append(
                f"CRITICAL: Melody register too low (mean pitch {pitch_mid:.0f}). "
                f"Move to mid-upper register (MIDI 60-96, C4-C6). "
            )

    if not parts:
        return ""

    return "CONSTRAINTS: " + "".join(parts)


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
        role: Optional[str] = None,
        previous_features: "MusicFeatures | None" = None,
        example_context: str = "",
    ) -> MusicianGenerationResult:
        """Generate MIDI tokens from instruction.

        Args:
            instruction: Natural language music description
            temperature: Sampling temperature (default from settings)
            top_p: Nucleus sampling parameter (default from settings)
            max_tokens: Maximum tokens to generate (default from settings)
            role: Track role for role-specific hints and constraints
            previous_features: Features from a previous failed attempt (for adaptive constraints)
            example_context: Pre-formatted few-shot examples from MIDIExampleDB

        Returns:
            MusicianGenerationResult with extracted MIDI tokens

        Raises:
            httpx.HTTPError: If Ollama API fails
            ValueError: If no valid MIDI tokens extracted
        """
        # Prepend role-specific musical hints
        role_hint = ROLE_HINTS.get(role, "") if role else ""

        # Build dynamic negative constraints
        constraints = build_dynamic_constraints(role or "", previous_features) if role else ""

        # Build prompt with system template + optional few-shot examples
        full_prompt = settings.system_prompt + role_hint + constraints + example_context + instruction

        # Resolve role-specific temperature if not explicitly provided
        if temperature is None and role and role in settings.role_temperature:
            temperature = settings.role_temperature[role]

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

    async def generate_with_prefix(
        self,
        instruction: str,
        prefix_tokens: list[int],
        prefix_ratio: float = 0.3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: Optional[str] = None,
    ) -> MusicianGenerationResult:
        """Generate MIDI with old tokens as prefix for continuity.

        EXPERIMENTAL: May not work reliably with current Ollama setup.
        Falls back to normal generation if prefix continuation fails.

        Args:
            instruction: New instruction
            prefix_tokens: Previous MIDI token sequence
            prefix_ratio: Fraction of old tokens to use as prefix (0.3 = 30%)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            MusicianGenerationResult with new tokens

        Note:
            This method attempts to guide the model with old tokens,
            but Ollama may not interpret them as actual continuation points.
            Feature-based guidance (via enhanced instructions) is more reliable.
        """
        if not prefix_tokens:
            return await self.generate(instruction, temperature, top_p, max_tokens, role=role)

        # Use first 30% of old tokens as prefix
        prefix_len = int(len(prefix_tokens) * prefix_ratio)
        prefix = prefix_tokens[:prefix_len]

        # Convert tokens back to string format for Ollama prompt
        prefix_str = "".join([f"<|midi_token_{t}|>" for t in prefix])

        # Build prompt: instruction + token prefix
        full_prompt = (
            f"{settings.system_prompt}{instruction}\n\n"
            f"Continue from this musical start:\n{prefix_str}"
        )

        # Resolve role-specific temperature if not explicitly provided
        if temperature is None and role and role in settings.role_temperature:
            temperature = settings.role_temperature[role]

        # Call Ollama with modified prompt
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

        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        response_text = data.get("response", "")
        generation_time_ms = data.get("total_duration", 0) / 1_000_000

        # Extract tokens from response
        all_tokens = self._extract_midi_tokens(response_text)

        if not all_tokens:
            raise ValueError(f"No MIDI tokens found in response: {response_text[:200]}")

        return MusicianGenerationResult(
            instruction=instruction,
            midi_token_ids=all_tokens,
            generation_time_ms=generation_time_ms,
            raw_response=response_text,
        )

    async def generate_with_reference(
        self,
        instruction: str,
        reference_tokens: list[int],
        reference_instrument: Optional[str] = None,
        reference_features: Optional[dict] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: Optional[str] = None,
    ) -> MusicianGenerationResult:
        """Generate MIDI that matches/complements a reference track.

        This method provides both structured musical context and token samples,
        allowing MIDI-LLM to generate complementary parts (e.g., bass matching piano).

        Args:
            instruction: Generation instruction (e.g., "bass line that follows harmony")
            reference_tokens: Tokens from reference track to match/complement
            reference_instrument: Instrument name of reference track (for context)
            reference_features: Structured features dict (key, tempo, density, etc.)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            role: Track role for role-specific temperature

        Returns:
            MusicianGenerationResult with new tokens
        """
        if not reference_tokens:
            return await self.generate(instruction, temperature, top_p, max_tokens, role=role)

        # Build structured musical context (more interpretable than raw tokens)
        structured_context = ""
        if reference_features:
            inst_name = reference_instrument or "unknown"
            parts = [f"[REFERENCE TRACK CONTEXT]"]
            parts.append(f"Reference track ({inst_name}):")
            if reference_features.get("estimated_key"):
                key = reference_features["estimated_key"]
                parts.append(f"  Key: {key[0]} {key[1]} — your track MUST use the same key")
            if reference_features.get("estimated_tempo", 0) > 0:
                parts.append(f"  Tempo: {reference_features['estimated_tempo']:.0f} BPM — match this tempo exactly")
            if reference_features.get("note_density", 0) > 0:
                parts.append(f"  Density: {reference_features['note_density']:.1f} notes/sec")
            if reference_features.get("pitch_range"):
                pr = reference_features["pitch_range"]
                parts.append(f"  Pitch range: {pr[0]}-{pr[1]}")
            if reference_features.get("chord_progression"):
                chords = reference_features["chord_progression"][:8]
                parts.append(f"  Chords: {' | '.join(chords)}")
            parts.append("")
            parts.append("[MUSICAL RELATIONSHIP]")
            parts.append("Your track should COMPLEMENT the reference — do not duplicate it.")
            parts.append("If reference is melody, play supporting harmony or bass.")
            parts.append("If reference is chords, play a contrasting melodic line.")
            parts.append("Match the harmonic rhythm (chord changes) of the reference.")
            structured_context = "\n".join(parts) + "\n\n"

        # Distributed sampling: take tokens from beginning, middle, and end
        # for a broader picture of the reference track's harmonic arc
        n = len(reference_tokens)
        max_sample = 150
        if n <= max_sample:
            reference_sample = reference_tokens
        else:
            # Sample from 3 segments: first 20%, middle 20%, last 20% (~60% coverage, capped)
            seg_size = max_sample // 3
            # Ensure segment boundaries are multiples of 3 (token triples)
            seg_size = (seg_size // 3) * 3
            seg1 = reference_tokens[:seg_size]
            mid_start = n // 2 - seg_size // 2
            mid_start = (mid_start // 3) * 3  # Align to triple boundary
            seg2 = reference_tokens[mid_start:mid_start + seg_size]
            seg3 = reference_tokens[n - seg_size:]
            reference_sample = seg1 + seg2 + seg3

        # Convert reference tokens to string format
        reference_str = "".join([f"<|midi_token_{t}|>" for t in reference_sample])

        # Prepend role-specific hints for reference generation too
        role_hint = ROLE_HINTS.get(role, "") if role else ""

        # Build prompt with both structured and token context
        full_prompt = (
            f"{settings.system_prompt}{role_hint}"
            f"{structured_context}"
            f"[REFERENCE TOKENS]\n{reference_str}\n\n"
            f"[YOUR TASK]\n{instruction}"
        )

        logger.info(
            f"Generating with reference: {len(reference_sample)} reference tokens, "
            f"instrument: {reference_instrument}"
        )

        # Resolve role-specific temperature if not explicitly provided
        if temperature is None and role and role in settings.role_temperature:
            temperature = settings.role_temperature[role]

        # Call Ollama with reference context
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

        start_time = time.time()

        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        response_text = data.get("response", "")
        generation_time_ms = (time.time() - start_time) * 1000

        # Extract tokens from response
        all_tokens = self._extract_midi_tokens(response_text)

        if not all_tokens:
            raise ValueError(f"No MIDI tokens found in response: {response_text[:200]}")

        return MusicianGenerationResult(
            instruction=instruction,
            midi_token_ids=all_tokens,
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

        # Truncate to multiple of 3
        remainder = len(token_ids) % 3
        if remainder != 0:
            token_ids = token_ids[: len(token_ids) - remainder]

        # Validate each triple: time < 10000, duration 10000-10999, pitch >= 11000
        valid_tokens = []
        invalid_count = 0
        for i in range(0, len(token_ids), 3):
            time_tok = token_ids[i]
            dur_tok = token_ids[i + 1]
            pitch_tok = token_ids[i + 2]

            if time_tok < 10000 and 10000 <= dur_tok <= 10999 and pitch_tok >= 11000:
                valid_tokens.extend([time_tok, dur_tok, pitch_tok])
            else:
                invalid_count += 1

        if invalid_count > 0:
            logger.warning(f"Dropped {invalid_count} invalid triples from {len(token_ids) // 3} total")

        if len(valid_tokens) < 3:
            logger.error(f"Only {len(valid_tokens) // 3} valid triples after cleaning")

        return valid_tokens

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
