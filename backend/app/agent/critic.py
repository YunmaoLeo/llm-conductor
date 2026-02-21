"""GPT-4o Music Critic: Provides intelligent feedback for generation refinement."""

import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from app.core.feature_extractor import MusicFeatures

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are an expert music critic analyzing computer-generated MIDI music.
You receive extracted musical features from a generated track and the original user request.
Your job is to provide specific, actionable feedback to improve the next generation attempt.

You are NOT listening to audio — you are analyzing symbolic music features (note counts, pitch ranges, density, key detection, rhythm patterns, velocity statistics).

**How to interpret features:**
- note_density: notes per second. Melody: 2-6 ideal. Harmony: 1-4. Bass: 1-3. Rhythm: 3-8.
- pitch_range: (min, max) MIDI pitch. 60=C4 (middle C). Melody should be 60-96, bass 28-55.
- in_key_ratio: fraction of notes on the detected key's scale. >0.85 = good, <0.7 = poor.
- silence_ratio: fraction of time windows with no notes. >0.3 = too many gaps.
- velocity_mean/std: dynamics. std < 5 = flat/robotic dynamics.
- onset_density_curve: notes per 2-second window. High variance = erratic rhythm.
- estimated_key: detected key and confidence. Confidence < 0.5 = ambiguous tonality.
- chord_progression: detected chords in 2-second windows.

**Output JSON with exactly these fields:**
{
  "quality_score": 0.0-1.0,
  "diagnosis": "One-sentence overall assessment",
  "corrections": [
    "Specific actionable instruction for the musician (e.g., 'Reduce note density to 3 notes/sec')",
    "Another correction..."
  ]
}

Keep corrections to 2-4 items maximum. Each correction must be a concrete instruction
that can be directly appended to the musician's generation prompt.
Focus on the BIGGEST problems first."""


class GPTCritic:
    """Uses GPT-4o-mini to provide intelligent music critique for quality gate refinement."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    async def critique(
        self,
        features: MusicFeatures,
        user_intent: str,
        role: str,
        attempt_number: int = 1,
        composition_key: Optional[tuple[str, str]] = None,
        composition_tempo: float = 0.0,
    ) -> dict:
        """Analyze generated music features and return structured feedback.

        Args:
            features: Extracted features from the generated MIDI
            user_intent: What the user originally asked for
            role: Track role (melody, harmony, bass, rhythm)
            attempt_number: Which retry attempt this is
            composition_key: The composition's established key
            composition_tempo: The composition's established tempo

        Returns:
            Dict with quality_score, diagnosis, and corrections list
        """
        # Build feature summary for GPT
        feature_summary = self._format_features(features, role, composition_key, composition_tempo)

        user_prompt = (
            f"User's request: \"{user_intent}\"\n"
            f"Track role: {role}\n"
            f"Attempt: {attempt_number}\n\n"
            f"{feature_summary}\n\n"
            f"Analyze this generation and provide corrections to improve quality."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            if not content:
                return self._fallback_result()

            result = json.loads(content)

            # Validate expected fields
            if "corrections" not in result:
                result["corrections"] = []
            if "quality_score" not in result:
                result["quality_score"] = 0.5
            if "diagnosis" not in result:
                result["diagnosis"] = "No diagnosis provided"

            logger.info(
                f"GPT critic: score={result['quality_score']:.2f}, "
                f"{len(result['corrections'])} corrections, "
                f"diagnosis={result['diagnosis'][:80]}"
            )

            return result

        except Exception as e:
            logger.warning(f"GPT critic failed: {e}")
            return self._fallback_result()

    def _format_features(
        self,
        features: MusicFeatures,
        role: str,
        composition_key: Optional[tuple[str, str]],
        composition_tempo: float,
    ) -> str:
        """Format MusicFeatures into a readable summary for GPT."""
        parts = ["Generated track features:"]
        parts.append(f"  Notes: {features.note_count}")
        parts.append(f"  Duration: {features.duration_seconds:.1f}s")
        parts.append(f"  Note density: {features.note_density:.1f} notes/sec")
        parts.append(f"  Pitch range: {features.pitch_range[0]}-{features.pitch_range[1]} "
                      f"(mean={features.pitch_mean:.0f}, std={features.pitch_std:.1f})")

        if features.estimated_key:
            key_name, mode = features.estimated_key
            parts.append(f"  Detected key: {key_name} {mode} (confidence={features.key_confidence:.2f})")
            parts.append(f"  In-key ratio: {features.in_key_ratio:.0%}")
        else:
            parts.append("  Detected key: none (ambiguous tonality)")

        if features.estimated_tempo > 0:
            parts.append(f"  Detected tempo: {features.estimated_tempo:.0f} BPM")

        parts.append(f"  Silence ratio: {features.silence_ratio:.0%}")
        parts.append(f"  Velocity: mean={features.velocity_mean:.0f}, std={features.velocity_std:.1f}")

        if features.chord_progression:
            chords = features.chord_progression[:8]
            parts.append(f"  Chords: {' | '.join(chords)}")

        # Rhythm analysis
        if features.onset_density_curve and len(features.onset_density_curve) > 2:
            import numpy as np
            curve = [v for v in features.onset_density_curve if v > 0]
            if curve:
                cv = float(np.std(curve) / np.mean(curve)) if np.mean(curve) > 0 else 0
                parts.append(f"  Rhythm regularity: CV={cv:.2f} ({'steady' if cv < 0.5 else 'erratic' if cv > 1.5 else 'moderate'})")

        # Composition context
        if composition_key:
            parts.append(f"\n  Composition key: {composition_key[0]} {composition_key[1]}")
        if composition_tempo > 0:
            parts.append(f"  Composition tempo: {composition_tempo:.0f} BPM")

        # Role expectations
        role_expectations = {
            "melody": "Expected: single-voice line, pitch range 60-96, density 2-6 notes/sec, expressive dynamics",
            "harmony": "Expected: chords/arpeggios, pitch range 48-84, density 1-4 notes/sec, smooth voice leading",
            "bass": "Expected: single-voice low line, pitch range 28-55, density 1-3 notes/sec, steady rhythm",
            "rhythm": "Expected: percussive patterns, density 3-8 notes/sec, consistent pulse, short durations",
        }
        if role in role_expectations:
            parts.append(f"\n  {role_expectations[role]}")

        return "\n".join(parts)

    def _fallback_result(self) -> dict:
        """Return a neutral result when GPT critique fails."""
        return {
            "quality_score": 0.5,
            "diagnosis": "Critique unavailable",
            "corrections": [],
        }
