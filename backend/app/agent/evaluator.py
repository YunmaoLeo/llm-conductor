"""Rule-based evaluation of generated MIDI music."""

from dataclasses import asdict

from app.agent.memory import EvaluationResult, GenerationRecord
from app.core.feature_extractor import MusicFeatures


# MIDI General MIDI program number to category mapping
INSTRUMENT_CATEGORIES = {
    "piano": range(0, 8),
    "chromatic_percussion": range(8, 16),
    "organ": range(16, 24),
    "guitar": range(24, 32),
    "bass": range(32, 40),
    "strings": range(40, 48),
    "ensemble": range(48, 56),
    "brass": range(56, 64),
    "reed": range(64, 72),
    "pipe": range(72, 80),
    "synth_lead": range(80, 88),
    "synth_pad": range(88, 96),
}

# Keyword to instrument category mapping for intent matching
INTENT_INSTRUMENT_MAP = {
    "piano": "piano",
    "guitar": "guitar",
    "bass": "bass",
    "violin": "strings",
    "cello": "strings",
    "strings": "strings",
    "orchestra": "ensemble",
    "brass": "brass",
    "trumpet": "brass",
    "saxophone": "reed",
    "flute": "pipe",
    "synth": "synth_lead",
}


class Evaluator:
    """Evaluates MIDI generations against user intent using rules."""

    def __init__(
        self,
        accept_threshold: float = 0.65,
        reject_threshold: float = 0.35,
        composition_key: tuple[str, str] | None = None,
        composition_tempo: float = 0.0,
    ):
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.composition_key = composition_key
        self.composition_tempo = composition_tempo

    def evaluate(
        self,
        features: MusicFeatures,
        user_intent: str,
        history: list[GenerationRecord],
    ) -> EvaluationResult:
        """Evaluate a generation against user intent.

        Scoring dimensions:
        - Validity (is it a reasonable piece of music?)
        - Intent matching (does it match what the user asked for?)
        - Quality heuristics (density, range, balance)

        Args:
            features: Extracted music features.
            user_intent: Original user request text.
            history: Previous generation attempts.

        Returns:
            EvaluationResult with score, verdict, and feedback.
        """
        scores: list[float] = []
        strengths: list[str] = []
        weaknesses: list[str] = []
        suggestions: list[str] = []

        # 1. Validity check (weight: high)
        validity_score = self._check_validity(features, strengths, weaknesses, suggestions)
        scores.append(validity_score * 2)  # Double weight

        # 2. Intent matching (weight: high)
        intent_score = self._check_intent(features, user_intent, strengths, weaknesses, suggestions)
        scores.append(intent_score * 2)

        # 3. Quality heuristics (weight: normal)
        quality_score = self._check_quality(features, strengths, weaknesses, suggestions)
        scores.append(quality_score)

        # 4. Key/tempo consistency (weight: normal)
        coherence_score = self._check_coherence(features, strengths, weaknesses, suggestions)
        scores.append(coherence_score)

        # 5. Rhythmic regularity (weight: normal)
        rhythm_score = self._check_rhythm(features, strengths, weaknesses, suggestions)
        scores.append(rhythm_score)

        # 6. Key conformance (weight: normal)
        key_conf_score = self._check_key_conformance(features, strengths, weaknesses, suggestions)
        scores.append(key_conf_score)

        # 7. Improvement over history (weight: low)
        if history:
            improvement_score = self._check_improvement(features, history, strengths, weaknesses)
            scores.append(improvement_score * 0.5)

        # Compute weighted average (validity*2 + intent*2 + quality + coherence + rhythm + key_conf = 8, +0.5 with history)
        total_score = sum(scores) / (8.5 if history else 8.0)
        total_score = max(0.0, min(1.0, total_score))

        # Determine verdict
        if total_score >= self.accept_threshold:
            verdict = "accept"
        elif total_score >= self.reject_threshold:
            verdict = "refine"
        else:
            verdict = "reject"

        return EvaluationResult(
            score=round(total_score, 3),
            verdict=verdict,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
        )

    def _check_validity(
        self, features: MusicFeatures, strengths, weaknesses, suggestions
    ) -> float:
        """Check basic validity of the generation."""
        score = 1.0

        if features.has_excessive_notes:
            score -= 0.5
            weaknesses.append("Excessive simultaneous notes detected")
            suggestions.append("Reduce polyphony density")

        if features.duration_seconds < 3.0:
            score -= 0.4
            weaknesses.append(f"Too short ({features.duration_seconds:.1f}s)")
            suggestions.append("Generate a longer piece")

        if features.note_count < 10:
            score -= 0.4
            weaknesses.append(f"Too few notes ({features.note_count})")
            suggestions.append("Generate more musical content")

        if features.silence_ratio > 0.5:
            score -= 0.3
            weaknesses.append(f"High silence ratio ({features.silence_ratio:.0%})")

        if score > 0.7:
            strengths.append("Valid musical structure")

        return max(0.0, score)

    def _check_intent(
        self, features: MusicFeatures, user_intent: str, strengths, weaknesses, suggestions
    ) -> float:
        """Check how well the generation matches user intent."""
        score = 0.5  # Start neutral
        intent_lower = user_intent.lower()

        # Check instrument matching
        for keyword, category in INTENT_INSTRUMENT_MAP.items():
            if keyword in intent_lower:
                cat_range = INSTRUMENT_CATEGORIES.get(category, range(0))
                if any(prog in cat_range for prog in features.instruments_used):
                    score += 0.15
                    strengths.append(f"Uses requested instrument: {keyword}")
                else:
                    score -= 0.1
                    weaknesses.append(f"Missing requested instrument: {keyword}")
                    suggestions.append(f"Try to include {keyword} sounds")

        # Check energy level
        if "energetic" in intent_lower or "fast" in intent_lower or "upbeat" in intent_lower:
            if features.note_density > 8:
                score += 0.1
                strengths.append("High energy matches request")
            else:
                score -= 0.1
                suggestions.append("Increase note density for more energy")

        if "gentle" in intent_lower or "calm" in intent_lower or "slow" in intent_lower:
            if features.note_density < 8:
                score += 0.1
                strengths.append("Calm energy matches request")
            else:
                score -= 0.1
                suggestions.append("Reduce note density for calmer feel")

        return max(0.0, min(1.0, score))

    def _check_quality(
        self, features: MusicFeatures, strengths, weaknesses, suggestions
    ) -> float:
        """Check general musical quality heuristics."""
        score = 0.5

        # Note density: not too sparse, not too dense
        if 2.0 <= features.note_density <= 20.0:
            score += 0.2
        elif features.note_density > 30.0:
            score -= 0.2
            weaknesses.append("Note density too high")
            suggestions.append("Generate a less dense arrangement")
        elif features.note_density < 1.0:
            score -= 0.2
            weaknesses.append("Note density too low")

        # Pitch range: reasonable spread
        if features.pitch_range[1] - features.pitch_range[0] > 12:
            score += 0.1
            strengths.append("Good pitch range variety")
        else:
            score -= 0.1
            weaknesses.append("Narrow pitch range")

        # Duration: reasonable length
        if 10.0 <= features.duration_seconds <= 120.0:
            score += 0.1

        # Multiple instruments adds richness
        if len(features.instruments_used) >= 2:
            score += 0.1
            strengths.append(f"Uses {len(features.instruments_used)} instruments")

        return max(0.0, min(1.0, score))

    def _check_coherence(
        self, features: MusicFeatures, strengths, weaknesses, suggestions
    ) -> float:
        """Check key and tempo consistency with the composition."""
        score = 0.5  # Neutral baseline

        # Key consistency
        if self.composition_key and features.estimated_key and features.key_confidence > 0.5:
            comp_key, comp_mode = self.composition_key
            track_key, track_mode = features.estimated_key

            # Compatible keys: same key, or relative major/minor
            _relative_map = {
                "C": "A", "G": "E", "D": "B", "A": "F#", "E": "C#", "B": "G#",
                "F": "D", "A#": "G", "D#": "C", "G#": "F", "C#": "A#", "F#": "D#",
            }
            is_same = (track_key == comp_key and track_mode == comp_mode)
            is_relative = (
                comp_mode == "major" and track_mode == "minor"
                and _relative_map.get(comp_key) == track_key
            ) or (
                comp_mode == "minor" and track_mode == "major"
                and _relative_map.get(track_key) == comp_key
            )

            if is_same:
                score += 0.25
                strengths.append(f"Key matches composition ({comp_key} {comp_mode})")
            elif is_relative:
                score += 0.15
                strengths.append(f"Relative key to composition ({track_key} {track_mode})")
            else:
                score -= 0.2
                weaknesses.append(
                    f"Key mismatch: track={track_key} {track_mode}, "
                    f"composition={comp_key} {comp_mode}"
                )
                suggestions.append(
                    f"Regenerate in {comp_key} {comp_mode} to match composition"
                )

        # Tempo consistency (within 15% tolerance)
        if self.composition_tempo > 0 and features.estimated_tempo > 0:
            tempo_ratio = features.estimated_tempo / self.composition_tempo
            if 0.85 <= tempo_ratio <= 1.15:
                score += 0.2
                strengths.append(f"Tempo consistent (~{features.estimated_tempo:.0f} BPM)")
            elif 0.5 <= tempo_ratio <= 2.0:
                # Could be half/double time - acceptable
                score += 0.05
            else:
                score -= 0.15
                weaknesses.append(
                    f"Tempo mismatch: track~{features.estimated_tempo:.0f} BPM, "
                    f"composition~{self.composition_tempo:.0f} BPM"
                )
                suggestions.append(
                    f"Target tempo around {self.composition_tempo:.0f} BPM"
                )

        # Duration check - reasonable length
        if features.duration_seconds >= 15.0:
            score += 0.1
        elif features.duration_seconds < 8.0:
            score -= 0.1
            weaknesses.append(f"Track too short ({features.duration_seconds:.1f}s)")
            suggestions.append("Generate a longer piece (at least 15 seconds)")

        return max(0.0, min(1.0, score))

    def _check_rhythm(
        self, features: MusicFeatures, strengths, weaknesses, suggestions
    ) -> float:
        """Check rhythmic regularity using onset density curve."""
        score = 0.5  # Neutral baseline

        curve = features.onset_density_curve
        if not curve or len(curve) < 3:
            return score

        arr = [v for v in curve if v > 0]  # Non-silent windows only
        if not arr:
            return score

        import numpy as np

        arr_np = np.array(arr)
        mean_val = arr_np.mean()

        if mean_val < 0.01:
            return score

        cv = float(arr_np.std() / mean_val)  # Coefficient of variation

        if cv < 0.5:
            score += 0.15
            strengths.append("Steady rhythmic pattern")
        elif cv < 1.0:
            score += 0.05
        elif cv > 1.5:
            score -= 0.10
            weaknesses.append(f"Erratic rhythm (CV={cv:.2f})")
            suggestions.append("Use more consistent rhythmic patterns")

        # Check for empty windows in the middle (stuttering)
        if len(curve) >= 4:
            middle = curve[1:-1]  # Exclude first and last windows
            empty_middle = sum(1 for d in middle if d == 0)
            if empty_middle > len(middle) * 0.4:
                score -= 0.1
                weaknesses.append("Gaps in the middle of the piece")
                suggestions.append("Fill rhythmic gaps for smoother continuity")

        return max(0.0, min(1.0, score))

    def _check_key_conformance(
        self, features: MusicFeatures, strengths, weaknesses, suggestions
    ) -> float:
        """Check what fraction of notes conform to the detected key."""
        score = 0.5  # Neutral baseline

        if not features.estimated_key or features.key_confidence < 0.5:
            return score  # Can't evaluate without a confident key

        ratio = features.in_key_ratio

        if ratio > 0.85:
            score += 0.15
            strengths.append(f"Strong key conformance ({ratio:.0%} in-key)")
        elif ratio > 0.70:
            score += 0.05
        elif ratio < 0.50:
            score -= 0.15
            weaknesses.append(f"Poor key conformance ({ratio:.0%} in-key)")
            suggestions.append(
                f"Stay closer to {features.estimated_key[0]} {features.estimated_key[1]} scale"
            )

        return max(0.0, min(1.0, score))

    def _check_improvement(
        self, features: MusicFeatures, history: list[GenerationRecord], strengths, weaknesses
    ) -> float:
        """Check if this generation improves over the previous one."""
        if not history:
            return 0.5

        prev = history[-1]
        prev_score = prev.evaluation.get("score", 0)

        # Simple comparison: is the current note count and duration reasonable?
        if features.note_count > prev.num_notes * 0.8:
            strengths.append("Maintained or improved content density")
            return 0.7

        return 0.4
