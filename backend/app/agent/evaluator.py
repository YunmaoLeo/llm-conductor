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
    ):
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

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

        # 4. Improvement over history (weight: low)
        if history:
            improvement_score = self._check_improvement(features, history, strengths, weaknesses)
            scores.append(improvement_score * 0.5)

        # Compute weighted average
        total_score = sum(scores) / (5.5 if history else 5.0)
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
