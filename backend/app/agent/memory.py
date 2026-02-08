"""Session memory for the Conductor Agent."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.feature_extractor import MusicFeatures
from app.core.prompt_builder import GenerationPlan


@dataclass
class EvaluationResult:
    """Evaluation outcome for a single generation."""

    score: float  # 0-1 overall quality
    verdict: str  # "accept", "refine", "reject"
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class GenerationRecord:
    """Record of a single generation attempt."""

    id: str
    iteration: int
    timestamp: str
    prompt: str
    plan: dict  # Serialized GenerationPlan
    token_count: int
    num_notes: int
    duration_seconds: float
    features: dict  # Serialized MusicFeatures
    evaluation: dict  # Serialized EvaluationResult
    action_taken: str  # "accept", "refine", "reject"
    text_description: str = ""


class Memory:
    """In-memory session state for the Conductor Agent."""

    def __init__(self):
        self._records: list[GenerationRecord] = []
        self._user_intent: str = ""
        self._composition_state: dict = {}

    def add_record(self, record: GenerationRecord):
        self._records.append(record)

    def get_history(self) -> list[GenerationRecord]:
        return list(self._records)

    def get_latest(self) -> Optional[GenerationRecord]:
        return self._records[-1] if self._records else None

    @property
    def user_intent(self) -> str:
        return self._user_intent

    @user_intent.setter
    def user_intent(self, value: str):
        self._user_intent = value

    @property
    def iteration_count(self) -> int:
        return len(self._records)

    def get_best_record(self) -> Optional[GenerationRecord]:
        """Return the record with the highest evaluation score."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r.evaluation.get("score", 0))

    def clear(self):
        self._records.clear()
        self._user_intent = ""
        self._composition_state = {}
