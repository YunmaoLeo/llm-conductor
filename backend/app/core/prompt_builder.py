"""Prompt construction for MIDI-LLM generation."""

from dataclasses import dataclass, field
from typing import Optional

from app.config import settings


@dataclass
class GenerationPlan:
    """High-level description of what to generate, decided by the Conductor."""

    style: Optional[str] = None
    instruments: Optional[list[str]] = None
    energy: Optional[str] = None  # "low", "medium", "high"
    mood: Optional[str] = None
    tempo: Optional[str] = None  # "slow", "moderate", "fast"
    constraints: Optional[list[str]] = None
    raw_user_text: Optional[str] = None
    continuation_context: Optional[str] = None


class PromptBuilder:
    """Transforms GenerationPlan into text prompts for MIDI-LLM."""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or settings.system_prompt

    def build(self, plan: GenerationPlan) -> str:
        """Construct a prompt from a generation plan.

        Combines plan fields into a natural-language music description
        that MIDI-LLM can interpret.

        Args:
            plan: The generation plan from the Conductor.

        Returns:
            Prompt string (without system prompt prefix).
        """
        parts: list[str] = []

        # Start with raw user text if available
        if plan.raw_user_text:
            parts.append(plan.raw_user_text)

        # Add style
        if plan.style:
            parts.append(f"in a {plan.style} style")

        # Add mood
        if plan.mood:
            parts.append(f"with a {plan.mood} mood")

        # Add energy level
        if plan.energy:
            energy_map = {
                "low": "calm and gentle",
                "medium": "moderate energy",
                "high": "energetic and dynamic",
            }
            parts.append(energy_map.get(plan.energy, plan.energy))

        # Add tempo
        if plan.tempo:
            tempo_map = {
                "slow": "at a slow tempo",
                "moderate": "at a moderate tempo",
                "fast": "at a fast tempo",
            }
            parts.append(tempo_map.get(plan.tempo, f"at a {plan.tempo} tempo"))

        # Add instruments
        if plan.instruments:
            instr_str = ", ".join(plan.instruments)
            parts.append(f"featuring {instr_str}")

        # Add constraints
        if plan.constraints:
            for constraint in plan.constraints:
                parts.append(constraint)

        # Add continuation context
        if plan.continuation_context:
            parts.insert(0, f"Continuing from: {plan.continuation_context}.")

        return " ".join(parts) if parts else "A pleasant musical piece"

    def build_refinement(self, plan: GenerationPlan, feedback: str) -> str:
        """Build a prompt that incorporates evaluation feedback.

        Args:
            plan: The original generation plan.
            feedback: Specific feedback about what to improve.

        Returns:
            Refined prompt string.
        """
        base = self.build(plan)
        return f"{base}. {feedback}"

    def get_system_prompt(self) -> str:
        """Get the system prompt for MIDI-LLM."""
        return self.system_prompt
