"""GPT-4o Conductor: Musical director with natural language understanding."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from app.core.feature_extractor import MusicFeatures


@dataclass
class Track:
    """Represents a single musical track in the composition."""

    id: str
    instrument: str
    role: str  # melody, harmony, rhythm, bass
    midi_path: str
    audio_path: str
    features: MusicFeatures
    metadata: dict = field(default_factory=dict)  # key, tempo, bars, etc.

    def to_summary(self) -> str:
        """Convert track to human-readable summary for GPT."""
        return (
            f"Track {self.id} ({self.instrument}, {self.role}): "
            f"{self.features.note_count} notes, "
            f"{self.features.duration_seconds:.1f}s, "
            f"density {self.features.note_density:.1f} notes/sec, "
            f"pitch range {self.features.pitch_range[0]}-{self.features.pitch_range[1]} "
            f"(mean {self.features.pitch_mean:.0f})"
        )


@dataclass
class CompositionState:
    """Current state of the composition."""

    composition_id: str
    tracks: list[Track] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_context(self) -> str:
        """Convert composition state to text for GPT."""
        if not self.tracks:
            return "There are no tracks yet."

        context = f"The current composition has {len(self.tracks)} track(s):\n"
        for track in self.tracks:
            context += f"- {track.to_summary()}\n"

        if self.metadata:
            context += f"\nGlobal metadata: {json.dumps(self.metadata, ensure_ascii=False)}"

        return context


@dataclass
class Action:
    """An action to be executed by the system."""

    type: str  # create_track, regenerate_track, modify_track, delete_track
    parameters: dict = field(default_factory=dict)


@dataclass
class ConductorResponse:
    """Response from GPT-4o Conductor."""

    message: str  # Natural language response to user
    actions: list[Action] = field(default_factory=list)
    reasoning: str = ""  # Internal reasoning (for debugging)


# System prompt for GPT-4o Conductor
CONDUCTOR_SYSTEM_PROMPT = """You are a professional music producer and arranger. Your role is to understand the user's musical intent, analyze the generated music, and plan how to improve or expand the composition.

You must communicate in English only.

**What you can do:**
- Understand musical terms, styles, moods, and arrangement
- Analyze MIDI feature data (note density, pitch distribution, instruments, etc.)
- Plan creative steps (create new tracks, modify existing ones)

**What you do NOT do:**
- Generate MIDI notes directly (the MIDI-LLM does that)
- Perform audio signal processing
- Remember full conversation history (only current state is provided)

**MIDI feature schema:**
- `note_density`: notes/sec (higher means denser)
- `pitch_range`: MIDI pitch range (60 = middle C)
- `instruments_used`: GM program numbers
- `duration_seconds`: duration in seconds
- `onset_density_curve`: notes per time window

**Required output format (JSON only):**
```json
{
  "message": "Natural language reply to the user (English)",
  "actions": [
    {
      "type": "create_track | regenerate_track | modify_track | delete_track",
      "parameters": {
        "track_id": "required for modify/delete",
        "instrument": "Piano | Drums | Bass | Strings | Guitar ...",
        "role": "melody | harmony | rhythm | bass",
        "instruction": "Detailed English prompt for the MIDI-LLM (style, tempo, feel, key, texture)"
      }
    }
  ],
  "reasoning": "Optional internal reasoning for debugging"
}
```

**Guiding principles:**
1. Prioritize user experience with clear, friendly English.
2. Make musically sound decisions grounded in theory.
3. Use gradual iteration: add or change a small number of tracks per turn.
4. Be feature-driven: reference MIDI features, not vague guesses.
5. Provide precise instructions to the MIDI-LLM.

**Example:**
User: "I want a calm piano piece"
Output:
```json
{
  "message": "Great—I'll create a calm piano melody with a gentle pace and a warm middle register.",
  "actions": [{
    "type": "create_track",
    "parameters": {
      "instrument": "Piano",
      "role": "melody",
      "instruction": "Calm piano melody in C major, slow tempo around 70 BPM, sparse eighth notes, middle register (C4-C6), gentle dynamics, peaceful mood"
    }
  }]
}
```

Now, based on the user's message and the current composition state, output your reply and action plan as JSON."""


class GPTConductor:
    """GPT-4o powered musical conductor."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT-4o client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or parameters")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    async def plan_action(
        self,
        user_message: str,
        composition_state: Optional[CompositionState] = None,
    ) -> ConductorResponse:
        """Plan musical actions based on user message and current state.

        Args:
            user_message: User's natural language request
            composition_state: Current composition (tracks + metadata)

        Returns:
            ConductorResponse with message + actions
        """
        # Build user message with context
        context = composition_state.to_context() if composition_state else "There are no tracks yet."

        user_prompt = f"""User message: {user_message}

Current composition state:
{context}

Analyze the user's intent and the current state, then output your reply and action plan as JSON."""

        # Call GPT-4o
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONDUCTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from GPT-4o")

            # Parse JSON response
            parsed = json.loads(content)
            message = parsed.get("message", "")
            actions_data = parsed.get("actions", [])
            reasoning = parsed.get("reasoning", "")

            # Convert to Action objects
            actions = [
                Action(type=a["type"], parameters=a.get("parameters", {}))
                for a in actions_data
            ]

            return ConductorResponse(
                message=message,
                actions=actions,
                reasoning=reasoning,
            )

        except json.JSONDecodeError as e:
            # Fallback if GPT doesn't return valid JSON
            return ConductorResponse(
                message=f"Sorry, I had trouble understanding your request: {e}",
                actions=[],
            )

        except Exception as e:
            return ConductorResponse(
                message=f"系统错误：{e}",
                actions=[],
            )
