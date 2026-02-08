"""Token processing pipeline: raw MIDI tokens → structured MIDI → audio."""

import io
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from anticipation.convert import events_to_midi

from app.config import settings


@dataclass
class NoteEvent:
    """A single note parsed from the token triple (arrival_time, duration, pitch)."""

    arrival_time: int
    duration: int
    instrument_pitch: int


@dataclass
class ProcessedGeneration:
    """Result of processing raw MIDI tokens into structured data."""

    id: str
    token_ids: list[int]
    note_events: list[NoteEvent]
    midi_bytes: bytes
    num_notes: int
    duration_seconds: float
    instruments_used: list[int]
    is_valid: bool
    validation_message: str
    midi_path: Optional[str] = None
    audio_path: Optional[str] = None


def validate_tokens(token_ids: list[int], max_notes_per_time: int = 64) -> tuple[bool, str]:
    """Validate a MIDI token sequence.

    Checks:
    - Non-empty
    - Length divisible by 3 (each note = 3 tokens)
    - No excessive simultaneous notes at any time point

    Args:
        token_ids: List of AMT token IDs.
        max_notes_per_time: Maximum allowed notes at a single time.

    Returns:
        Tuple of (is_valid, message).
    """
    if not token_ids:
        return False, "Empty token sequence"

    if len(token_ids) < 3:
        return False, f"Token sequence too short ({len(token_ids)} tokens, need at least 3)"

    # Truncate to multiple of 3
    remainder = len(token_ids) % 3
    if remainder != 0:
        token_ids = token_ids[: len(token_ids) - remainder]

    # Check for excessive simultaneous notes
    times = token_ids[::3]
    time_counts: dict[int, int] = {}
    for t in times:
        time_counts[t] = time_counts.get(t, 0) + 1

    max_count = max(time_counts.values()) if time_counts else 0
    if max_count > max_notes_per_time:
        return False, f"Excessive simultaneous notes: {max_count} at one time point"

    return True, "Valid"


def parse_note_events(token_ids: list[int]) -> list[NoteEvent]:
    """Parse token triples into NoteEvent objects.

    AMT tokenization: every 3 consecutive tokens represent one note:
    [arrival_time, duration, instrument_pitch]

    Args:
        token_ids: List of AMT token IDs (length should be divisible by 3).

    Returns:
        List of NoteEvent objects.
    """
    # Truncate to multiple of 3
    usable = len(token_ids) - (len(token_ids) % 3)
    events = []
    for i in range(0, usable, 3):
        events.append(
            NoteEvent(
                arrival_time=token_ids[i],
                duration=token_ids[i + 1],
                instrument_pitch=token_ids[i + 2],
            )
        )
    return events


class TokenProcessor:
    """Processes raw MIDI token IDs into structured MIDI data."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, token_ids: list[int], validate: bool = True) -> ProcessedGeneration:
        """Full processing pipeline: validate → parse → convert to MIDI.

        Args:
            token_ids: Raw AMT token IDs from Ollama response.
            validate: Whether to validate tokens before processing.

        Returns:
            ProcessedGeneration with MIDI bytes and metadata.
        """
        gen_id = str(uuid.uuid4())[:8]

        # Truncate to multiple of 3
        remainder = len(token_ids) % 3
        if remainder != 0:
            token_ids = token_ids[: len(token_ids) - remainder]

        # Validate
        if validate:
            is_valid, message = validate_tokens(token_ids)
            if not is_valid:
                return ProcessedGeneration(
                    id=gen_id,
                    token_ids=token_ids,
                    note_events=[],
                    midi_bytes=b"",
                    num_notes=0,
                    duration_seconds=0.0,
                    instruments_used=[],
                    is_valid=False,
                    validation_message=message,
                )

        # Parse note events
        note_events = parse_note_events(token_ids)

        # Convert to MIDI using anticipation library
        try:
            midi_obj = events_to_midi(token_ids)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            return ProcessedGeneration(
                id=gen_id,
                token_ids=token_ids,
                note_events=note_events,
                midi_bytes=b"",
                num_notes=len(note_events),
                duration_seconds=0.0,
                instruments_used=[],
                is_valid=False,
                validation_message=f"MIDI conversion failed: {type(e).__name__}: {e}\n{tb}",
            )

        # Serialize MIDI to bytes (mido.MidiFile returned by anticipation)
        midi_bytes = self._midi_to_bytes(midi_obj)

        # Extract metadata via pretty_midi for richer analysis
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        duration = pm.get_end_time()
        instruments = sorted({int(inst.program) for inst in pm.instruments if not inst.is_drum})
        num_notes = sum(len(inst.notes) for inst in pm.instruments)

        # Save MIDI file
        midi_path = self.output_dir / f"{gen_id}.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)

        return ProcessedGeneration(
            id=gen_id,
            token_ids=token_ids,
            note_events=note_events,
            midi_bytes=midi_bytes,
            num_notes=num_notes,
            duration_seconds=duration,
            instruments_used=instruments,
            is_valid=True,
            validation_message="Valid",
            midi_path=str(midi_path),
        )

    def _midi_to_bytes(self, midi_obj) -> bytes:
        """Serialize a mido.MidiFile object to bytes in memory."""
        buf = io.BytesIO()
        # mido.MidiFile.save() accepts a file object
        midi_obj.save(file=buf)
        return buf.getvalue()
