"""Symbolic feature extraction from processed MIDI generations."""

import io
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Union

import numpy as np
import pretty_midi

from app.core.token_processor import ProcessedGeneration


@dataclass
class MusicFeatures:
    """Extracted musical features for conductor evaluation."""

    # Density
    note_density: float = 0.0  # Notes per second
    note_count: int = 0
    duration_seconds: float = 0.0

    # Pitch
    pitch_range: tuple[int, int] = (0, 0)  # (min, max) MIDI pitch
    pitch_mean: float = 0.0
    pitch_std: float = 0.0

    # Rhythm
    onset_density_curve: list[float] = field(default_factory=list)

    # Instruments
    instruments_used: list[int] = field(default_factory=list)
    instrument_note_counts: dict[int, int] = field(default_factory=dict)

    # Quality indicators
    has_excessive_notes: bool = False
    silence_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d["pitch_range"] = list(d["pitch_range"])
        # Convert dict keys to str for JSON
        d["instrument_note_counts"] = {
            str(k): v for k, v in d["instrument_note_counts"].items()
        }
        return d


class FeatureExtractor:
    """Extracts musical features from ProcessedGeneration objects."""

    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = window_seconds

    def extract(self, generation: ProcessedGeneration) -> MusicFeatures:
        """Extract all features from a processed generation.

        Args:
            generation: A ProcessedGeneration with valid midi_bytes.

        Returns:
            MusicFeatures with all fields populated.
        """
        if not generation.is_valid or not generation.midi_bytes:
            return MusicFeatures()

        pm = pretty_midi.PrettyMIDI(io.BytesIO(generation.midi_bytes))
        duration = pm.get_end_time()

        if duration <= 0:
            return MusicFeatures()

        # Collect all notes across instruments
        all_notes = []
        instrument_counts: dict[int, int] = {}
        for inst in pm.instruments:
            prog = int(inst.program) if not inst.is_drum else -1
            count = len(inst.notes)
            instrument_counts[prog] = instrument_counts.get(prog, 0) + count
            all_notes.extend(inst.notes)

        if not all_notes:
            return MusicFeatures(duration_seconds=duration)

        pitches = [n.pitch for n in all_notes]
        onsets = [n.start for n in all_notes]

        # Pitch statistics
        pitch_arr = np.array(pitches)
        pitch_range = (int(pitch_arr.min()), int(pitch_arr.max()))
        pitch_mean = float(pitch_arr.mean())
        pitch_std = float(pitch_arr.std())

        # Note density
        note_count = len(all_notes)
        note_density = note_count / duration

        # Onset density curve (notes per window)
        num_windows = max(1, int(np.ceil(duration / self.window_seconds)))
        density_curve = [0.0] * num_windows
        for onset in onsets:
            idx = min(int(onset / self.window_seconds), num_windows - 1)
            density_curve[idx] += 1.0

        # Silence ratio (fraction of windows with no onsets)
        silent_windows = sum(1 for d in density_curve if d == 0)
        silence_ratio = silent_windows / num_windows

        # Excessive notes check
        time_counts = Counter(int(o * 50) for o in onsets)  # 50 ticks/sec resolution
        has_excessive = max(time_counts.values()) > 64 if time_counts else False

        instruments_used = sorted(
            int(inst.program) for inst in pm.instruments if not inst.is_drum
        )

        return MusicFeatures(
            note_density=round(note_density, 2),
            note_count=note_count,
            duration_seconds=round(duration, 2),
            pitch_range=pitch_range,
            pitch_mean=round(pitch_mean, 2),
            pitch_std=round(pitch_std, 2),
            onset_density_curve=[round(d, 1) for d in density_curve],
            instruments_used=instruments_used,
            instrument_note_counts=instrument_counts,
            has_excessive_notes=has_excessive,
            silence_ratio=round(silence_ratio, 3),
        )


def extract_features(
    source: Union[ProcessedGeneration, pretty_midi.PrettyMIDI], window_seconds: float = 2.0
) -> MusicFeatures:
    """Convenience function to extract features from a generation or PrettyMIDI object.

    Args:
        source: Either a ProcessedGeneration or a PrettyMIDI object
        window_seconds: Window size for onset density curve

    Returns:
        MusicFeatures with all fields populated
    """
    extractor = FeatureExtractor(window_seconds=window_seconds)

    # If it's a PrettyMIDI object, create a temporary ProcessedGeneration
    if isinstance(source, pretty_midi.PrettyMIDI):
        # Convert PrettyMIDI to bytes
        midi_buffer = io.BytesIO()
        source.write(midi_buffer)
        midi_bytes = midi_buffer.getvalue()

        # Create temporary ProcessedGeneration
        temp_gen = ProcessedGeneration(
            midi_token_ids=[],
            midi_bytes=midi_bytes,
            pretty_midi=source,
            is_valid=True,
            note_count=sum(len(inst.notes) for inst in source.instruments),
        )
        return extractor.extract(temp_gen)

    # Otherwise it's already a ProcessedGeneration
    return extractor.extract(source)
