"""Symbolic feature extraction from processed MIDI generations."""

import io
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

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

    # Musical structure (from music_analysis)
    estimated_key: Optional[tuple[str, str]] = None  # ("C", "major")
    key_confidence: float = 0.0
    estimated_tempo: float = 0.0  # BPM
    chord_progression: list[str] = field(default_factory=list)

    # Key conformance
    in_key_ratio: float = 0.0  # Fraction of note durations on scale degrees of detected key

    # Dynamics
    velocity_mean: float = 0.0
    velocity_std: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d["pitch_range"] = list(d["pitch_range"])
        # Convert dict keys to str for JSON
        d["instrument_note_counts"] = {
            str(k): v for k, v in d["instrument_note_counts"].items()
        }
        # Convert estimated_key tuple to list for JSON
        if d.get("estimated_key") is not None:
            d["estimated_key"] = list(d["estimated_key"])
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

        # Music analysis: key, tempo, chords, velocity
        from app.core.music_analysis import detect_key, detect_tempo, estimate_chords, get_velocity_stats

        key_estimate = detect_key(pm)
        estimated_key = None
        key_confidence = 0.0
        if key_estimate:
            estimated_key = (key_estimate.key, key_estimate.mode)
            key_confidence = key_estimate.confidence

        estimated_tempo = detect_tempo(pm)

        # Compute in-key ratio (fraction of note duration on scale degrees)
        in_key_ratio = 0.0
        if estimated_key and key_confidence > 0.3:
            _MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
            _MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
            key_name, mode = estimated_key
            _pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            root_pc = _pc_names.index(key_name) if key_name in _pc_names else 0
            scale_pcs = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
            # Shift scale degrees to the actual key root
            scale_set = {(root_pc + d) % 12 for d in scale_pcs}

            in_key_dur = 0.0
            total_dur = 0.0
            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                for note in inst.notes:
                    dur = max(note.end - note.start, 0.01)
                    total_dur += dur
                    if note.pitch % 12 in scale_set:
                        in_key_dur += dur

            if total_dur > 0:
                in_key_ratio = round(in_key_dur / total_dur, 3)

        chord_estimates = estimate_chords(pm, window_seconds=self.window_seconds)
        chord_progression = [c.name for c in chord_estimates]

        velocity_mean, velocity_std = get_velocity_stats(pm)

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
            estimated_key=estimated_key,
            key_confidence=key_confidence,
            estimated_tempo=estimated_tempo,
            chord_progression=chord_progression,
            in_key_ratio=in_key_ratio,
            velocity_mean=velocity_mean,
            velocity_std=velocity_std,
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
        instruments = sorted({int(inst.program) for inst in source.instruments if not inst.is_drum})
        num_notes = sum(len(inst.notes) for inst in source.instruments)
        temp_gen = ProcessedGeneration(
            id="temp",
            token_ids=[],
            note_events=[],
            midi_bytes=midi_bytes,
            num_notes=num_notes,
            duration_seconds=source.get_end_time(),
            instruments_used=instruments,
            is_valid=True,
            validation_message="Valid",
        )
        return extractor.extract(temp_gen)

    # Otherwise it's already a ProcessedGeneration
    return extractor.extract(source)
