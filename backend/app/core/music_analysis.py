"""Music analysis: key detection, tempo estimation, chord estimation.

Implements lightweight Krumhansl-Schmuckler key-finding algorithm
using numpy (no music21 dependency required).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles
# Correlation weights for each pitch class (C, C#, D, ..., B) in major and minor keys
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
])
_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
])

_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Simple chord templates (pitch class sets relative to root)
_CHORD_TEMPLATES = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "7": {0, 4, 7, 10},
    "m7": {0, 3, 7, 10},
}


@dataclass
class KeyEstimate:
    """Result of key detection."""

    key: str  # e.g., "C", "F#"
    mode: str  # "major" or "minor"
    confidence: float  # Correlation coefficient (0-1)


@dataclass
class ChordEstimate:
    """A chord detected in a time window."""

    name: str  # e.g., "Cmaj", "Am", "G7"
    start_time: float
    end_time: float


def detect_key(pm: pretty_midi.PrettyMIDI) -> Optional[KeyEstimate]:
    """Detect key signature using Krumhansl-Schmuckler algorithm.

    Computes a pitch class histogram weighted by note duration,
    then correlates against all 24 major/minor key profiles.

    Args:
        pm: PrettyMIDI object to analyze.

    Returns:
        KeyEstimate with key name, mode, and confidence, or None if too few notes.
    """
    # Collect all notes with duration weighting
    pitch_class_weights = np.zeros(12)
    total_notes = 0

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pc = note.pitch % 12
            duration = max(note.end - note.start, 0.01)
            pitch_class_weights[pc] += duration
            total_notes += 1

    if total_notes < 5 or pitch_class_weights.sum() < 0.1:
        return None

    # Normalize
    pitch_class_weights = pitch_class_weights / pitch_class_weights.sum()

    # Correlate against all 24 keys (12 major + 12 minor)
    best_key = None
    best_mode = None
    best_corr = -1.0

    for shift in range(12):
        # Rotate pitch class distribution to test each key
        rotated = np.roll(pitch_class_weights, -shift)

        # Major key correlation
        major_corr = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = _PITCH_CLASS_NAMES[shift]
            best_mode = "major"

        # Minor key correlation
        minor_corr = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = _PITCH_CLASS_NAMES[shift]
            best_mode = "minor"

    if best_key is None:
        return None

    return KeyEstimate(
        key=best_key,
        mode=best_mode,
        confidence=round(max(0.0, best_corr), 3),
    )


def detect_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Estimate tempo from MIDI using pretty_midi's built-in method.

    Args:
        pm: PrettyMIDI object.

    Returns:
        Estimated BPM (0.0 if detection fails).
    """
    try:
        tempo = pm.estimate_tempo()
        if tempo and tempo > 0:
            return round(float(tempo), 1)
    except Exception as e:
        logger.debug(f"Tempo detection failed: {e}")

    # Fallback: check MIDI tempo changes
    tempo_changes = pm.get_tempo_changes()
    if tempo_changes and len(tempo_changes[1]) > 0:
        return round(float(np.median(tempo_changes[1])), 1)

    return 0.0


def estimate_chords(
    pm: pretty_midi.PrettyMIDI, window_seconds: float = 2.0
) -> list[ChordEstimate]:
    """Estimate chord progression using pitch class analysis per time window.

    Divides the track into windows, finds dominant pitch classes in each,
    and maps to the best-matching chord template.

    Args:
        pm: PrettyMIDI object.
        window_seconds: Analysis window size in seconds.

    Returns:
        List of ChordEstimate objects.
    """
    duration = pm.get_end_time()
    if duration <= 0:
        return []

    num_windows = max(1, int(np.ceil(duration / window_seconds)))
    chords = []

    for i in range(num_windows):
        start = i * window_seconds
        end = min((i + 1) * window_seconds, duration)

        # Collect pitch classes in this window (weighted by duration overlap)
        pc_weights = np.zeros(12)
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                # Check overlap with window
                overlap_start = max(note.start, start)
                overlap_end = min(note.end, end)
                if overlap_start < overlap_end:
                    weight = overlap_end - overlap_start
                    pc_weights[note.pitch % 12] += weight

        if pc_weights.sum() < 0.01:
            chords.append(ChordEstimate(name="N.C.", start_time=start, end_time=end))
            continue

        # Find best matching chord
        best_chord = _match_chord(pc_weights)
        chords.append(ChordEstimate(name=best_chord, start_time=start, end_time=end))

    return chords


def _match_chord(pc_weights: np.ndarray) -> str:
    """Match pitch class weights to the best chord template.

    Args:
        pc_weights: Array of 12 pitch class weights.

    Returns:
        Chord name string (e.g., "Cmaj", "Am7").
    """
    # Get top pitch classes
    top_indices = set(np.argsort(pc_weights)[-4:])  # Top 4 pitch classes
    top_indices = {i for i in top_indices if pc_weights[i] > 0}

    if not top_indices:
        return "N.C."

    best_name = "N.C."
    best_score = -1.0

    for root in range(12):
        for template_name, template_pcs in _CHORD_TEMPLATES.items():
            # Shift template to this root
            shifted = {(root + pc) % 12 for pc in template_pcs}

            # Score: sum of weights for matching pitch classes
            match_weight = sum(pc_weights[pc] for pc in shifted if pc in top_indices)
            total_weight = sum(pc_weights[pc] for pc in top_indices)

            if total_weight > 0:
                score = match_weight / total_weight
            else:
                score = 0.0

            # Bonus for having the root as strongest note
            if pc_weights[root] == pc_weights.max():
                score += 0.1

            if score > best_score:
                best_score = score
                root_name = _PITCH_CLASS_NAMES[root]
                suffix = template_name if template_name != "maj" else "maj"
                if template_name == "min":
                    suffix = "m"
                best_name = f"{root_name}{suffix}"

    return best_name


def get_velocity_stats(pm: pretty_midi.PrettyMIDI) -> tuple[float, float]:
    """Get velocity (dynamics) statistics.

    Args:
        pm: PrettyMIDI object.

    Returns:
        (mean_velocity, std_velocity) tuple. Values 0-127.
    """
    velocities = []
    for inst in pm.instruments:
        for note in inst.notes:
            velocities.append(note.velocity)

    if not velocities:
        return 0.0, 0.0

    arr = np.array(velocities, dtype=float)
    return round(float(arr.mean()), 1), round(float(arr.std()), 1)
