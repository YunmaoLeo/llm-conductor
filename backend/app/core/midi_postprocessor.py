"""MIDI post-processing: quantization, velocity normalization, overlap resolution."""

import logging
import random

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)


def quantize_timing(pm: pretty_midi.PrettyMIDI, grid_resolution: float = 1 / 16, tempo: float = 120.0) -> None:
    """Snap note start/end times to the nearest grid division.

    Args:
        pm: PrettyMIDI object (modified in place).
        grid_resolution: Grid subdivision (1/16 = sixteenth note).
        tempo: Tempo in BPM for grid calculation.
    """
    if tempo <= 0:
        tempo = 120.0

    # Grid size in seconds: one beat = 60/tempo seconds, times resolution (e.g. 1/4 of a beat for 16th)
    beat_duration = 60.0 / tempo
    grid_size = beat_duration * (grid_resolution * 4)  # grid_resolution=1/16 -> 1/4 beat

    if grid_size <= 0:
        return

    quantized_count = 0
    for inst in pm.instruments:
        for note in inst.notes:
            old_start = note.start
            old_end = note.end

            note.start = round(note.start / grid_size) * grid_size
            note.end = round(note.end / grid_size) * grid_size

            # Ensure minimum note duration (one grid unit)
            if note.end <= note.start:
                note.end = note.start + grid_size

            if note.start != old_start or note.end != old_end:
                quantized_count += 1

    if quantized_count > 0:
        logger.debug(f"Quantized {quantized_count} note timings to grid={grid_size:.4f}s")


def normalize_velocity(
    pm: pretty_midi.PrettyMIDI,
    target_mean: int = 80,
    target_range: tuple[int, int] = (45, 115),
) -> None:
    """Scale velocities to a musically reasonable range.

    Args:
        pm: PrettyMIDI object (modified in place).
        target_mean: Target mean velocity.
        target_range: (min, max) velocity range.
    """
    all_notes = [note for inst in pm.instruments for note in inst.notes]
    if not all_notes:
        return

    velocities = np.array([n.velocity for n in all_notes], dtype=float)
    current_mean = velocities.mean()
    current_std = velocities.std()

    if current_std < 1.0:
        # All same velocity - add slight random variation
        for note in all_notes:
            note.velocity = int(np.clip(target_mean + random.randint(-5, 5), target_range[0], target_range[1]))
        logger.debug(f"Added velocity variation (was uniform at {int(current_mean)})")
        return

    # Linear scaling to target range
    v_min, v_max = velocities.min(), velocities.max()
    target_min, target_max = target_range

    for note in all_notes:
        # Normalize to 0-1, then scale to target range
        normalized = (note.velocity - v_min) / (v_max - v_min) if v_max > v_min else 0.5
        new_vel = target_min + normalized * (target_max - target_min)
        note.velocity = int(np.clip(new_vel, target_min, target_max))

    logger.debug(
        f"Normalized velocity: mean {current_mean:.0f}->{target_mean}, "
        f"range [{v_min:.0f},{v_max:.0f}]->[{target_min},{target_max}]"
    )


def resolve_overlaps(pm: pretty_midi.PrettyMIDI, max_polyphony: int = 4) -> None:
    """Resolve overlapping notes and cap polyphony per instrument.

    For each instrument:
    - Same-pitch overlaps: shorten earlier note to end before later one starts.
    - Cap simultaneous notes to max_polyphony.

    Args:
        pm: PrettyMIDI object (modified in place).
        max_polyphony: Maximum simultaneous notes per instrument.
    """
    for inst in pm.instruments:
        if not inst.notes:
            continue

        # Sort by start time, then pitch
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

        # 1. Fix same-pitch overlaps
        by_pitch: dict[int, list] = {}
        for note in inst.notes:
            by_pitch.setdefault(note.pitch, []).append(note)

        overlap_fixes = 0
        for pitch, notes in by_pitch.items():
            notes.sort(key=lambda n: n.start)
            for i in range(len(notes) - 1):
                if notes[i].end > notes[i + 1].start:
                    # Shorten earlier note to end just before next starts
                    notes[i].end = notes[i + 1].start - 0.001
                    if notes[i].end <= notes[i].start:
                        notes[i].end = notes[i].start + 0.01
                    overlap_fixes += 1

        # 2. Cap polyphony: remove excess notes at any time point
        inst.notes.sort(key=lambda n: n.start)
        notes_to_keep = []
        for note in inst.notes:
            # Count active notes at this note's start time
            active = sum(1 for n in notes_to_keep if n.start <= note.start < n.end)
            if active < max_polyphony:
                notes_to_keep.append(note)

        dropped = len(inst.notes) - len(notes_to_keep)
        inst.notes = notes_to_keep

        if overlap_fixes > 0 or dropped > 0:
            logger.debug(
                f"Instrument {inst.program}: fixed {overlap_fixes} overlaps, "
                f"dropped {dropped} notes (polyphony cap={max_polyphony})"
            )


def postprocess(pm: pretty_midi.PrettyMIDI, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Full post-processing pipeline: quantize, normalize velocity, resolve overlaps.

    Args:
        pm: PrettyMIDI object (modified in place and returned).
        tempo: Estimated tempo in BPM.

    Returns:
        The post-processed PrettyMIDI object.
    """
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    if total_notes == 0:
        return pm

    logger.info(f"Post-processing MIDI: {total_notes} notes, tempo={tempo:.0f} BPM")

    quantize_timing(pm, grid_resolution=1 / 16, tempo=tempo)
    normalize_velocity(pm)
    resolve_overlaps(pm)

    final_notes = sum(len(inst.notes) for inst in pm.instruments)
    logger.info(f"Post-processing complete: {total_notes} -> {final_notes} notes")

    return pm
