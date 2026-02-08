"""Waveform extraction for audio visualization."""

import io
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


class WaveformExtractor:
    """Extract waveform data from audio files for visualization."""

    def __init__(self, target_samples: int = 200):
        """Initialize waveform extractor.

        Args:
            target_samples: Number of waveform samples to generate (for visualization)
        """
        self.target_samples = target_samples

    def extract_from_audio(
        self, audio_path: Path, max_duration: Optional[float] = None
    ) -> list[float]:
        """Extract waveform amplitudes from audio file.

        Uses ffmpeg to decode audio and compute RMS amplitude for each segment.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            max_duration: Maximum duration to process (seconds)

        Returns:
            List of normalized amplitude values (0.0 to 1.0)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            subprocess.CalledProcessError: If ffmpeg fails
        """
        if not audio_path.exists():
            print(f"[WaveformExtractor] Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[WaveformExtractor] Extracting waveform from: {audio_path}")

        # Use ffmpeg to extract raw PCM data
        # -f s16le: 16-bit signed little-endian PCM
        # -ac 1: mono (mix to single channel)
        # -ar 44100: 44.1kHz sample rate
        # -loglevel error: suppress verbose output
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-",
        ]

        if max_duration:
            cmd.insert(4, "-t")
            cmd.insert(5, str(max_duration))

        # Run ffmpeg and capture raw audio data
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio_data = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[WaveformExtractor] ffmpeg error: {e.stderr.decode()}")
            raise

        # Convert bytes to int16 array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        print(f"[WaveformExtractor] Extracted {len(samples)} samples")

        if len(samples) == 0:
            print("[WaveformExtractor] Warning: No samples extracted, returning zeros")
            return [0.0] * self.target_samples

        # Downsample by computing RMS in chunks
        chunk_size = max(1, len(samples) // self.target_samples)
        waveform = []

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            if len(chunk) > 0:
                # Compute RMS (root mean square) amplitude
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                waveform.append(float(rms))

        # Normalize to 0.0 - 1.0 range
        if waveform:
            max_amp = max(waveform)
            if max_amp > 0:
                waveform = [amp / max_amp for amp in waveform]
            print(f"[WaveformExtractor] Generated {len(waveform)} waveform points (max: {max_amp:.2f})")

        # Pad or truncate to target length
        if len(waveform) < self.target_samples:
            waveform.extend([0.0] * (self.target_samples - len(waveform)))
        elif len(waveform) > self.target_samples:
            waveform = waveform[: self.target_samples]

        print(f"[WaveformExtractor] Final waveform: {len(waveform)} points")
        return waveform

    def extract_from_midi(self, midi_path: Path) -> list[float]:
        """Extract waveform-like visualization from MIDI file.

        Since MIDI doesn't have audio waveform, we compute note density over time.

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of normalized density values (0.0 to 1.0)
        """
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        duration = pm.get_end_time()

        if duration == 0:
            return [0.0] * self.target_samples

        # Divide time into bins
        bin_size = duration / self.target_samples
        density = [0.0] * self.target_samples

        # Count notes in each time bin
        for inst in pm.instruments:
            for note in inst.notes:
                start_bin = int(note.start / bin_size)
                end_bin = int(note.end / bin_size)

                for bin_idx in range(start_bin, min(end_bin + 1, self.target_samples)):
                    density[bin_idx] += note.velocity / 127.0  # Normalize velocity

        # Normalize to 0.0 - 1.0
        if density:
            max_density = max(density)
            if max_density > 0:
                density = [d / max_density for d in density]

        return density
