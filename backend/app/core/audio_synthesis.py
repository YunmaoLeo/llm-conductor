"""Audio Synthesis: MIDI to audio conversion with per-track support."""

import subprocess
from pathlib import Path
from typing import Optional

import pretty_midi

from app.config import settings


class AudioSynthesizer:
    """Handles MIDI to audio conversion using FluidSynth."""

    def __init__(self, soundfont_path: Optional[str] = None):
        """Initialize audio synthesizer.

        Args:
            soundfont_path: Path to SoundFont file (default from settings)
        """
        self.soundfont_path = Path(soundfont_path or settings.soundfont_path)

        if not self.soundfont_path.exists():
            raise FileNotFoundError(
                f"SoundFont not found: {self.soundfont_path}. "
                "Please download FluidR3_GM.sf2 and place it in soundfonts/"
            )

    def midi_to_audio(
        self,
        midi_path: Path,
        audio_path: Path,
        format: str = "mp3",
    ) -> Path:
        """Convert MIDI file to audio.

        Args:
            midi_path: Path to input MIDI file
            audio_path: Path to output audio file
            format: Audio format (mp3, wav, flac)

        Returns:
            Path to generated audio file

        Raises:
            subprocess.CalledProcessError: If conversion fails
        """
        # Ensure input exists
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        # Create output directory
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert MIDI → WAV using FluidSynth
        wav_path = audio_path.with_suffix(".wav")

        fluidsynth_cmd = [
            "fluidsynth",
            "-ni",  # No interactive shell
            str(self.soundfont_path),
            str(midi_path),
            "-F", str(wav_path),
            "-r", "44100",  # Sample rate
        ]

        subprocess.run(fluidsynth_cmd, check=True, capture_output=True)

        # If MP3 requested, convert WAV → MP3 using ffmpeg
        if format == "mp3":
            mp3_path = audio_path.with_suffix(".mp3")

            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",  # High quality
                "-y",  # Overwrite
                str(mp3_path),
            ]

            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

            # Remove intermediate WAV
            wav_path.unlink()

            return mp3_path

        return wav_path

    def combine_tracks(
        self,
        track_midi_paths: list[Path],
        output_midi_path: Path,
    ) -> Path:
        """Combine multiple MIDI tracks into one file.

        Args:
            track_midi_paths: List of MIDI file paths
            output_midi_path: Path to output combined MIDI

        Returns:
            Path to combined MIDI file
        """
        if not track_midi_paths:
            raise ValueError("No MIDI tracks to combine")

        # Load all MIDI files
        midi_objects = []
        for path in track_midi_paths:
            if path.exists():
                midi_objects.append(pretty_midi.PrettyMIDI(str(path)))

        if not midi_objects:
            raise ValueError("No valid MIDI files found")

        # Create new MIDI with all tracks
        combined = pretty_midi.PrettyMIDI()

        # Copy all instruments from all MIDI files
        for midi_obj in midi_objects:
            for instrument in midi_obj.instruments:
                combined.instruments.append(instrument)

        # Write combined MIDI
        output_midi_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write(str(output_midi_path))

        return output_midi_path

    def synthesize_track(
        self,
        track_id: str,
        midi_bytes: bytes,
        output_dir: Path,
        format: str = "mp3",
    ) -> tuple[Path, Path]:
        """Synthesize a single track from MIDI bytes.

        Preserves the previous version by moving existing files to *_prev
        before generating the new version. This allows version comparison.

        Args:
            track_id: Track identifier
            midi_bytes: MIDI file bytes
            output_dir: Output directory
            format: Audio format (mp3, wav, flac)

        Returns:
            Tuple of (midi_path, audio_path)
        """
        import shutil

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define paths
        midi_path = output_dir / f"{track_id}.mid"
        audio_ext = ".mp3" if format == "mp3" else f".{format}"
        audio_path = output_dir / f"{track_id}{audio_ext}"

        # NEW: Backup existing files to _prev before overwriting
        prev_midi_path = output_dir / f"{track_id}_prev.mid"
        prev_audio_path = output_dir / f"{track_id}_prev{audio_ext}"

        if midi_path.exists():
            # Move current to previous (overwrites old _prev if exists)
            shutil.move(str(midi_path), str(prev_midi_path))
            if audio_path.exists():
                shutil.move(str(audio_path), str(prev_audio_path))

        # Write new MIDI file
        midi_path.write_bytes(midi_bytes)

        # Convert to audio
        self.midi_to_audio(midi_path, audio_path, format=format)

        return midi_path, audio_path

    def synthesize_mix(
        self,
        composition_id: str,
        track_midi_paths: list[Path],
        output_dir: Path,
        format: str = "mp3",
    ) -> tuple[Path, Path]:
        """Synthesize a mixed version of all tracks.

        Args:
            composition_id: Composition identifier
            track_midi_paths: List of track MIDI paths
            output_dir: Output directory
            format: Audio format

        Returns:
            Tuple of (combined_midi_path, mixed_audio_path)
        """
        # Combine MIDI tracks
        combined_midi = output_dir / f"{composition_id}_mix.mid"
        self.combine_tracks(track_midi_paths, combined_midi)

        # Convert to audio
        audio_ext = ".mp3" if format == "mp3" else f".{format}"
        mixed_audio = output_dir / f"{composition_id}_mix{audio_ext}"

        self.midi_to_audio(combined_midi, mixed_audio, format=format)

        return combined_midi, mixed_audio
