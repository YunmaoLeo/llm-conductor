"""Audio Synthesis: MIDI to audio conversion with per-track support."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

from app.config import settings

logger = logging.getLogger(__name__)

# Role-specific FluidSynth settings
_ROLE_SYNTH_SETTINGS = {
    "melody": {
        "room_size": 0.5, "reverb_level": 0.35, "reverb_damp": 0.4,
        "chorus": True, "chorus_depth": 3.0, "chorus_level": 0.3,
    },
    "harmony": {
        "room_size": 0.7, "reverb_level": 0.5, "reverb_damp": 0.5,
        "chorus": True, "chorus_depth": 4.0, "chorus_level": 0.4,
    },
    "bass": {
        "room_size": 0.3, "reverb_level": 0.2, "reverb_damp": 0.3,
        "chorus": False, "chorus_depth": 0.0, "chorus_level": 0.0,
    },
    "rhythm": {
        "room_size": 0.2, "reverb_level": 0.15, "reverb_damp": 0.3,
        "chorus": False, "chorus_depth": 0.0, "chorus_level": 0.0,
    },
}

# Default stereo pan positions by role (-1.0 = full left, 0.0 = center, 1.0 = full right)
_ROLE_PAN = {
    "melody": 0.0,
    "harmony": 0.3,
    "bass": 0.0,
    "rhythm": -0.3,
}


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
            "-g", "0.8",    # Gain (prevent clipping)
            "-R", "1",      # Enable reverb
            "-C", "1",      # Enable chorus
            "-o", "synth.reverb.room-size=0.6",
            "-o", "synth.reverb.level=0.4",
            "-o", "synth.reverb.damp=0.4",
            "-o", "synth.chorus.depth=3.0",
            "-o", "synth.chorus.level=0.3",
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

    def midi_to_audio_with_role(
        self,
        midi_path: Path,
        audio_path: Path,
        role: str = "melody",
        format: str = "mp3",
    ) -> Path:
        """Convert MIDI to audio with role-specific reverb/chorus settings.

        Args:
            midi_path: Path to input MIDI file
            audio_path: Path to output audio file
            role: Track role (melody, harmony, bass, rhythm)
            format: Audio format

        Returns:
            Path to generated audio file
        """
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        audio_path.parent.mkdir(parents=True, exist_ok=True)

        synth = _ROLE_SYNTH_SETTINGS.get(role, _ROLE_SYNTH_SETTINGS["melody"])
        wav_path = audio_path.with_suffix(".wav")

        fluidsynth_cmd = [
            "fluidsynth",
            "-ni",
            str(self.soundfont_path),
            str(midi_path),
            "-F", str(wav_path),
            "-r", "44100",
            "-g", "0.8",
            "-R", "1",
            "-C", "1" if synth["chorus"] else "0",
            "-o", f"synth.reverb.room-size={synth['room_size']}",
            "-o", f"synth.reverb.level={synth['reverb_level']}",
            "-o", f"synth.reverb.damp={synth['reverb_damp']}",
            "-o", f"synth.chorus.depth={synth['chorus_depth']}",
            "-o", f"synth.chorus.level={synth['chorus_level']}",
        ]

        subprocess.run(fluidsynth_cmd, check=True, capture_output=True)

        if format == "mp3":
            mp3_path = audio_path.with_suffix(".mp3")
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",
                "-y",
                str(mp3_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
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
        target_duration_seconds: Optional[float] = None,
        format: str = "mp3",
    ) -> tuple[Path, Path]:
        """Synthesize a single track from MIDI bytes.

        Preserves the previous version by moving existing files to *_prev
        before generating the new version. This allows version comparison.

        Args:
            track_id: Track identifier
            midi_bytes: MIDI file bytes
            output_dir: Output directory
            target_duration_seconds: Optional fixed output audio duration in seconds.
                If set, audio will be trimmed/padded to this duration.
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

        # Normalize track audio length to composition target duration.
        if target_duration_seconds and target_duration_seconds > 0:
            from pydub import AudioSegment

            target_ms = max(1, int(target_duration_seconds * 1000))
            audio = AudioSegment.from_file(str(audio_path))

            # Ignore tiny differences from encoder/synth rounding.
            if abs(len(audio) - target_ms) > 20:
                if len(audio) > target_ms:
                    audio = audio[:target_ms]
                else:
                    audio += AudioSegment.silent(
                        duration=target_ms - len(audio),
                        frame_rate=audio.frame_rate,
                    )

                if format == "mp3":
                    audio.export(str(audio_path), format="mp3", bitrate="192k")
                else:
                    audio.export(str(audio_path), format=format)

        return midi_path, audio_path

    def synthesize_mix(
        self,
        composition_id: str,
        track_midi_paths: list[Path],
        output_dir: Path,
        track_volumes: Optional[dict[str, float]] = None,
        track_roles: Optional[dict[str, str]] = None,
        target_duration_seconds: Optional[float] = None,
        format: str = "mp3",
    ) -> tuple[Path, Path]:
        """Synthesize a mixed version of all tracks with mastering.

        Features: role-aware reverb, stereo panning, soft limiting, LUFS normalization.

        Args:
            composition_id: Composition identifier
            track_midi_paths: List of track MIDI paths
            output_dir: Output directory
            track_volumes: Dict mapping track filenames to volume (0.0-1.0)
            track_roles: Dict mapping track filenames to role (melody, harmony, bass, rhythm)
            target_duration_seconds: Optional fixed mix duration in seconds. Each
                rendered track will be trimmed/padded to this duration before overlay.
            format: Audio format

        Returns:
            Tuple of (combined_midi_path, mixed_audio_path)
        """
        import math
        from pydub import AudioSegment

        # Combine MIDI tracks (for reference)
        combined_midi = output_dir / f"{composition_id}_mix.mid"
        self.combine_tracks(track_midi_paths, combined_midi)

        if not track_volumes:
            track_volumes = {}
        if not track_roles:
            track_roles = {}

        logger.info(f"Mixing {len(track_midi_paths)} tracks with mastering pipeline")

        # Render each track with role-aware settings
        track_audio_segments = []
        target_ms = (
            max(1, int(target_duration_seconds * 1000))
            if target_duration_seconds and target_duration_seconds > 0
            else None
        )
        for midi_path in track_midi_paths:
            track_filename = midi_path.name
            role = track_roles.get(track_filename, "melody")
            volume = track_volumes.get(track_filename, 1.0)

            if volume <= 0.0:
                logger.info(f"Track {track_filename} muted (volume=0.0)")
                continue

            # Render with role-specific reverb/chorus
            wav_path = output_dir / f"{midi_path.stem}_temp.wav"
            self.midi_to_audio_with_role(midi_path, wav_path, role=role, format="wav")

            audio = AudioSegment.from_wav(str(wav_path))

            # Apply volume
            if volume < 1.0:
                db_change = 20 * math.log10(volume)
                audio = audio + db_change
                logger.info(f"Track {track_filename} ({role}) volume={volume:.2f} ({db_change:+.1f} dB)")

            # Apply stereo panning
            pan = _ROLE_PAN.get(role, 0.0)
            if abs(pan) > 0.01 and audio.channels == 2:
                audio = audio.pan(pan)
                logger.debug(f"Track {track_filename} panned to {pan:+.1f}")

            # Force same duration for all tracks in the mix.
            if target_ms is not None and abs(len(audio) - target_ms) > 20:
                if len(audio) > target_ms:
                    audio = audio[:target_ms]
                else:
                    audio += AudioSegment.silent(
                        duration=target_ms - len(audio),
                        frame_rate=audio.frame_rate,
                    )

            track_audio_segments.append(audio)
            wav_path.unlink()

        if not track_audio_segments:
            raise ValueError("No tracks to mix (all muted?)")

        # Overlay all tracks
        mixed = track_audio_segments[0]
        for audio in track_audio_segments[1:]:
            mixed = mixed.overlay(audio)

        # Soft limiter (tanh clipping) to prevent digital clipping
        mixed = self._apply_soft_limiter(mixed)

        # LUFS normalization (target -16 LUFS for web streaming)
        mixed = self._apply_lufs_normalization(mixed, target_lufs=-16.0)

        # Export
        audio_ext = ".mp3" if format == "mp3" else f".{format}"
        mixed_audio = output_dir / f"{composition_id}_mix{audio_ext}"

        if format == "mp3":
            mixed.export(str(mixed_audio), format="mp3", bitrate="192k")
        else:
            mixed.export(str(mixed_audio), format=format)

        return combined_midi, mixed_audio

    def _apply_soft_limiter(self, audio_segment) -> "AudioSegment":
        """Apply tanh soft clipping to prevent digital clipping.

        Args:
            audio_segment: pydub AudioSegment

        Returns:
            Soft-limited AudioSegment
        """
        from pydub import AudioSegment

        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)

        # Normalize to -1..1
        max_val = float(2 ** (audio_segment.sample_width * 8 - 1))
        samples = samples / max_val

        # Reshape for stereo
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))

        # Apply tanh soft clipping with mild gain
        gain = 1.5
        samples = np.tanh(samples * gain) / np.tanh(gain)

        # Convert back
        if audio_segment.channels == 2:
            samples = samples.flatten()

        samples = (samples * max_val).astype(np.int16)

        return audio_segment._spawn(samples.tobytes())

    def _apply_lufs_normalization(self, audio_segment, target_lufs: float = -16.0) -> "AudioSegment":
        """Normalize loudness to target LUFS using pyloudnorm.

        Falls back to pydub normalize if pyloudnorm fails.

        Args:
            audio_segment: pydub AudioSegment
            target_lufs: Target integrated loudness in LUFS

        Returns:
            Loudness-normalized AudioSegment
        """
        from pydub import AudioSegment

        try:
            import soundfile as sf
            import pyloudnorm as pyln

            # Convert to numpy
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
            max_val = float(2 ** (audio_segment.sample_width * 8 - 1))
            samples = samples / max_val

            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))

            rate = audio_segment.frame_rate
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(samples)

            if np.isfinite(loudness) and loudness > -70.0:
                samples = pyln.normalize.loudness(samples, loudness, target_lufs)
                # Clip to prevent overflow
                samples = np.clip(samples, -1.0, 1.0)

                if audio_segment.channels == 2:
                    samples = samples.flatten()

                samples = (samples * max_val).astype(np.int16)
                logger.info(f"LUFS normalized: {loudness:.1f} -> {target_lufs:.1f} LUFS")
                return audio_segment._spawn(samples.tobytes())
            else:
                logger.warning(f"LUFS too low ({loudness:.1f}), falling back to pydub normalize")
        except Exception as e:
            logger.warning(f"LUFS normalization failed ({e}), falling back to pydub normalize")

        from pydub import effects
        return effects.normalize(audio_segment)
