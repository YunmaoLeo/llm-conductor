"""Audio synthesis service: MIDI → WAV → MP3."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.config import settings

# Optional dependencies for audio synthesis
SYNTHESIS_AVAILABLE = False
LOUDNESS_NORM_AVAILABLE = False

try:
    import librosa
    import librosa.effects
    import midi2audio
    import soundfile as sf

    SYNTHESIS_AVAILABLE = True

    try:
        import pyloudnorm as pyln

        LOUDNESS_NORM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


class AudioSynthesizer:
    """Synthesizes MIDI files to audio (WAV/MP3)."""

    def __init__(self, soundfont_path: str | None = None):
        self.soundfont_path = soundfont_path or settings.soundfont_path

    @property
    def available(self) -> bool:
        return SYNTHESIS_AVAILABLE and os.path.exists(self.soundfont_path)

    def synthesize_to_mp3(
        self,
        midi_path: str,
        output_path: Optional[str] = None,
        target_loudness: float = -18.0,
    ) -> Optional[str]:
        """Synthesize a MIDI file to MP3.

        Args:
            midi_path: Path to the input MIDI file.
            output_path: Path for the output MP3. Defaults to same dir as MIDI.
            target_loudness: Target loudness in LUFS.

        Returns:
            Path to the generated MP3 file, or None if synthesis failed.
        """
        if not SYNTHESIS_AVAILABLE:
            return None

        if output_path is None:
            output_path = str(Path(midi_path).with_suffix(".mp3"))

        try:
            # MIDI → WAV via FluidSynth
            wav_path = str(Path(midi_path).with_suffix(".wav"))
            fs = midi2audio.FluidSynth(self.soundfont_path)
            fs.midi_to_audio(midi_path, wav_path)

            # Load and trim silence
            wav, sr = librosa.load(wav_path)
            wav, _ = librosa.effects.trim(wav, top_db=30)

            # Loudness normalization
            if LOUDNESS_NORM_AVAILABLE:
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(wav)
                    wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                    if wav.max() > 1.0 or wav.min() < -1.0:
                        wav = wav / max(abs(wav.max()), abs(wav.min()))
                except Exception:
                    pass

            # Write normalized WAV
            sf.write(wav_path, wav, sr)

            # WAV → MP3 via ffmpeg
            subprocess.run(
                [
                    "ffmpeg", "-i", wav_path,
                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                    output_path, "-y",
                ],
                capture_output=True,
                check=True,
            )

            # Clean up WAV
            if os.path.exists(wav_path):
                os.remove(wav_path)

            return output_path

        except Exception as e:
            # Clean up on failure
            for path in [wav_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
            return None

    def synthesize_bytes(
        self,
        midi_bytes: bytes,
        target_loudness: float = -18.0,
    ) -> Optional[bytes]:
        """Synthesize MIDI bytes to MP3 bytes (in-memory workflow).

        Args:
            midi_bytes: Raw MIDI file content.
            target_loudness: Target loudness in LUFS.

        Returns:
            MP3 file content as bytes, or None if synthesis failed.
        """
        if not self.available:
            return None

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_tmp:
            midi_tmp.write(midi_bytes)
            midi_path = midi_tmp.name

        try:
            mp3_path = self.synthesize_to_mp3(midi_path, target_loudness=target_loudness)
            if mp3_path and os.path.exists(mp3_path):
                with open(mp3_path, "rb") as f:
                    mp3_bytes = f.read()
                os.remove(mp3_path)
                return mp3_bytes
            return None
        finally:
            if os.path.exists(midi_path):
                os.remove(midi_path)
