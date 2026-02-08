from pathlib import Path

from app.conductors.gpt_conductor import Track
from app.core.feature_extractor import MusicFeatures
from app.core.track_manager import TrackManager


def test_add_track_respects_explicit_id(tmp_path: Path):
    manager = TrackManager(output_dir=str(tmp_path))
    composition_id = manager.create_session()

    features = MusicFeatures(note_density=1.0, note_count=8, duration_seconds=4.0)
    track_id = manager.add_track(
        composition_id=composition_id,
        track_id="track_99",
        instrument="Piano",
        role="melody",
        midi_path=str(tmp_path / "track_99.mid"),
        audio_path=str(tmp_path / "track_99.mp3"),
        features=features,
        metadata={"instruction": "Test"},
    )

    assert track_id == "track_99"
    state = manager.get_state(composition_id)
    assert state is not None
    assert len(state.tracks) == 1
    assert state.tracks[0].id == "track_99"


def test_auto_track_id_increments(tmp_path: Path):
    manager = TrackManager(output_dir=str(tmp_path))
    composition_id = manager.create_session()

    features = MusicFeatures(note_density=1.0, note_count=8, duration_seconds=4.0)
    track_id = manager.add_track(
        composition_id=composition_id,
        track_id=None,
        instrument="Piano",
        role="melody",
        midi_path=str(tmp_path / "track_1.mid"),
        audio_path=str(tmp_path / "track_1.mp3"),
        features=features,
    )

    assert track_id == "track_1"


def test_track_summary_english():
    features = MusicFeatures(
        note_density=2.5,
        note_count=20,
        duration_seconds=8.0,
        pitch_range=(60, 72),
        pitch_mean=66.0,
    )
    track = Track(
        id="track_1",
        instrument="Piano",
        role="melody",
        midi_path="/tmp/track_1.mid",
        audio_path="/tmp/track_1.mp3",
        features=features,
    )
    summary = track.to_summary()
    assert "Track" in summary
    assert "Piano" in summary
