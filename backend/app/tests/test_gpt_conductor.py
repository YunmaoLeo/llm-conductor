from app.conductors.gpt_conductor import CompositionState, Track
from app.core.feature_extractor import MusicFeatures


def test_composition_context_english():
    state = CompositionState(composition_id="test")
    assert state.to_context().startswith("There are no tracks")

    features = MusicFeatures(
        note_density=1.2,
        note_count=10,
        duration_seconds=5.0,
        pitch_range=(60, 67),
        pitch_mean=64.0,
    )
    state.tracks.append(
        Track(
            id="track_1",
            instrument="Piano",
            role="melody",
            midi_path="/tmp/track_1.mid",
            audio_path="/tmp/track_1.mp3",
            features=features,
        )
    )

    context = state.to_context()
    assert "current composition" in context
    assert "Track track_1" in context
