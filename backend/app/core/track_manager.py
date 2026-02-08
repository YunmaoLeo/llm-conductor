"""Track Manager: Multi-track composition state management."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.conductors.gpt_conductor import CompositionState, Track
from app.config import settings
from app.core.feature_extractor import MusicFeatures


@dataclass
class CompositionSession:
    """A complete composition session with metadata."""

    composition_id: str
    state: CompositionState
    created_at: str
    updated_at: str
    user_messages: list[str] = field(default_factory=list)
    conductor_responses: list[str] = field(default_factory=list)


class TrackManager:
    """Manages multi-track composition state."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize track manager.

        Args:
            output_dir: Directory for output files (default from settings)
        """
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory sessions (could be moved to database)
        self.sessions: dict[str, CompositionSession] = {}

    def create_session(self) -> str:
        """Create a new composition session.

        Returns:
            Composition ID
        """
        composition_id = str(uuid.uuid4())[:8]

        # Create session directory
        session_dir = self.output_dir / composition_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty composition state
        state = CompositionState(composition_id=composition_id)

        # Create session
        from datetime import datetime

        now = datetime.utcnow().isoformat()
        session = CompositionSession(
            composition_id=composition_id,
            state=state,
            created_at=now,
            updated_at=now,
        )

        self.sessions[composition_id] = session
        return composition_id

    def get_session(self, composition_id: str) -> Optional[CompositionSession]:
        """Get composition session by ID.

        Args:
            composition_id: Composition ID

        Returns:
            CompositionSession or None if not found
        """
        return self.sessions.get(composition_id)

    def get_state(self, composition_id: str) -> Optional[CompositionState]:
        """Get composition state by ID.

        Args:
            composition_id: Composition ID

        Returns:
            CompositionState or None if not found
        """
        session = self.get_session(composition_id)
        return session.state if session else None

    def add_track(
        self,
        composition_id: str,
        instrument: str,
        role: str,
        midi_path: str,
        audio_path: str,
        features: MusicFeatures,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a new track to composition.

        Args:
            composition_id: Composition ID
            instrument: Instrument name
            role: Track role (melody, harmony, rhythm, bass)
            midi_path: Path to MIDI file
            audio_path: Path to audio file
            features: Extracted music features
            metadata: Optional track metadata

        Returns:
            Track ID

        Raises:
            ValueError: If composition not found
        """
        session = self.get_session(composition_id)
        if not session:
            raise ValueError(f"Composition {composition_id} not found")

        # Generate track ID
        track_id = f"track_{len(session.state.tracks) + 1}"

        # Create track
        track = Track(
            id=track_id,
            instrument=instrument,
            role=role,
            midi_path=midi_path,
            audio_path=audio_path,
            features=features,
            metadata=metadata or {},
        )

        # Add to composition
        session.state.tracks.append(track)

        # Update timestamp
        from datetime import datetime

        session.updated_at = datetime.utcnow().isoformat()

        return track_id

    def remove_track(self, composition_id: str, track_id: str) -> bool:
        """Remove a track from composition.

        Args:
            composition_id: Composition ID
            track_id: Track ID to remove

        Returns:
            True if removed, False if not found
        """
        session = self.get_session(composition_id)
        if not session:
            return False

        # Find and remove track
        original_count = len(session.state.tracks)
        session.state.tracks = [t for t in session.state.tracks if t.id != track_id]

        if len(session.state.tracks) < original_count:
            # Update timestamp
            from datetime import datetime

            session.updated_at = datetime.utcnow().isoformat()
            return True

        return False

    def get_track(self, composition_id: str, track_id: str) -> Optional[Track]:
        """Get a specific track.

        Args:
            composition_id: Composition ID
            track_id: Track ID

        Returns:
            Track or None if not found
        """
        session = self.get_session(composition_id)
        if not session:
            return None

        for track in session.state.tracks:
            if track.id == track_id:
                return track

        return None

    def update_metadata(
        self, composition_id: str, metadata: dict
    ) -> bool:
        """Update composition-level metadata.

        Args:
            composition_id: Composition ID
            metadata: Metadata dict to merge

        Returns:
            True if updated, False if composition not found
        """
        session = self.get_session(composition_id)
        if not session:
            return False

        session.state.metadata.update(metadata)

        # Update timestamp
        from datetime import datetime

        session.updated_at = datetime.utcnow().isoformat()
        return True

    def add_conversation_turn(
        self, composition_id: str, user_message: str, conductor_response: str
    ) -> bool:
        """Add a conversation turn to session history.

        Args:
            composition_id: Composition ID
            user_message: User's message
            conductor_response: Conductor's response

        Returns:
            True if added, False if composition not found
        """
        session = self.get_session(composition_id)
        if not session:
            return False

        session.user_messages.append(user_message)
        session.conductor_responses.append(conductor_response)

        # Update timestamp
        from datetime import datetime

        session.updated_at = datetime.utcnow().isoformat()
        return True

    def list_sessions(self) -> list[dict]:
        """List all composition sessions.

        Returns:
            List of session summaries
        """
        return [
            {
                "composition_id": session.composition_id,
                "track_count": len(session.state.tracks),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
            for session in self.sessions.values()
        ]

    def get_session_dir(self, composition_id: str) -> Path:
        """Get the directory for a composition session.

        Args:
            composition_id: Composition ID

        Returns:
            Path to session directory
        """
        return self.output_dir / composition_id
