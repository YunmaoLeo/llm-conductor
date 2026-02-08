"""Chat API: GPT-4o Conductor conversation endpoints."""

import asyncio
import json
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.conductors.gpt_conductor import GPTConductor, CompositionState
from app.config import settings
from app.core.audio_synthesis import AudioSynthesizer
from app.core.feature_extractor import extract_features
from app.core.token_processor import TokenProcessor
from app.core.track_manager import TrackManager
from app.musicians.midi_llm_musician import MIDILLMMusician


router = APIRouter(prefix="/api", tags=["chat"])

# Global instances (in production, use dependency injection)
track_manager = TrackManager()
conductor = None  # Lazy init to check API key
token_processor = TokenProcessor()
audio_synthesizer = None  # Lazy init to check soundfont


def get_conductor() -> GPTConductor:
    """Get or create GPT-4o Conductor instance."""
    global conductor
    if conductor is None:
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Set OPENAI_API_KEY in .env",
            )
        conductor = GPTConductor(api_key=settings.openai_api_key)
    return conductor


def get_audio_synthesizer() -> AudioSynthesizer:
    """Get or create AudioSynthesizer instance."""
    global audio_synthesizer
    if audio_synthesizer is None:
        try:
            audio_synthesizer = AudioSynthesizer()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )
    return audio_synthesizer


# Request/Response schemas
class ChatRequest(BaseModel):
    """User message to Conductor."""

    composition_id: Optional[str] = None  # None = new session
    message: str


class TrackInfo(BaseModel):
    """Track information for response."""

    id: str
    instrument: str
    role: str
    midi_url: str
    audio_url: str
    features: dict
    version: int = 1


class ChatResponse(BaseModel):
    """Conductor response with updated composition."""

    composition_id: str
    message: str  # Conductor's natural language response
    tracks: list[TrackInfo]
    reasoning: Optional[str] = None  # Debug info
    mix_midi_url: Optional[str] = None
    mix_audio_url: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with GPT-4o Conductor (blocking REST endpoint).

    Args:
        request: User message + optional composition ID

    Returns:
        Conductor response with updated composition state
    """
    # Get or create session
    composition_id = request.composition_id
    if not composition_id:
        composition_id = track_manager.create_session()

    composition_state = track_manager.get_state(composition_id)
    if not composition_state:
        raise HTTPException(status_code=404, detail="Composition not found")

    # Ask Conductor to plan actions
    conductor_instance = get_conductor()
    conductor_response = await conductor_instance.plan_action(
        user_message=request.message,
        composition_state=composition_state,
    )

    # Execute actions
    session_dir = track_manager.get_session_dir(composition_id)

    for action in conductor_response.actions:
        if action.type == "create_track":
            await _execute_create_track(
                composition_id=composition_id,
                session_dir=session_dir,
                parameters=action.parameters,
            )

        elif action.type == "regenerate_track":
            await _execute_regenerate_track(
                composition_id=composition_id,
                session_dir=session_dir,
                parameters=action.parameters,
            )

        elif action.type == "modify_track":
            await _execute_modify_track(
                composition_id=composition_id,
                session_dir=session_dir,
                parameters=action.parameters,
            )

        elif action.type == "delete_track":
            track_id = action.parameters.get("track_id")
            if track_id:
                track_manager.remove_track(composition_id, track_id)

        # modify_track is supported

    # Save conversation turn
    track_manager.add_conversation_turn(
        composition_id, request.message, conductor_response.message
    )

    # Build response
    updated_state = track_manager.get_state(composition_id)
    if not updated_state:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated state")

    mix_urls = await _ensure_mix(composition_id, session_dir, updated_state)

    tracks_info = [
        TrackInfo(
            id=track.id,
            instrument=track.instrument,
            role=track.role,
            midi_url=_track_url(
                composition_id, track.id, "mid", track.metadata.get("version", 1)
            ),
            audio_url=_track_url(
                composition_id, track.id, "mp3", track.metadata.get("version", 1)
            ),
            features=track.features.__dict__,
            version=track.metadata.get("version", 1),
        )
        for track in updated_state.tracks
    ]

    return ChatResponse(
        composition_id=composition_id,
        message=conductor_response.message,
        tracks=tracks_info,
        reasoning=conductor_response.reasoning,
        mix_midi_url=mix_urls.get("midi_url"),
        mix_audio_url=mix_urls.get("audio_url"),
    )


async def _execute_create_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
) -> None:
    """Execute a create_track action.

    Args:
        composition_id: Composition ID
        session_dir: Session output directory
        parameters: Action parameters (instrument, role, instruction)
    """
    instrument = _infer_instrument(
        parameters.get("instrument", ""), parameters.get("instruction", "")
    )
    role = parameters.get("role", "melody")
    instruction = parameters.get("instruction", "")

    # Generate MIDI tokens using MIDI-LLM
    musician = MIDILLMMusician()
    try:
        result = await musician.generate(instruction)
    finally:
        await musician.close()

    # Convert tokens to MIDI
    midi_result = token_processor.tokens_to_midi(result.midi_token_ids)
    midi_bytes, pretty = _apply_instrument_override(
        midi_result.pretty_midi, instrument
    )

    # Synthesize audio
    track_id = f"track_{len(track_manager.get_state(composition_id).tracks) + 1}"
    synthesizer = get_audio_synthesizer()
    midi_path, audio_path = synthesizer.synthesize_track(
        track_id=track_id,
        midi_bytes=midi_bytes,
        output_dir=session_dir,
        format="mp3",
    )

    # Extract features
    features = extract_features(pretty)

    # Add track to manager
    track_manager.add_track(
        composition_id=composition_id,
        track_id=track_id,
        instrument=instrument,
        role=role,
        midi_path=str(midi_path),
        audio_path=str(audio_path),
        features=features,
        metadata={
            "instruction": instruction,
            "token_count": len(result.midi_token_ids),
            "generation_time_ms": result.generation_time_ms,
        },
    )


async def _execute_regenerate_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
) -> None:
    """Execute a regenerate_track action."""
    track_id = parameters.get("track_id")
    if not track_id:
        return

    instruction = parameters.get("instruction", "")
    existing = track_manager.get_track(composition_id, track_id)
    if not existing:
        return

    instrument = _infer_instrument(
        parameters.get("instrument", ""), instruction or existing.instrument
    )
    role = parameters.get("role", existing.role)

    musician = MIDILLMMusician()
    try:
        result = await musician.generate(instruction)
    finally:
        await musician.close()

    midi_result = token_processor.tokens_to_midi(result.midi_token_ids)
    midi_bytes, pretty = _apply_instrument_override(
        midi_result.pretty_midi, instrument
    )

    synthesizer = get_audio_synthesizer()
    midi_path, audio_path = synthesizer.synthesize_track(
        track_id=track_id,
        midi_bytes=midi_bytes,
        output_dir=session_dir,
        format="mp3",
    )

    features = extract_features(pretty)

    track_manager.update_track(
        composition_id=composition_id,
        track_id=track_id,
        instrument=instrument,
        role=role,
        midi_path=str(midi_path),
        audio_path=str(audio_path),
        features=features,
        metadata={
            "instruction": instruction,
            "token_count": len(result.midi_token_ids),
            "generation_time_ms": result.generation_time_ms,
            "instrument": instrument,
            "role": role,
        },
    )


async def _execute_modify_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
) -> None:
    """Execute a modify_track action.

    If instruction is provided, regenerate the track with the new instruction.
    Otherwise, update metadata/instrument/role only.
    """
    track_id = parameters.get("track_id")
    if not track_id:
        return

    existing = track_manager.get_track(composition_id, track_id)
    if not existing:
        return

    instruction = parameters.get("instruction", "")
    instrument = _infer_instrument(
        parameters.get("instrument", ""), instruction or existing.instrument
    )
    role = parameters.get("role", existing.role)

    if instruction:
        await _execute_regenerate_track(
            composition_id=composition_id,
            session_dir=session_dir,
            parameters={
                "track_id": track_id,
                "instruction": instruction,
                "instrument": instrument,
                "role": role,
            },
        )
        return

    track_manager.update_track(
        composition_id=composition_id,
        track_id=track_id,
        instrument=instrument,
        role=role,
        midi_path=existing.midi_path,
        audio_path=existing.audio_path,
        features=existing.features,
        metadata={
            "instruction": existing.metadata.get("instruction", ""),
            "instrument": instrument,
            "role": role,
        },
    )


def _track_url(composition_id: str, track_id: str, ext: str, version: int) -> str:
    return f"/api/outputs/{composition_id}/{track_id}.{ext}?v={version}"


def _mix_url(composition_id: str, filename: str, version: int | None) -> str:
    suffix = f"?v={version}" if version else ""
    return f"/api/outputs/{composition_id}/{filename}{suffix}"


def _apply_instrument_override(
    pretty, instrument_name: str
) -> tuple[bytes, "pretty_midi.PrettyMIDI"]:
    """Force the instrument program to match the requested instrument."""
    import pretty_midi

    program_map = {
        "piano": 0,
        "electric piano": 4,
        "keys": 0,
        "strings": 48,
        "string": 48,
        "violin": 40,
        "cello": 42,
        "bass": 32,
        "bass guitar": 34,
        "guitar": 24,
        "acoustic guitar": 24,
        "electric guitar": 27,
        "sax": 65,
        "saxophone": 65,
        "trumpet": 56,
        "flute": 73,
        "clarinet": 71,
        "choir": 52,
        "pad": 88,
        "organ": 19,
        "harp": 46,
        "brass": 61,
    }

    name = instrument_name.lower().strip()
    target = program_map.get(name)
    if target is None:
        for key, value in program_map.items():
            if key in name:
                target = value
                break

    if "drum" in name:
        for inst in pretty.instruments:
            inst.is_drum = True
        target = None

    if target is not None:
        for inst in pretty.instruments:
            if not inst.is_drum:
                inst.program = target

    buf = io.BytesIO()
    pretty.write(buf)
    return buf.getvalue(), pretty


def _infer_instrument(instrument: str, instruction: str) -> str:
    """Infer instrument from instruction if not provided."""
    if instrument:
        return instrument

    text = instruction.lower()
    keywords = [
        "piano",
        "electric piano",
        "strings",
        "string",
        "violin",
        "cello",
        "bass",
        "guitar",
        "drum",
        "sax",
        "trumpet",
        "flute",
        "clarinet",
        "choir",
        "pad",
        "organ",
        "harp",
        "brass",
    ]
    for key in keywords:
        if key in text:
            return key.title() if key != "drum" else "Drums"

    return "Piano"


def _mix_url(composition_id: str, filename: str, version: int | None) -> str:
    suffix = f"?v={version}" if version else ""
    return f"/api/outputs/{composition_id}/{filename}{suffix}"


async def _ensure_mix(
    composition_id: str,
    session_dir: Path,
    composition_state: CompositionState,
) -> dict:
    """Ensure a mixed MIDI/MP3 exists for the composition."""
    if not composition_state.tracks:
        return {}

    existing_midi = composition_state.metadata.get("mix_midi_path")
    existing_audio = composition_state.metadata.get("mix_audio_path")
    existing_track_versions = composition_state.metadata.get("mix_track_versions")
    existing_mix_version = composition_state.metadata.get("mix_version")
    current_track_versions = [
        {"id": track.id, "version": track.metadata.get("version", 1)}
        for track in composition_state.tracks
    ]
    if (
        existing_midi
        and existing_audio
        and existing_track_versions == current_track_versions
    ):
        midi_path = Path(existing_midi)
        audio_path = Path(existing_audio)
        if midi_path.exists() and audio_path.exists():
            return {
                "midi_url": _mix_url(composition_id, midi_path.name, existing_mix_version),
                "audio_url": _mix_url(composition_id, audio_path.name, existing_mix_version),
            }

    track_midi_paths = [Path(track.midi_path) for track in composition_state.tracks]
    synthesizer = get_audio_synthesizer()
    try:
        combined_midi, mixed_audio = synthesizer.synthesize_mix(
            composition_id=composition_id,
            track_midi_paths=track_midi_paths,
            output_dir=session_dir,
            format="mp3",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mix synthesis failed: {e}")

    import time
    mix_version = int(time.time() * 1000)

    track_manager.update_metadata(
        composition_id,
        {
            "mix_midi_path": str(combined_midi),
            "mix_audio_path": str(mixed_audio),
            "mix_track_versions": current_track_versions,
            "mix_version": mix_version,
        },
    )

    return {
        "midi_url": _mix_url(composition_id, combined_midi.name, mix_version),
        "audio_url": _mix_url(composition_id, mixed_audio.name, mix_version),
    }


# WebSocket endpoint for streaming progress
@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    """Chat with Conductor via WebSocket (streaming progress).

    Message format (client → server):
    {
        "composition_id": "abc123" | null,
        "message": "Add a piano melody"
    }

    Message format (server → client):
    {
        "type": "status" | "action" | "track_generated" | "completed" | "error",
        "data": {...}
    }
    """
    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request_data = json.loads(data)

            composition_id = request_data.get("composition_id")
            user_message = request_data.get("message", "")

            # Get or create session
            if not composition_id:
                composition_id = track_manager.create_session()
                await websocket.send_json({
                    "type": "status",
                    "data": {"message": f"Created new composition: {composition_id}"},
                })

            composition_state = track_manager.get_state(composition_id)
            if not composition_state:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Composition not found"},
                })
                continue

            # Ask Conductor
            await websocket.send_json({
                "type": "status",
                "data": {"message": "Conductor is thinking..."},
            })
            await websocket.send_json({
                "type": "debug",
                "data": {"message": f"Received user message: {user_message}"},
            })

            conductor_instance = get_conductor()
            conductor_response = await conductor_instance.plan_action(
                user_message=user_message,
                composition_state=composition_state,
            )

            # Send Conductor's message
            await websocket.send_json({
                "type": "conductor_message",
                "data": {"message": conductor_response.message},
            })
            await websocket.send_json({
                "type": "debug",
                "data": {
                    "message": f"Planned actions: {len(conductor_response.actions)}",
                    "actions": [a.type for a in conductor_response.actions],
                },
            })

            if not conductor_response.actions:
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "message": "No actions were planned. Try asking for a specific change (e.g., regenerate track_1)."
                    },
                })

            # Execute actions
            session_dir = track_manager.get_session_dir(composition_id)

            for i, action in enumerate(conductor_response.actions, 1):
                await websocket.send_json({
                    "type": "action",
                    "data": {
                        "action_type": action.type,
                        "parameters": action.parameters,
                        "progress": f"{i}/{len(conductor_response.actions)}",
                    },
                })
                await websocket.send_json({
                    "type": "debug",
                    "data": {
                        "message": f"Executing action {i}/{len(conductor_response.actions)}: {action.type}",
                        "parameters": action.parameters,
                    },
                })

                if action.type == "create_track":
                    await _execute_create_track(
                        composition_id=composition_id,
                        session_dir=session_dir,
                        parameters=action.parameters,
                    )
                    updated_state = track_manager.get_state(composition_id)
                    if updated_state and updated_state.tracks:
                        latest_track = updated_state.tracks[-1]
                        await websocket.send_json({
                            "type": "track_generated",
                            "data": {
                                "track_id": latest_track.id,
                                "instrument": latest_track.instrument,
                                "role": latest_track.role,
                                "midi_url": _track_url(
                                    composition_id,
                                    latest_track.id,
                                    "mid",
                                    latest_track.metadata.get("version", 1),
                                ),
                                "audio_url": _track_url(
                                    composition_id,
                                    latest_track.id,
                                    "mp3",
                                    latest_track.metadata.get("version", 1),
                                ),
                            },
                        })

                elif action.type == "regenerate_track":
                    await _execute_regenerate_track(
                        composition_id=composition_id,
                        session_dir=session_dir,
                        parameters=action.parameters,
                    )
                    updated_state = track_manager.get_state(composition_id)
                    track_id = action.parameters.get("track_id")
                    if updated_state and track_id:
                        updated_track = track_manager.get_track(composition_id, track_id)
                        if updated_track:
                            await websocket.send_json({
                                "type": "track_updated",
                                "data": {
                                    "track_id": updated_track.id,
                                    "instrument": updated_track.instrument,
                                    "role": updated_track.role,
                                    "midi_url": _track_url(
                                        composition_id,
                                        updated_track.id,
                                        "mid",
                                        updated_track.metadata.get("version", 1),
                                    ),
                                    "audio_url": _track_url(
                                        composition_id,
                                        updated_track.id,
                                        "mp3",
                                        updated_track.metadata.get("version", 1),
                                    ),
                                },
                            })

                elif action.type == "modify_track":
                    await _execute_modify_track(
                        composition_id=composition_id,
                        session_dir=session_dir,
                        parameters=action.parameters,
                    )
                    updated_state = track_manager.get_state(composition_id)
                    track_id = action.parameters.get("track_id")
                    if updated_state and track_id:
                        updated_track = track_manager.get_track(composition_id, track_id)
                        if updated_track:
                            await websocket.send_json({
                                "type": "track_updated",
                                "data": {
                                    "track_id": updated_track.id,
                                    "instrument": updated_track.instrument,
                                    "role": updated_track.role,
                                    "midi_url": _track_url(
                                        composition_id,
                                        updated_track.id,
                                        "mid",
                                        updated_track.metadata.get("version", 1),
                                    ),
                                    "audio_url": _track_url(
                                        composition_id,
                                        updated_track.id,
                                        "mp3",
                                        updated_track.metadata.get("version", 1),
                                    ),
                                },
                            })

                elif action.type == "delete_track":
                    track_id = action.parameters.get("track_id")
                    if track_id:
                        track_manager.remove_track(composition_id, track_id)

            # Save conversation
            track_manager.add_conversation_turn(
                composition_id, user_message, conductor_response.message
            )

            # Send completion
            updated_state = track_manager.get_state(composition_id)
            tracks_info = []
            mix_urls = {}
            if updated_state:
                mix_urls = await _ensure_mix(composition_id, session_dir, updated_state)
                tracks_info = [
                    {
                        "id": track.id,
                        "instrument": track.instrument,
                        "role": track.role,
                        "midi_url": _track_url(
                            composition_id,
                            track.id,
                            "mid",
                            track.metadata.get("version", 1),
                        ),
                        "audio_url": _track_url(
                            composition_id,
                            track.id,
                            "mp3",
                            track.metadata.get("version", 1),
                        ),
                        "features": track.features.__dict__,
                        "version": track.metadata.get("version", 1),
                    }
                    for track in updated_state.tracks
                ]

            await websocket.send_json({
                "type": "completed",
                "data": {
                    "composition_id": composition_id,
                    "tracks": tracks_info,
                    "reasoning": conductor_response.reasoning,
                    "mix_midi_url": mix_urls.get("midi_url"),
                    "mix_audio_url": mix_urls.get("audio_url"),
                },
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)},
        })


@router.get("/compositions")
async def list_compositions():
    """List all composition sessions."""
    return {"compositions": track_manager.list_sessions()}


@router.get("/compositions/{composition_id}")
async def get_composition(composition_id: str):
    """Get composition state and track URLs."""
    composition_state = track_manager.get_state(composition_id)
    if not composition_state:
        raise HTTPException(status_code=404, detail="Composition not found")

    session_dir = track_manager.get_session_dir(composition_id)
    mix_urls = await _ensure_mix(composition_id, session_dir, composition_state)

    tracks_info = [
        TrackInfo(
            id=track.id,
            instrument=track.instrument,
            role=track.role,
            midi_url=_track_url(
                composition_id,
                track.id,
                "mid",
                track.metadata.get("version", 1),
            ),
            audio_url=_track_url(
                composition_id,
                track.id,
                "mp3",
                track.metadata.get("version", 1),
            ),
            features=track.features.__dict__,
            version=track.metadata.get("version", 1),
        )
        for track in composition_state.tracks
    ]

    return {
        "composition_id": composition_id,
        "tracks": tracks_info,
        "mix_midi_url": mix_urls.get("midi_url"),
        "mix_audio_url": mix_urls.get("audio_url"),
        "metadata": composition_state.metadata,
    }
