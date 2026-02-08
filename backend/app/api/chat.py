"""Chat API: GPT-4o Conductor conversation endpoints."""

import asyncio
import json
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.conductors.gpt_conductor import GPTConductor, CompositionState
from app.config import settings
from app.core.audio_synthesis import AudioSynthesizer
from app.core.feature_extractor import extract_features
from app.core.token_processor import TokenProcessor
from app.core.track_manager import TrackManager
from app.musicians.midi_llm_musician import MIDILLMMusician


router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

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
    has_previous_version: bool = False  # NEW: Indicates if _prev files exist
    previous_version_number: Optional[int] = None  # NEW: Previous version number


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
        _create_track_info(track, composition_id)
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


def _build_refinement_instruction(
    base_instruction: str,
    track,  # Track object from gpt_conductor
    mode: str
) -> str:
    """Build instruction with style preservation constraints.

    Args:
        base_instruction: User's modification request
        track: Existing track to refine
        mode: Refinement mode (style, details, or full_regen)

    Returns:
        Enhanced instruction with style preservation hints
    """
    if mode == "full_regen":
        return base_instruction  # No preservation needed

    # Extract style descriptors from track features
    features = track.features
    density = "sparse" if features.note_density < 1.0 else "dense"
    pitch_mid = (features.pitch_range[0] + features.pitch_range[1]) / 2
    register = "low" if pitch_mid < 60 else "mid" if pitch_mid < 72 else "high"

    if mode == "refine_style":
        template = (
            f"PRESERVE: {density} texture, {register} register, "
            f"pitch range {features.pitch_range[0]}-{features.pitch_range[1]}\n"
            f"MODIFY: {base_instruction}\n"
            f"Keep overall musical character similar."
        )
    elif mode == "refine_details":
        template = (
            f"KEEP ALMOST EVERYTHING: Current density={features.note_density:.1f}, "
            f"register={register}\n"
            f"SMALL ADJUSTMENT: {base_instruction}"
        )
    else:
        template = base_instruction

    return template


async def _execute_create_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
    websocket: Optional[WebSocket] = None,
) -> None:
    """Execute a create_track action.

    Args:
        composition_id: Composition ID
        session_dir: Session output directory
        parameters: Action parameters (instrument, role, instruction)
        websocket: Optional WebSocket for debug messages
    """
    instrument = _infer_instrument(
        parameters.get("instrument", ""), parameters.get("instruction", "")
    )
    role = parameters.get("role", "melody")
    instruction = parameters.get("instruction", "")

    # Send debug message with MIDI-LLM prompt
    if websocket:
        await websocket.send_json({
            "type": "debug",
            "data": {
                "message": f"[MIDI-LLM] Generating {instrument} ({role})",
                "prompt": instruction,
            },
        })

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
            "midi_token_ids": result.midi_token_ids,  # NEW: Save tokens for future refinement
            "refinement_history": [],  # NEW: Track refinement chain
        },
    )


async def _execute_regenerate_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
    websocket: Optional[WebSocket] = None,
) -> None:
    """Execute a regenerate_track action."""
    track_id = parameters.get("track_id")
    if not track_id:
        return

    instruction = parameters.get("instruction", "")
    existing = track_manager.get_track(composition_id, track_id)
    if not existing:
        return

    # Backup old tokens and instruction before regenerating (for refinement history)
    old_tokens = existing.metadata.get("midi_token_ids", [])
    old_instruction = existing.metadata.get("instruction", "")

    instrument = _infer_instrument(
        parameters.get("instrument", ""), instruction or existing.instrument
    )
    role = parameters.get("role", existing.role)

    # Check refinement mode for style preservation and progress notifications
    refinement_mode = parameters.get("refinement_mode", "full_regen")

    # Send debug message with MIDI-LLM prompt and refinement mode
    if websocket:
        mode_label = {
            "full_regen": "Full Regeneration",
            "refine_style": "Style Refinement",
            "refine_details": "Detail Refinement",
        }.get(refinement_mode, refinement_mode)

        await websocket.send_json({
            "type": "debug",
            "data": {
                "message": f"[MIDI-LLM] {mode_label}: {track_id} ({instrument})",
                "prompt": instruction,
                "refinement_mode": refinement_mode,
            },
        })

    # Enhance instruction based on refinement mode
    if refinement_mode == "refine_style" and existing:
        instruction = _build_refinement_instruction(
            base_instruction=instruction,
            track=existing,
            mode="style"
        )
    elif refinement_mode == "refine_details" and existing:
        instruction = _build_refinement_instruction(
            base_instruction=instruction,
            track=existing,
            mode="refine_details"
        )

    # Prepend track state summary to instruction for better continuity
    context_prefix = ""
    if existing:
        # Add reference track state to help MIDI-LLM preserve style
        context_prefix = (
            f"[REFERENCE TRACK STATE]\n"
            f"{existing.to_detailed_summary(include_style_hints=True)}\n"
            f"[MODIFICATION REQUEST]\n"
        )

    # Combine context + instruction
    full_instruction = context_prefix + instruction

    musician = MIDILLMMusician()
    try:
        # Experimental: Try token prefix continuation for refine_style mode
        if refinement_mode == "refine_style" and old_tokens:
            if hasattr(musician, 'generate_with_prefix'):
                try:
                    logger.info(f"Attempting token prefix continuation with {len(old_tokens)} old tokens")
                    result = await musician.generate_with_prefix(
                        instruction=full_instruction,
                        prefix_tokens=old_tokens,
                        prefix_ratio=0.3,
                    )
                except Exception as e:
                    logger.warning(f"Prefix continuation failed: {e}, falling back to normal generation")
                    result = await musician.generate(full_instruction)
            else:
                result = await musician.generate(full_instruction)
        else:
            # Full regeneration (current behavior)
            result = await musician.generate(full_instruction)
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
            "midi_token_ids": result.midi_token_ids,  # NEW: Current tokens
            "previous_midi_token_ids": old_tokens,     # NEW: Backup old tokens
            "refinement_history": existing.metadata.get("refinement_history", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "instruction": old_instruction,
                "token_count": len(old_tokens),
            }] if old_tokens else existing.metadata.get("refinement_history", []),
        },
    )


async def _execute_modify_track(
    composition_id: str,
    session_dir: Path,
    parameters: dict,
    websocket: Optional[WebSocket] = None,  # NEW: WebSocket support
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
        # Send debug message with prompt before regenerating
        if websocket:
            await websocket.send_json({
                "type": "debug",
                "data": {
                    "message": f"[MIDI-LLM] Modifying {track_id}",
                    "prompt": instruction,
                },
            })

        await _execute_regenerate_track(
            composition_id=composition_id,
            session_dir=session_dir,
            parameters={
                "track_id": track_id,
                "instruction": instruction,
                "instrument": instrument,
                "role": role,
            },
            websocket=websocket,  # NEW: Pass websocket to regenerate
        )
        return

    # Metadata-only update
    if websocket:
        await websocket.send_json({
            "type": "debug",
            "data": {
                "message": f"[Metadata] Updating {track_id} metadata only (no regeneration)",
            },
        })

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


def _create_track_info(track, composition_id: str) -> TrackInfo:
    """Create TrackInfo object with version checking.

    Args:
        track: Track object from track_manager
        composition_id: Composition ID

    Returns:
        TrackInfo with has_previous_version field populated
    """
    # Check if _prev files exist for this track
    output_dir = Path(settings.output_dir) / composition_id
    prev_audio_exists = (output_dir / f"{track.id}_prev.mp3").exists()

    version = track.metadata.get("version", 1)

    return TrackInfo(
        id=track.id,
        instrument=track.instrument,
        role=track.role,
        midi_url=_track_url(composition_id, track.id, "mid", version),
        audio_url=_track_url(composition_id, track.id, "mp3", version),
        features=track.features.__dict__,
        version=version,
        has_previous_version=prev_audio_exists,
        previous_version_number=version - 1 if prev_audio_exists else None,
    )


def _apply_instrument_override(
    pretty, instrument_name: str
) -> tuple[bytes, "pretty_midi.PrettyMIDI"]:
    """Force the instrument to be single-track with the requested instrument.

    CRITICAL FIX: MIDI-LLM often generates multiple instrument tracks (piano+strings+bass).
    This function now:
    1. Merges all non-drum tracks into a SINGLE track
    2. Sets the program to the requested instrument
    3. Removes duplicate/overlapping notes

    This ensures clean, single-instrument tracks instead of chaotic multi-instrument mixes.
    """
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

    # Handle drums specially
    if "drum" in name:
        drum_tracks = [inst for inst in pretty.instruments if inst.is_drum]
        if drum_tracks:
            pretty.instruments = drum_tracks[:1]  # Keep only first drum track
        else:
            # Convert first track to drums
            if pretty.instruments:
                pretty.instruments[0].is_drum = True
                pretty.instruments = pretty.instruments[:1]
        buf = io.BytesIO()
        pretty.write(buf)
        return buf.getvalue(), pretty

    # For non-drum instruments: MERGE all non-drum tracks into ONE
    if target is not None:
        non_drum_instruments = [inst for inst in pretty.instruments if not inst.is_drum]

        if not non_drum_instruments:
            # No non-drum tracks, create empty one
            merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)
        elif len(non_drum_instruments) == 1:
            # Only one track, just change program
            merged = non_drum_instruments[0]
            merged.program = target
            merged.name = instrument_name
        else:
            # MULTIPLE TRACKS - MERGE THEM
            merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)

            # Collect all notes from all non-drum tracks
            all_notes = []
            for inst in non_drum_instruments:
                all_notes.extend(inst.notes)

            # Sort by start time
            all_notes.sort(key=lambda n: n.start)

            # Remove overlapping/duplicate notes (keep unique notes only)
            unique_notes = []
            for note in all_notes:
                # Check if this note is too similar to existing notes
                is_duplicate = False
                for existing in unique_notes:
                    # Same pitch, overlapping time → duplicate
                    if (note.pitch == existing.pitch and
                        abs(note.start - existing.start) < 0.05):  # Within 50ms
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_notes.append(note)

            merged.notes = unique_notes

        # Replace all instruments with the single merged track
        pretty.instruments = [merged]

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
                        websocket=websocket,
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
                        websocket=websocket,
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
                        websocket=websocket,  # NEW: Pass websocket for prompt logging
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
                    _create_track_info(track, composition_id).model_dump()
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
        _create_track_info(track, composition_id)
        for track in composition_state.tracks
    ]

    return {
        "composition_id": composition_id,
        "tracks": tracks_info,
        "mix_midi_url": mix_urls.get("midi_url"),
        "mix_audio_url": mix_urls.get("audio_url"),
        "metadata": composition_state.metadata,
    }


@router.get("/tracks/{composition_id}/{track_id}/previous")
async def get_previous_version(composition_id: str, track_id: str, type: str = "audio"):
    """Get previous version of track audio or MIDI.

    Allows comparison between current and previous versions by serving
    the *_prev backup files created during track regeneration.

    Args:
        composition_id: Composition ID
        track_id: Track ID (e.g., "track_1")
        type: File type - "audio" for MP3, "midi" for MIDI (default: "audio")

    Returns:
        FileResponse with audio/midi file

    Raises:
        HTTPException: 404 if previous version not found, 400 if invalid type
    """
    output_dir = Path(settings.output_dir) / composition_id

    if type == "audio":
        file_path = output_dir / f"{track_id}_prev.mp3"
        media_type = "audio/mpeg"
        filename = f"{track_id}_prev.mp3"
    elif type == "midi":
        file_path = output_dir / f"{track_id}_prev.mid"
        media_type = "audio/midi"
        filename = f"{track_id}_prev.mid"
    else:
        raise HTTPException(status_code=400, detail="Type must be 'audio' or 'midi'")

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Previous version not found for {track_id}. This track may not have been regenerated yet."
        )

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/tracks/{composition_id}/{track_id}/previous/features")
async def get_previous_features(composition_id: str, track_id: str):
    """Get features extracted from previous version MIDI file.

    This endpoint reads the *_prev.mid file and extracts musical features
    for comparison with the current version.

    Args:
        composition_id: Composition ID
        track_id: Track ID (e.g., "track_1")

    Returns:
        JSON object with MusicFeatures data

    Raises:
        HTTPException: 404 if previous version not found
    """
    import pretty_midi
    from app.core.feature_extractor import extract_features

    output_dir = Path(settings.output_dir) / composition_id
    prev_midi_path = output_dir / f"{track_id}_prev.mid"

    if not prev_midi_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Previous version not found for {track_id}"
        )

    # Load MIDI and extract features
    try:
        midi = pretty_midi.PrettyMIDI(str(prev_midi_path))
        features = extract_features(midi)

        # Return features as dict (using to_dict method if available, otherwise asdict)
        if hasattr(features, 'to_dict'):
            return features.to_dict()
        else:
            from dataclasses import asdict
            result = asdict(features)
            # Ensure pitch_range is a list for JSON serialization
            if 'pitch_range' in result and isinstance(result['pitch_range'], tuple):
                result['pitch_range'] = list(result['pitch_range'])
            return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract features from previous version: {str(e)}"
        )
