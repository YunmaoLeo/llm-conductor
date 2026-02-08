"""Chat API: GPT-4o Conductor conversation endpoints."""

import asyncio
import json
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


class ChatResponse(BaseModel):
    """Conductor response with updated composition."""

    composition_id: str
    message: str  # Conductor's natural language response
    tracks: list[TrackInfo]
    reasoning: Optional[str] = None  # Debug info


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

        elif action.type == "delete_track":
            track_id = action.parameters.get("track_id")
            if track_id:
                track_manager.remove_track(composition_id, track_id)

        # TODO: Support regenerate_track, modify_track

    # Save conversation turn
    track_manager.add_conversation_turn(
        composition_id, request.message, conductor_response.message
    )

    # Build response
    updated_state = track_manager.get_state(composition_id)
    if not updated_state:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated state")

    tracks_info = [
        TrackInfo(
            id=track.id,
            instrument=track.instrument,
            role=track.role,
            midi_url=f"/api/outputs/{composition_id}/{track.id}.mid",
            audio_url=f"/api/outputs/{composition_id}/{track.id}.mp3",
            features=track.features.__dict__,
        )
        for track in updated_state.tracks
    ]

    return ChatResponse(
        composition_id=composition_id,
        message=conductor_response.message,
        tracks=tracks_info,
        reasoning=conductor_response.reasoning,
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
    instrument = parameters.get("instrument", "Piano")
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

    # Synthesize audio
    track_id = f"track_{len(track_manager.get_state(composition_id).tracks) + 1}"
    synthesizer = get_audio_synthesizer()
    midi_path, audio_path = synthesizer.synthesize_track(
        track_id=track_id,
        midi_bytes=midi_result.midi_bytes,
        output_dir=session_dir,
        format="mp3",
    )

    # Extract features
    features = extract_features(midi_result.pretty_midi)

    # Add track to manager
    track_manager.add_track(
        composition_id=composition_id,
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

                if action.type == "create_track":
                    await _execute_create_track(
                        composition_id=composition_id,
                        session_dir=session_dir,
                        parameters=action.parameters,
                    )

                    # Notify track generated
                    updated_state = track_manager.get_state(composition_id)
                    if updated_state and updated_state.tracks:
                        latest_track = updated_state.tracks[-1]
                        await websocket.send_json({
                            "type": "track_generated",
                            "data": {
                                "track_id": latest_track.id,
                                "instrument": latest_track.instrument,
                                "role": latest_track.role,
                                "midi_url": f"/api/outputs/{composition_id}/{latest_track.id}.mid",
                                "audio_url": f"/api/outputs/{composition_id}/{latest_track.id}.mp3",
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
            if updated_state:
                tracks_info = [
                    {
                        "id": track.id,
                        "instrument": track.instrument,
                        "role": track.role,
                        "midi_url": f"/api/outputs/{composition_id}/{track.id}.mid",
                        "audio_url": f"/api/outputs/{composition_id}/{track.id}.mp3",
                        "features": track.features.__dict__,
                    }
                    for track in updated_state.tracks
                ]

            await websocket.send_json({
                "type": "completed",
                "data": {
                    "composition_id": composition_id,
                    "tracks": tracks_info,
                    "reasoning": conductor_response.reasoning,
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
