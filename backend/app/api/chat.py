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

from app.agent.critic import GPTCritic
from app.agent.evaluator import Evaluator
from app.agent.planner import Planner
from app.core.example_db import MIDIExampleDB
from app.core.prompt_builder import PromptBuilder, GenerationPlan
from app.conductors.gpt_conductor import GPTConductor, CompositionState
from app.config import settings
from app.core.audio_synthesis import AudioSynthesizer
from app.core.feature_extractor import extract_features, MusicFeatures
from app.core.token_processor import TokenProcessor
from app.core.track_manager import TrackManager
from app.musicians.midi_llm_musician import MIDILLMMusician, MusicianGenerationResult


router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

# Global instances (in production, use dependency injection)
track_manager = TrackManager()
conductor = None  # Lazy init to check API key
token_processor = TokenProcessor()
audio_synthesizer = None  # Lazy init to check soundfont
example_db = MIDIExampleDB()  # Few-shot example retrieval database


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

    # Build conversation history from session
    session = track_manager.get_session(composition_id)
    conversation_history = None
    if session and session.user_messages:
        conversation_history = [
            {"user": u, "conductor": c}
            for u, c in zip(session.user_messages, session.conductor_responses)
        ]

    # Ask Conductor to plan actions
    conductor_instance = get_conductor()
    conductor_response = await conductor_instance.plan_action(
        user_message=request.message,
        composition_state=composition_state,
        conversation_history=conversation_history,
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

        elif action.type == "adjust_volume":
            await _execute_adjust_volume(
                composition_id=composition_id,
                parameters=action.parameters,
            )

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


def _analyze_feature_change_magnitude(instruction: str, track) -> float:
    """Analyze how much the instruction requests feature changes.

    Detects keywords and numerical targets in the instruction to estimate
    the magnitude of requested changes. Returns a value between 0.0 (no change)
    and 1.0+ (very large change).

    Args:
        instruction: User's modification instruction
        track: Existing track with current features

    Returns:
        Change magnitude (0.0-1.0+, where >0.5 is considered "large")
    """
    import re

    instruction_lower = instruction.lower()
    current_density = track.features.note_density

    # Initialize change score
    change_score = 0.0

    # Detect explicit density/tempo change requests
    density_keywords = {
        "sparse": 0.8,      # "make it sparse" = large change if currently dense
        "simpler": 0.6,
        "minimal": 0.7,
        "reduce": 0.5,
        "decrease": 0.5,
        "less busy": 0.6,
        "slower": 0.4,
        "much slower": 0.7,
        "drastically": 0.8,
        "completely different": 1.0,
        "totally different": 1.0,
        "change style": 0.9,
    }

    for keyword, weight in density_keywords.items():
        if keyword in instruction_lower:
            change_score = max(change_score, weight)

    # Detect numerical density targets (e.g., "2 notes per second", "density 1.5")
    density_pattern = r"(?:density|notes?[\s/]+(?:per|\/)\s*(?:sec|second))[\s:]*(\d+(?:\.\d+)?)"
    match = re.search(density_pattern, instruction_lower)
    if match:
        target_density = float(match.group(1))
        # Calculate relative change
        if current_density > 0:
            density_change = abs(target_density - current_density) / current_density
            change_score = max(change_score, min(density_change, 1.0))

    # Detect pitch range changes
    if any(word in instruction_lower for word in ["octave", "higher", "lower", "transpose"]):
        change_score = max(change_score, 0.4)

    # Detect style/genre changes
    if any(word in instruction_lower for word in ["jazz", "classical", "rock", "blues", "style"]):
        change_score = max(change_score, 0.7)

    return change_score


def _build_refinement_instruction(
    base_instruction: str,
    track,  # Track object from gpt_conductor
    mode: str
) -> str:
    """Build instruction with style preservation constraints.

    Args:
        base_instruction: User's modification request
        track: Existing track to refine
        mode: Refinement mode (refinement, variation, or rewrite)

    Returns:
        Enhanced instruction with appropriate preservation hints
    """
    if mode == "rewrite":
        return base_instruction  # No preservation needed

    # Extract style descriptors from track features
    features = track.features

    # Density characterization
    if features.note_density < 1.0:
        density = "sparse"
    elif features.note_density < 3.0:
        density = "moderate"
    else:
        density = "dense"

    # Register characterization
    pitch_mid = (features.pitch_range[0] + features.pitch_range[1]) / 2
    register = "low" if pitch_mid < 60 else "mid" if pitch_mid < 72 else "high"

    if mode == "refinement":
        # Strong preservation: maintain overall character
        template = (
            f"PRESERVE: {density} texture ({features.note_density:.1f} notes/sec), "
            f"{register} register (pitch range {features.pitch_range[0]}-{features.pitch_range[1]})\n"
            f"MODIFY: {base_instruction}\n"
            f"Keep overall musical character and style similar, only make the requested adjustments."
        )
    elif mode == "variation":
        # Moderate preservation: allow more freedom
        template = (
            f"REFERENCE STYLE: {density} texture, {register} register\n"
            f"VARIATION REQUEST: {base_instruction}\n"
            f"You may deviate from the reference style to achieve the requested variation."
        )
    else:
        template = base_instruction

    return template


def _auto_select_reference(
    composition_id: str,
    new_role: str,
    tm: TrackManager,
) -> Optional[object]:
    """Auto-select the best reference track for a new track based on role.

    Role complementarity rules:
    - bass -> reference melody track (follow harmonic progression)
    - harmony -> reference melody track (complement melody)
    - rhythm -> reference any track (sync tempo)
    - melody -> reference harmony or bass track (fit with existing)

    Args:
        composition_id: Composition ID
        new_role: Role of the new track being created
        tm: TrackManager instance

    Returns:
        Best reference Track, or None if no tracks exist
    """
    state = tm.get_state(composition_id)
    if not state or not state.tracks:
        return None

    # Priority order for reference by new role
    role_priority = {
        "bass": ["melody", "harmony", "rhythm"],
        "harmony": ["melody", "bass", "rhythm"],
        "rhythm": ["melody", "bass", "harmony"],
        "melody": ["harmony", "bass", "rhythm"],
    }

    priorities = role_priority.get(new_role, ["melody", "harmony", "bass", "rhythm"])

    for target_role in priorities:
        for track in state.tracks:
            if track.role == target_role and track.metadata.get("midi_token_ids"):
                return track

    # Fallback: return first track with tokens
    for track in state.tracks:
        if track.metadata.get("midi_token_ids"):
            return track

    return None


def _resolve_target_duration_seconds(
    composition_state: Optional[CompositionState],
) -> float:
    """Resolve composition target duration, with fallback for legacy sessions."""
    if not composition_state:
        return 0.0

    target = float(composition_state.metadata.get("target_duration_seconds", 0.0) or 0.0)
    if target > 0:
        return target

    for track in composition_state.tracks:
        duration = float(getattr(track.features, "duration_seconds", 0.0) or 0.0)
        if duration > 0:
            return duration

    return 0.0


def _estimate_bar_count(duration_seconds: float, tempo_bpm: float) -> int:
    """Estimate total bars from duration and tempo (assume 4/4)."""
    if duration_seconds <= 0:
        return 8
    tempo = tempo_bpm if tempo_bpm > 0 else 90.0
    seconds_per_bar = (60.0 / tempo) * 4.0
    if seconds_per_bar <= 0:
        return 8
    bars = int(round(duration_seconds / seconds_per_bar))
    return max(4, bars)


def _default_form_from_bars(bar_total: int) -> str:
    """Generate a simple section form string from total bars."""
    if bar_total <= 8:
        return f"A{bar_total}"
    if bar_total <= 16:
        return "A8 A8"
    if bar_total <= 24:
        return "A8 B8 A8"
    return "A8 A8 B8 A8"


def _build_arrangement_contract(
    composition_state: Optional[CompositionState],
    reference_track: Optional[object] = None,
) -> dict:
    """Build a composition-level arrangement contract for all tracks."""
    if composition_state and composition_state.metadata:
        meta = composition_state.metadata
    else:
        meta = {}

    comp_key = meta.get("composition_key")
    key_name = "C major"
    if comp_key and isinstance(comp_key, (list, tuple)) and len(comp_key) == 2:
        key_name = f"{comp_key[0]} {comp_key[1]}"

    tempo = float(meta.get("composition_tempo", 0.0) or 0.0)
    if tempo <= 0:
        tempo = 90.0

    target_duration = float(meta.get("target_duration_seconds", 0.0) or 0.0)
    if target_duration <= 0 and composition_state:
        target_duration = _resolve_target_duration_seconds(composition_state)

    bar_total = _estimate_bar_count(target_duration, tempo)

    chord_source = None
    if reference_track and getattr(reference_track, "features", None):
        chord_source = reference_track.features.chord_progression
    elif composition_state:
        for t in composition_state.tracks:
            if t.features and t.features.chord_progression:
                chord_source = t.features.chord_progression
                break

    if chord_source:
        chord_map = " | ".join(chord_source[: min(8, len(chord_source))])
    else:
        chord_map = "Use diatonic progression matching the detected key"

    return {
        "key": key_name,
        "bpm": int(round(tempo)),
        "time_signature": "4/4",
        "bars": bar_total,
        "form": _default_form_from_bars(bar_total),
        "chord_map": chord_map,
        "energy_curve": "low -> medium -> high -> resolve",
    }


def _render_structured_instruction(
    *,
    instruction_spec: dict,
    fallback_instruction: str,
    instrument: str,
    role: str,
    arrangement_contract: dict,
    reference_track: Optional[object] = None,
) -> str:
    """Render structured instruction_spec into a deterministic musician prompt."""
    global_spec = instruction_spec.get("global_contract", {}) if isinstance(instruction_spec, dict) else {}
    track_spec = instruction_spec.get("track_role", {}) if isinstance(instruction_spec, dict) else {}
    relation_spec = instruction_spec.get("relation_to_reference", {}) if isinstance(instruction_spec, dict) else {}
    rhythm_spec = instruction_spec.get("rhythm_phrase_rules", {}) if isinstance(instruction_spec, dict) else {}
    output_rules = instruction_spec.get("output_rules", []) if isinstance(instruction_spec, dict) else []

    key_name = global_spec.get("key") or arrangement_contract["key"]
    bpm = int(global_spec.get("bpm") or arrangement_contract["bpm"])
    time_sig = global_spec.get("time_signature") or arrangement_contract["time_signature"]
    form = global_spec.get("form") or arrangement_contract["form"]
    bars = int(global_spec.get("bars") or arrangement_contract["bars"])
    chord_map = global_spec.get("chord_map") or arrangement_contract["chord_map"]
    energy_curve = global_spec.get("energy_curve") or arrangement_contract["energy_curve"]

    bar_start = int(track_spec.get("bar_start", 1))
    bar_end = int(track_spec.get("bar_end", bars))
    register_low = int(track_spec.get("register_low", 48))
    register_high = int(track_spec.get("register_high", 84))
    density_target = track_spec.get("density_target", "role-appropriate")
    function_in_mix = track_spec.get("function_in_mix", role)
    interaction_type = relation_spec.get("interaction_type", "complement")

    if reference_track:
        ref_id = getattr(reference_track, "id", "")
        ref_inst = getattr(reference_track, "instrument", "unknown")
        ref_desc = f"{ref_id} ({ref_inst})"
    else:
        ref_desc = relation_spec.get("reference_track_id", "none")

    phrase_len = rhythm_spec.get("phrase_length_bars", 4)
    anchor_beats = rhythm_spec.get("anchor_beats", "1 and 3")
    cadence_bars = rhythm_spec.get("cadence_bars", "end of each phrase")

    creative_intent = (
        instruction_spec.get("creative_intent")
        or fallback_instruction
        or f"Compose a {role} line that supports the arrangement."
    )

    rules = [
        "Single-instrument only",
        "Strictly follow key/chords/form",
        "Avoid register collision with reference when active",
    ]
    if isinstance(output_rules, list):
        for r in output_rules:
            if isinstance(r, str) and r.strip():
                rules.append(r.strip())

    return (
        f"[GLOBAL CONTRACT]\n"
        f"Key={key_name}, BPM={bpm}, TimeSig={time_sig}, Form={form}, Bars={bars}\n"
        f"Chord map: {chord_map}\n"
        f"Energy curve: {energy_curve}\n\n"
        f"[TRACK ROLE]\n"
        f"Instrument={instrument}, Role={role}, Bars={bar_start}-{bar_end}\n"
        f"Register={register_low}-{register_high}, Density={density_target} notes/sec\n"
        f"Function={function_in_mix}\n\n"
        f"[RELATION TO REFERENCE]\n"
        f"Reference={ref_desc}, Interaction={interaction_type}\n"
        f"Complement the reference; do not duplicate melodic contour or octave placement.\n\n"
        f"[RHYTHM AND PHRASE RULES]\n"
        f"Phrase length={phrase_len} bars, Anchors={anchor_beats}, Cadence={cadence_bars}\n\n"
        f"[CREATIVE INTENT]\n"
        f"{creative_intent}\n\n"
        f"[OUTPUT RULES]\n"
        f"{'; '.join(rules)}.\n"
    )


async def _generate_with_quality_gate(
    instruction: str,
    user_intent: str,
    musician: MIDILLMMusician,
    role: str = "melody",
    composition_key: tuple[str, str] | None = None,
    composition_tempo: float = 0.0,
    max_attempts: int = 3,
    reference_tokens: list[int] | None = None,
    reference_instrument: str | None = None,
    reference_features: dict | None = None,
    prefix_tokens: list[int] | None = None,
    prefix_ratio: float = 0.3,
    websocket: Optional[WebSocket] = None,
    target_instrument: str | None = None,
) -> MusicianGenerationResult:
    """Generate MIDI with automatic quality evaluation and retry.

    Uses the Evaluator to score each generation attempt. If the score is below
    the accept threshold, retries with enhanced instructions incorporating
    evaluation feedback.

    Args:
        instruction: Generation instruction for MIDI-LLM
        user_intent: Original user intent (for evaluation)
        musician: MIDI-LLM musician instance
        role: Track role (melody, harmony, bass, rhythm)
        composition_key: Composition key for coherence evaluation
        composition_tempo: Composition tempo for coherence evaluation
        max_attempts: Maximum generation attempts
        reference_tokens: Optional reference track tokens
        reference_instrument: Optional reference instrument name
        prefix_tokens: Optional prefix tokens for refinement
        prefix_ratio: Prefix ratio for refinement
        websocket: Optional WebSocket for debug messages

    Returns:
        Best MusicianGenerationResult from all attempts
    """
    evaluator = Evaluator(
        accept_threshold=0.65,
        reject_threshold=0.30,
        composition_key=composition_key,
        composition_tempo=composition_tempo,
    )

    # Initialize GPT critic for intelligent refinement feedback
    gpt_critic = None
    try:
        if settings.openai_api_key:
            gpt_critic = GPTCritic(api_key=settings.openai_api_key)
    except Exception as e:
        logger.warning(f"GPT critic unavailable, falling back to rule-based: {e}")

    planner = Planner()
    prompt_builder = PromptBuilder()
    current_plan = planner.plan_initial(user_intent)

    best_result = None
    best_score = -1.0
    current_instruction = instruction
    history: list = []
    prev_features = None  # Track features from failed attempts for adaptive constraints

    # Retrieve few-shot examples for the first attempt (RAG)
    example_context = ""
    try:
        # Extract key/tempo info for better matching
        key_str = f"{composition_key[0]} {composition_key[1]}" if composition_key else ""
        examples = example_db.query(
            role=role,
            instruction=instruction,
            key=key_str,
            tempo=composition_tempo,
        )
        if examples:
            example_context = example_db.format_examples_for_prompt(examples)
            logger.info(f"RAG: Retrieved {len(examples)} few-shot examples for {role}")
    except Exception as e:
        logger.warning(f"RAG example retrieval failed: {e}")

    for attempt in range(1, max_attempts + 1):
        # Generate (pass previous features for dynamic negative constraints on retries)
        if reference_tokens:
            result = await musician.generate_with_reference(
                instruction=current_instruction,
                reference_tokens=reference_tokens,
                reference_instrument=reference_instrument,
                reference_features=reference_features,
                role=role,
            )
        elif prefix_tokens:
            result = await musician.generate_with_prefix(
                instruction=current_instruction,
                prefix_tokens=prefix_tokens,
                prefix_ratio=prefix_ratio,
                role=role,
            )
        else:
            result = await musician.generate(
                current_instruction,
                role=role,
                previous_features=prev_features,
                example_context=example_context if attempt == 1 else "",
            )

        # Evaluate (apply instrument override BEFORE feature extraction
        # so quality gate sees the actual final output, not pre-override drums)
        try:
            midi_result = token_processor.tokens_to_midi(result.midi_token_ids)
            eval_pretty = midi_result.pretty_midi
            if target_instrument:
                _, eval_pretty = _apply_instrument_override(eval_pretty, target_instrument)
            features = extract_features(eval_pretty)
            evaluation = evaluator.evaluate(features, user_intent, history)

            if evaluation.score > best_score:
                best_score = evaluation.score
                best_result = result

            if websocket:
                await websocket.send_json({
                    "type": "debug",
                    "data": {
                        "message": (
                            f"[Quality Gate] Attempt {attempt}/{max_attempts}: "
                            f"score={evaluation.score:.2f}, verdict={evaluation.verdict}"
                        ),
                        "strengths": evaluation.strengths,
                        "weaknesses": evaluation.weaknesses,
                    },
                })

            if evaluation.verdict == "accept":
                logger.info(
                    f"Quality gate passed on attempt {attempt}: "
                    f"score={evaluation.score:.3f}"
                )
                # Save successful generation to example DB for future RAG retrieval
                try:
                    key_str = ""
                    if features.estimated_key:
                        key_str = f"{features.estimated_key[0]} {features.estimated_key[1]}"
                    example_db.add_example(
                        token_ids=result.midi_token_ids,
                        instruction=instruction,
                        role=role,
                        quality_score=evaluation.score,
                        note_density=features.note_density,
                        key=key_str,
                        tempo=features.estimated_tempo,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save example to DB: {e}")
                return result

            # Save features for adaptive constraints on next retry
            prev_features = features

            if attempt < max_attempts:
                # Try GPT critic first for intelligent refinement feedback
                critic_corrections = []
                if gpt_critic and evaluation.verdict != "accept":
                    try:
                        critique = await gpt_critic.critique(
                            features=features,
                            user_intent=user_intent,
                            role=role,
                            attempt_number=attempt,
                            composition_key=composition_key,
                            composition_tempo=composition_tempo,
                        )
                        critic_corrections = critique.get("corrections", [])

                        if websocket:
                            await websocket.send_json({
                                "type": "debug",
                                "data": {
                                    "message": (
                                        f"[GPT Critic] {critique.get('diagnosis', '')}"
                                    ),
                                    "corrections": critic_corrections,
                                    "critic_score": critique.get("quality_score", 0),
                                },
                            })
                    except Exception as e:
                        logger.warning(f"GPT critic failed on attempt {attempt}: {e}")

                # Build refined instruction from critic corrections + evaluator suggestions
                if critic_corrections:
                    # GPT critic provides more specific feedback
                    corrections_text = ". ".join(critic_corrections[:3])
                    current_instruction = f"{instruction}\nIMPROVEMENTS REQUIRED: {corrections_text}"
                    logger.info(
                        f"Quality gate retry {attempt}: score={evaluation.score:.3f}, "
                        f"critic corrections: {corrections_text[:100]}"
                    )
                elif evaluation.suggestions:
                    # Fallback to rule-based refinement
                    refined_plan = planner.plan_refinement(evaluation, current_plan, history)
                    refined_prompt = prompt_builder.build_refinement(
                        refined_plan,
                        ". ".join(evaluation.suggestions[:3]),
                    )
                    current_instruction = f"{instruction}\n{refined_prompt}"
                    current_plan = refined_plan
                    logger.info(
                        f"Quality gate retry {attempt}: score={evaluation.score:.3f}, "
                        f"verdict={evaluation.verdict}, refined_prompt={refined_prompt[:100]}"
                    )

        except Exception as e:
            logger.warning(f"Quality gate evaluation failed on attempt {attempt}: {e}")
            if best_result is None:
                best_result = result

    # Return best attempt
    logger.info(f"Quality gate: returning best of {max_attempts} attempts (score={best_score:.3f})")
    return best_result or result


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
        parameters: Action parameters (instrument, role, instruction, reference_track_id)
        websocket: Optional WebSocket for debug messages
    """
    instrument = _infer_instrument(
        parameters.get("instrument", ""), parameters.get("instruction", "")
    )
    role = parameters.get("role", "melody")
    instruction = parameters.get("instruction", "")
    instruction_spec = parameters.get("instruction_spec")
    reference_track_id = parameters.get("reference_track_id")  # Reference track
    volume = parameters.get("volume", 1.0)  # NEW: Track volume (0.0-1.0, default 1.0)

    # Get reference track tokens if specified, or auto-select best reference
    reference_tokens = None
    reference_track = None
    if reference_track_id:
        reference_track = track_manager.get_track(composition_id, reference_track_id)
    elif track_manager.get_state(composition_id).tracks:
        # Auto-select reference track based on role complementarity
        reference_track = _auto_select_reference(
            composition_id, role, track_manager
        )
        if reference_track:
            logger.info(f"Auto-selected reference track: {reference_track.id} ({reference_track.instrument})")

    if reference_track:
        reference_tokens = reference_track.metadata.get("midi_token_ids", [])
        if reference_tokens:
            logger.info(
                f"Using reference track {reference_track.id} "
                f"({reference_track.instrument}) with {len(reference_tokens)} tokens"
            )
        else:
            logger.info(
                f"Reference track {reference_track.id} found but no tokens available; "
                "using text-only reference guidance"
            )

    # Get composition-level key/tempo for quality evaluation
    comp_state = track_manager.get_state(composition_id)
    comp_key = comp_state.metadata.get("composition_key") if comp_state else None
    comp_tempo = comp_state.metadata.get("composition_tempo", 0.0) if comp_state else 0.0
    target_duration_seconds = _resolve_target_duration_seconds(comp_state)
    if (
        comp_state
        and target_duration_seconds > 0
        and float(comp_state.metadata.get("target_duration_seconds", 0.0) or 0.0) <= 0
    ):
        track_manager.update_metadata(composition_id, {
            "target_duration_seconds": round(target_duration_seconds, 2),
        })

    # Render structured prompt contract if provided by Conductor.
    if isinstance(instruction_spec, dict):
        arrangement_contract = _build_arrangement_contract(
            composition_state=comp_state,
            reference_track=reference_track,
        )
        instruction = _render_structured_instruction(
            instruction_spec=instruction_spec,
            fallback_instruction=instruction,
            instrument=instrument,
            role=role,
            arrangement_contract=arrangement_contract,
            reference_track=reference_track,
        )

    # Enforce key/tempo in the instruction for subsequent tracks
    if (not isinstance(instruction_spec, dict)) and (comp_key or comp_tempo > 0):
        key_tempo_suffix = []
        if comp_key:
            key_tempo_suffix.append(f"MUST be in {comp_key[0]} {comp_key[1]}")
        if comp_tempo > 0:
            key_tempo_suffix.append(f"tempo {comp_tempo:.0f} BPM")
        instruction = f"{instruction}. {', '.join(key_tempo_suffix)}."

    # Fallback: keep reference relation in prompt even when reference tokens are unavailable.
    if reference_track and not reference_tokens:
        ref_features = reference_track.features
        key_line = (
            f"Key={ref_features.estimated_key[0]} {ref_features.estimated_key[1]}. "
            if ref_features.estimated_key else ""
        )
        tempo_line = (
            f"Tempo={ref_features.estimated_tempo:.0f} BPM. "
            if ref_features.estimated_tempo > 0 else ""
        )
        chord_line = (
            f"Chords={' | '.join(ref_features.chord_progression[:8])}. "
            if ref_features.chord_progression else ""
        )
        instruction = (
            "[REFERENCE TRACK CONTEXT]\n"
            f"Reference={reference_track.id} ({reference_track.instrument}, {reference_track.role}). "
            f"{key_line}{tempo_line}{chord_line}"
            "Match harmonic rhythm and phrase boundaries; complement and avoid duplicate contour.\n\n"
            f"{instruction}"
        )

    # Send debug message with the final rendered instruction.
    if websocket:
        debug_data = {
            "message": f"[MIDI-LLM] Generating {instrument} ({role})",
            "prompt": instruction,
        }
        if reference_track:
            debug_data["reference_track"] = f"{reference_track.id} ({reference_track.instrument})"
            debug_data["reference_tokens"] = len(reference_tokens) if reference_tokens else 0
            debug_data["reference_mode"] = "tokens" if reference_tokens else "text_fallback"
        if isinstance(instruction_spec, dict):
            debug_data["prompt_mode"] = "structured_contract"

        await websocket.send_json({
            "type": "debug",
            "data": debug_data,
        })

    # Build reference features dict if reference track available
    reference_features = None
    if reference_track:
        reference_features = reference_track.features.to_dict()

    # Generate MIDI tokens with quality gate
    musician = MIDILLMMusician()
    try:
        result = await _generate_with_quality_gate(
            instruction=instruction,
            user_intent=instruction,
            musician=musician,
            role=role,
            composition_key=tuple(comp_key) if comp_key else None,
            composition_tempo=comp_tempo,
            reference_tokens=reference_tokens,
            reference_instrument=reference_track.instrument if reference_track else None,
            reference_features=reference_features,
            websocket=websocket,
            target_instrument=instrument,
        )
    finally:
        await musician.close()

    # Convert tokens to MIDI
    midi_result = token_processor.tokens_to_midi(result.midi_token_ids)
    midi_bytes, pretty = _apply_instrument_override(
        midi_result.pretty_midi, instrument
    )
    midi_bytes, pretty = _normalize_midi_start_time(pretty)
    if target_duration_seconds and target_duration_seconds > 0:
        midi_bytes, pretty = _trim_midi_to_duration(pretty, target_duration_seconds)

    # Synthesize audio
    track_id = f"track_{len(track_manager.get_state(composition_id).tracks) + 1}"
    synthesizer = get_audio_synthesizer()
    midi_path, audio_path = synthesizer.synthesize_track(
        track_id=track_id,
        midi_bytes=midi_bytes,
        output_dir=session_dir,
        target_duration_seconds=target_duration_seconds if target_duration_seconds > 0 else None,
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
            "midi_token_ids": result.midi_token_ids,  # Save tokens for future refinement
            "refinement_history": [],  # Track refinement chain
            "instruction_spec": instruction_spec if isinstance(instruction_spec, dict) else None,
            "volume": volume,  # NEW: Track volume for mixing (0.0-1.0)
        },
    )

    # Store composition-level key/tempo from the first track (or update if better confidence)
    comp_state = track_manager.get_state(composition_id)
    if comp_state and features.estimated_key:
        existing_key = comp_state.metadata.get("composition_key")
        existing_confidence = comp_state.metadata.get("composition_key_confidence", 0.0)
        if not existing_key or features.key_confidence > existing_confidence:
            track_manager.update_metadata(composition_id, {
                "composition_key": list(features.estimated_key),
                "composition_key_confidence": features.key_confidence,
            })
    if comp_state and features.estimated_tempo > 0:
        existing_tempo = comp_state.metadata.get("composition_tempo", 0.0)
        if existing_tempo == 0:
            track_manager.update_metadata(composition_id, {
                "composition_tempo": features.estimated_tempo,
            })
    # Set composition target duration from the first successful track.
    if comp_state:
        existing_target = comp_state.metadata.get("target_duration_seconds", 0.0)
        if existing_target <= 0 and features.duration_seconds > 0:
            track_manager.update_metadata(composition_id, {
                "target_duration_seconds": round(features.duration_seconds, 2),
            })


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
    instruction_spec = parameters.get("instruction_spec")
    existing = track_manager.get_track(composition_id, track_id)
    if not existing:
        return

    # Backup old tokens and instruction before regenerating (for refinement history)
    old_tokens = existing.metadata.get("midi_token_ids", [])
    old_instruction = existing.metadata.get("instruction", "")

    # NEW: Get volume (preserve existing if not specified)
    volume = parameters.get("volume", existing.metadata.get("volume", 1.0))

    instrument = _infer_instrument(
        parameters.get("instrument", ""), instruction or existing.instrument
    )
    role = parameters.get("role", existing.role)

    # Check refinement mode and preserve_style flag
    refinement_mode = parameters.get("refinement_mode", "refinement")  # Changed default
    preserve_style = parameters.get("preserve_style", True)  # NEW: Default to preserving style

    # Analyze requested feature changes to detect large modifications
    feature_change_magnitude = _analyze_feature_change_magnitude(instruction, existing)

    # Auto-disable style preservation for large changes
    if feature_change_magnitude > 0.5:  # >50% change
        preserve_style = False
        logger.info(f"Large feature change detected ({feature_change_magnitude:.1%}), disabling token prefix")

    # Get composition-level key/tempo for quality evaluation
    comp_state = track_manager.get_state(composition_id)
    comp_key = comp_state.metadata.get("composition_key") if comp_state else None
    comp_tempo = comp_state.metadata.get("composition_tempo", 0.0) if comp_state else 0.0
    target_duration_seconds = _resolve_target_duration_seconds(comp_state)
    if (
        comp_state
        and target_duration_seconds > 0
        and float(comp_state.metadata.get("target_duration_seconds", 0.0) or 0.0) <= 0
    ):
        track_manager.update_metadata(composition_id, {
            "target_duration_seconds": round(target_duration_seconds, 2),
        })

    # Render structured prompt contract if provided by Conductor.
    if isinstance(instruction_spec, dict):
        arrangement_contract = _build_arrangement_contract(
            composition_state=comp_state,
            reference_track=existing,
        )
        instruction = _render_structured_instruction(
            instruction_spec=instruction_spec,
            fallback_instruction=instruction,
            instrument=instrument,
            role=role,
            arrangement_contract=arrangement_contract,
            reference_track=existing,
        )

    # Enhance instruction based on refinement mode and preserve_style flag
    if existing and preserve_style:
        if refinement_mode == "refinement":
            instruction = _build_refinement_instruction(
                base_instruction=instruction,
                track=existing,
                mode="refinement"
            )
        elif refinement_mode == "variation":
            instruction = _build_refinement_instruction(
                base_instruction=instruction,
                track=existing,
                mode="variation"
            )
    # If preserve_style is False or mode is "rewrite", don't add preservation constraints

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

    # Send debug message with final prompt and refinement mode.
    if websocket:
        mode_label = {
            "full_regen": "Full Regeneration",
            "refinement": "Refinement (with token prefix)",
            "variation": "Variation (partial preservation)",
            "rewrite": "Complete Rewrite",
        }.get(refinement_mode, refinement_mode)

        await websocket.send_json({
            "type": "debug",
            "data": {
                "message": f"[MIDI-LLM] {mode_label}: {track_id} ({instrument})",
                "prompt": full_instruction,
                "refinement_mode": refinement_mode,
                "preserve_style": preserve_style,
                "feature_change": f"{feature_change_magnitude:.1%}",
                "prompt_mode": "structured_contract" if isinstance(instruction_spec, dict) else "legacy_instruction",
            },
        })

    # Calculate prefix ratio for style preservation
    prefix_ratio = 0.3
    prefix_tokens_for_gate = None
    if old_tokens and preserve_style:
        if feature_change_magnitude < 0.2:
            prefix_ratio = 0.6
        elif feature_change_magnitude < 0.4:
            prefix_ratio = 0.4
        else:
            prefix_ratio = 0.25
        prefix_tokens_for_gate = old_tokens
    else:
        logger.info(
            f"Full regeneration without token prefix "
            f"(preserve_style={preserve_style}, change={feature_change_magnitude:.1%})"
        )

    musician = MIDILLMMusician()
    try:
        result = await _generate_with_quality_gate(
            instruction=full_instruction,
            user_intent=full_instruction,
            musician=musician,
            role=role,
            composition_key=tuple(comp_key) if comp_key else None,
            composition_tempo=comp_tempo,
            prefix_tokens=prefix_tokens_for_gate,
            prefix_ratio=prefix_ratio,
            websocket=websocket,
            target_instrument=instrument,
        )
    finally:
        await musician.close()

    midi_result = token_processor.tokens_to_midi(result.midi_token_ids)
    midi_bytes, pretty = _apply_instrument_override(
        midi_result.pretty_midi, instrument
    )
    midi_bytes, pretty = _normalize_midi_start_time(pretty)
    if target_duration_seconds and target_duration_seconds > 0:
        midi_bytes, pretty = _trim_midi_to_duration(pretty, target_duration_seconds)

    synthesizer = get_audio_synthesizer()
    midi_path, audio_path = synthesizer.synthesize_track(
        track_id=track_id,
        midi_bytes=midi_bytes,
        output_dir=session_dir,
        target_duration_seconds=target_duration_seconds if target_duration_seconds > 0 else None,
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
            "instruction_spec": instruction_spec if isinstance(instruction_spec, dict) else existing.metadata.get("instruction_spec"),
            "volume": volume,  # NEW: Track volume for mixing
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
                "instruction_spec": instruction_spec,
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


async def _execute_adjust_volume(
    composition_id: str,
    parameters: dict,
    websocket: Optional[WebSocket] = None,
) -> None:
    """Execute a volume adjustment without regenerating the track.

    This is a lightweight operation that only updates the track's volume metadata
    and triggers a mix regeneration. The track's MIDI and audio files remain unchanged.

    Args:
        composition_id: Composition ID
        parameters: Action parameters (track_id, volume)
        websocket: Optional WebSocket for progress messages
    """
    track_id = parameters.get("track_id")
    new_volume = parameters.get("volume")

    if not track_id or new_volume is None:
        logger.warning("adjust_volume missing track_id or volume parameter")
        return

    # Validate volume range
    if not (0.0 <= new_volume <= 1.0):
        logger.warning(f"Invalid volume {new_volume}, must be 0.0-1.0")
        new_volume = max(0.0, min(1.0, new_volume))  # Clamp to valid range

    existing = track_manager.get_track(composition_id, track_id)
    if not existing:
        logger.warning(f"Track {track_id} not found for volume adjustment")
        return

    old_volume = existing.metadata.get("volume", 1.0)

    # Send debug message
    if websocket:
        await websocket.send_json({
            "type": "debug",
            "data": {
                "message": f"[Volume] Adjusting {track_id} volume: {old_volume:.2f} → {new_volume:.2f}",
                "old_volume": old_volume,
                "new_volume": new_volume,
            },
        })

    logger.info(
        f"Adjusting volume for {track_id}: {old_volume:.2f} → {new_volume:.2f} "
        f"(no regeneration, mix will be updated)"
    )

    # Update only the volume in metadata (no MIDI/audio regeneration)
    updated_metadata = existing.metadata.copy()
    updated_metadata["volume"] = new_volume

    track_manager.update_track(
        composition_id=composition_id,
        track_id=track_id,
        instrument=existing.instrument,
        role=existing.role,
        midi_path=existing.midi_path,
        audio_path=existing.audio_path,
        features=existing.features,
        metadata=updated_metadata,
    )

    # NOTE: The mix will be automatically regenerated when requested
    # because the track metadata has changed (volume is different)
    logger.info(f"Volume adjustment complete for {track_id}, mix will regenerate on next request")


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

    # For non-drum instruments: smart merge based on note distribution
    if target is not None:
        non_drum_instruments = [inst for inst in pretty.instruments if not inst.is_drum]
        drum_instruments = [inst for inst in pretty.instruments if inst.is_drum]

        if not non_drum_instruments and drum_instruments:
            # ALL notes are on drum channel — remap to target instrument.
            # This happens when MIDI-LLM generates drum-instrument tokens
            # (instrument 128 in AMT space) instead of the requested instrument.
            # The pitch values are still valid MIDI pitches, just on the wrong channel.
            logger.warning(
                f"All generated notes are on drum channel, remapping "
                f"{sum(len(d.notes) for d in drum_instruments)} notes to "
                f"{instrument_name} (program {target})"
            )
            merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)
            for drum_inst in drum_instruments:
                for note in drum_inst.notes:
                    merged.notes.append(note)
            merged.notes.sort(key=lambda n: n.start)
        elif not non_drum_instruments:
            merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)
        elif len(non_drum_instruments) == 1:
            merged = non_drum_instruments[0]
            merged.program = target
            merged.name = instrument_name
        else:
            # Multiple tracks - check if one dominates
            total_notes = sum(len(inst.notes) for inst in non_drum_instruments)
            if total_notes == 0:
                merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)
            else:
                # Find instrument with most notes
                main_inst = max(non_drum_instruments, key=lambda i: len(i.notes))
                main_ratio = len(main_inst.notes) / total_notes

                if main_ratio > 0.80:
                    # Main instrument dominates (>80%): just reassign program, drop others
                    merged = main_inst
                    merged.program = target
                    merged.name = instrument_name
                    logger.info(
                        f"Instrument override: main instrument has {main_ratio:.0%} of notes, "
                        f"dropping {len(non_drum_instruments) - 1} minor tracks"
                    )
                else:
                    # Merge all tracks
                    if main_ratio < 0.50:
                        logger.warning(
                            f"MIDI-LLM generated multi-instrument despite single-instrument instruction "
                            f"(main has only {main_ratio:.0%} of notes)"
                        )
                    merged = pretty_midi.Instrument(program=target, is_drum=False, name=instrument_name)

                    all_notes = []
                    for inst in non_drum_instruments:
                        all_notes.extend(inst.notes)

                    all_notes.sort(key=lambda n: n.start)

                    # Deduplicate
                    unique_notes = []
                    for note in all_notes:
                        is_duplicate = False
                        for existing in unique_notes:
                            if (note.pitch == existing.pitch and
                                abs(note.start - existing.start) < 0.05):
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


def _trim_midi_to_duration(
    pretty: "pretty_midi.PrettyMIDI",
    target_duration_seconds: float,
) -> tuple[bytes, "pretty_midi.PrettyMIDI"]:
    """Trim MIDI events that exceed the target duration.

    This keeps track lengths consistent across a composition once a target
    duration is established.
    """
    if target_duration_seconds <= 0:
        buf = io.BytesIO()
        pretty.write(buf)
        return buf.getvalue(), pretty

    min_note_len = 0.02
    for inst in pretty.instruments:
        kept_notes = []
        for note in inst.notes:
            if note.start >= target_duration_seconds:
                continue
            note.end = min(note.end, target_duration_seconds)
            if note.end - note.start >= min_note_len:
                kept_notes.append(note)
        inst.notes = kept_notes
        inst.control_changes = [
            cc for cc in inst.control_changes if cc.time <= target_duration_seconds
        ]
        inst.pitch_bends = [
            pb for pb in inst.pitch_bends if pb.time <= target_duration_seconds
        ]

    buf = io.BytesIO()
    pretty.write(buf)
    return buf.getvalue(), pretty


def _normalize_midi_start_time(
    pretty: "pretty_midi.PrettyMIDI",
    keep_preroll_seconds: float = 0.05,
    min_shift_threshold_seconds: float = 0.20,
) -> tuple[bytes, "pretty_midi.PrettyMIDI"]:
    """Shift MIDI events left if the first note starts too late.

    This avoids tracks that are silent for the first half and only start later.
    """
    note_starts = [
        n.start
        for inst in pretty.instruments
        for n in inst.notes
    ]
    if not note_starts:
        buf = io.BytesIO()
        pretty.write(buf)
        return buf.getvalue(), pretty

    first_start = min(note_starts)
    if first_start < min_shift_threshold_seconds:
        buf = io.BytesIO()
        pretty.write(buf)
        return buf.getvalue(), pretty

    shift = max(0.0, first_start - max(0.0, keep_preroll_seconds))

    for inst in pretty.instruments:
        for note in inst.notes:
            note.start = max(0.0, note.start - shift)
            note.end = max(note.start + 0.01, note.end - shift)
        for cc in inst.control_changes:
            cc.time = max(0.0, cc.time - shift)
        for pb in inst.pitch_bends:
            pb.time = max(0.0, pb.time - shift)

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

    # Extract track volumes and roles from metadata
    track_volumes = {}
    track_roles = {}
    for track in composition_state.tracks:
        midi_filename = Path(track.midi_path).name
        track_volumes[midi_filename] = track.metadata.get("volume", 1.0)
        track_roles[midi_filename] = track.role or "melody"

    synthesizer = get_audio_synthesizer()
    target_duration_seconds = _resolve_target_duration_seconds(composition_state)
    if (
        target_duration_seconds > 0
        and float(composition_state.metadata.get("target_duration_seconds", 0.0) or 0.0) <= 0
    ):
        track_manager.update_metadata(composition_id, {
            "target_duration_seconds": round(target_duration_seconds, 2),
        })
    try:
        combined_midi, mixed_audio = synthesizer.synthesize_mix(
            composition_id=composition_id,
            track_midi_paths=track_midi_paths,
            output_dir=session_dir,
            track_volumes=track_volumes,
            track_roles=track_roles,
            target_duration_seconds=target_duration_seconds if target_duration_seconds > 0 else None,
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

            # Build conversation history from session
            session = track_manager.get_session(composition_id)
            conversation_history = None
            if session and session.user_messages:
                conversation_history = [
                    {"user": u, "conductor": c}
                    for u, c in zip(session.user_messages, session.conductor_responses)
                ]

            conductor_instance = get_conductor()
            conductor_response = await conductor_instance.plan_action(
                user_message=user_message,
                composition_state=composition_state,
                conversation_history=conversation_history,
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

                elif action.type == "adjust_volume":
                    await _execute_adjust_volume(
                        composition_id=composition_id,
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
