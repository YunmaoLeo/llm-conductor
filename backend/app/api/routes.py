"""API route definitions."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from pydantic import BaseModel

from app.agent.conductor import Conductor
from app.core.feature_extractor import FeatureExtractor
from app.core.ollama_client import OllamaClient
from app.core.token_processor import TokenProcessor

router = APIRouter(prefix="/api")


@router.post("/test-generate")
async def test_generate(prompt: str = "A simple piano melody"):
    """Test endpoint: generate MIDI tokens from a text prompt.

    Returns token count and sample token IDs for verification.
    """
    client = OllamaClient()
    try:
        result = await client.generate(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        await client.close()

    return {
        "text_description": result.text_description,
        "token_count": result.token_count,
        "token_ids_sample": result.midi_token_ids[:30],
        "generation_time": round(result.generation_time, 2),
        "done_reason": result.done_reason,
        "valid": (
            result.token_count > 0
            and all(0 <= t <= 55025 for t in result.midi_token_ids)
        ),
    }


@router.post("/test-process")
async def test_process(prompt: str = "A simple piano melody"):
    """Test endpoint: generate tokens and convert to MIDI.

    Returns MIDI file metadata and the file as downloadable content.
    """
    client = OllamaClient()
    try:
        gen_result = await client.generate(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        await client.close()

    if not gen_result.midi_token_ids:
        raise HTTPException(status_code=500, detail="No MIDI tokens generated")

    processor = TokenProcessor()
    processed = processor.process(gen_result.midi_token_ids)

    if not processed.is_valid:
        raise HTTPException(
            status_code=500, detail=f"Token processing failed: {processed.validation_message}"
        )

    return {
        "id": processed.id,
        "num_notes": processed.num_notes,
        "duration_seconds": round(processed.duration_seconds, 2),
        "instruments_used": processed.instruments_used,
        "is_valid": processed.is_valid,
        "midi_path": processed.midi_path,
        "token_count": len(processed.token_ids),
    }


@router.get("/outputs/{gen_id}/midi")
async def download_midi(gen_id: str):
    """Download a generated MIDI file."""
    import os

    from app.config import settings

    midi_path = os.path.join(settings.output_dir, f"{gen_id}.mid")
    if not os.path.exists(midi_path):
        raise HTTPException(status_code=404, detail="MIDI file not found")

    with open(midi_path, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="audio/midi",
        headers={"Content-Disposition": f"attachment; filename={gen_id}.mid"},
    )


@router.post("/test-features")
async def test_features(prompt: str = "A simple piano melody"):
    """Test endpoint: full pipeline from prompt to extracted features."""
    client = OllamaClient()
    try:
        gen_result = await client.generate(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        await client.close()

    if not gen_result.midi_token_ids:
        raise HTTPException(status_code=500, detail="No MIDI tokens generated")

    processor = TokenProcessor()
    processed = processor.process(gen_result.midi_token_ids)

    if not processed.is_valid:
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {processed.validation_message}"
        )

    extractor = FeatureExtractor()
    features = extractor.extract(processed)

    return {
        "generation_id": processed.id,
        "text_description": gen_result.text_description,
        "features": features.to_dict(),
    }


class ComposeRequest(BaseModel):
    intent: str


@router.post("/compose")
async def compose(request: ComposeRequest):
    """Full Conductor composition loop.

    Interprets user intent, generates MIDI iteratively with evaluation
    and refinement, returns the best result.
    """
    conductor = Conductor()
    try:
        result = await conductor.compose(request.intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Composition failed: {e}")
    finally:
        await conductor.close()

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "success": result.success,
        "generation_id": result.generation_id,
        "text_description": result.text_description,
        "midi_url": result.midi_path,
        "features": result.features,
        "iterations": result.iterations,
        "agent_log": result.agent_log,
    }
