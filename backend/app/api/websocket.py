"""WebSocket endpoint for streaming composition progress."""

import json
from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.conductor import Conductor
from app.agent.evaluator import Evaluator
from app.agent.memory import Memory, GenerationRecord
from app.agent.planner import Planner
from app.config import settings
from app.core.feature_extractor import FeatureExtractor
from app.core.ollama_client import OllamaClient
from app.core.prompt_builder import GenerationPlan, PromptBuilder
from app.core.token_processor import TokenProcessor

ws_router = APIRouter()


async def send_json(ws: WebSocket, data: dict):
    await ws.send_text(json.dumps(data, default=str))


@ws_router.websocket("/ws/compose")
async def ws_compose(websocket: WebSocket):
    """WebSocket endpoint for real-time composition progress.

    Client sends: {"intent": "A cheerful piano melody"}
    Server streams back progress messages with types:
      - status: general progress update
      - plan: generation plan created
      - tokens: token generation complete
      - evaluation: evaluation result
      - complete: final result with MIDI URL
      - error: error occurred
    """
    await websocket.accept()

    try:
        # Wait for user intent
        data = await websocket.receive_text()
        request = json.loads(data)
        intent = request.get("intent", "")

        if not intent:
            await send_json(websocket, {"type": "error", "message": "No intent provided"})
            return

        await send_json(websocket, {"type": "status", "message": "Starting composition..."})

        # Initialize components
        client = OllamaClient()
        processor = TokenProcessor()
        extractor = FeatureExtractor()
        prompt_builder = PromptBuilder()
        evaluator = Evaluator()
        planner = Planner()
        memory = Memory()
        memory.user_intent = intent

        try:
            # Initial plan
            plan = planner.plan_initial(intent)
            await send_json(websocket, {
                "type": "plan",
                "message": "Created generation plan",
                "plan": asdict(plan),
            })

            max_iterations = settings.max_iterations
            best_result = None
            best_score = -1.0

            for iteration in range(1, max_iterations + 1):
                await send_json(websocket, {
                    "type": "status",
                    "message": f"Generating music (iteration {iteration}/{max_iterations})...",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                })

                # Build prompt
                if iteration == 1:
                    prompt = prompt_builder.build(plan)
                else:
                    latest = memory.get_latest()
                    if latest and latest.evaluation.get("suggestions"):
                        feedback = ". ".join(latest.evaluation["suggestions"])
                        prompt = prompt_builder.build_refinement(plan, feedback)
                    else:
                        prompt = prompt_builder.build(plan)

                # Generate
                try:
                    gen_result = await client.generate(prompt=prompt)
                except Exception as e:
                    await send_json(websocket, {
                        "type": "error",
                        "message": f"Generation failed: {e}",
                        "iteration": iteration,
                    })
                    continue

                if not gen_result.midi_token_ids:
                    await send_json(websocket, {
                        "type": "error",
                        "message": "No MIDI tokens generated",
                        "iteration": iteration,
                    })
                    continue

                await send_json(websocket, {
                    "type": "tokens",
                    "message": f"Generated {gen_result.token_count} tokens",
                    "token_count": gen_result.token_count,
                    "generation_time": round(gen_result.generation_time, 1),
                    "text_description": gen_result.text_description,
                    "iteration": iteration,
                })

                # Process
                processed = processor.process(gen_result.midi_token_ids)
                if not processed.is_valid:
                    await send_json(websocket, {
                        "type": "error",
                        "message": f"Processing failed: {processed.validation_message}",
                        "iteration": iteration,
                    })
                    continue

                # Extract features
                features = extractor.extract(processed)

                # Evaluate
                evaluation = evaluator.evaluate(
                    features=features,
                    user_intent=intent,
                    history=memory.get_history(),
                )

                await send_json(websocket, {
                    "type": "evaluation",
                    "score": evaluation.score,
                    "verdict": evaluation.verdict,
                    "strengths": evaluation.strengths,
                    "weaknesses": evaluation.weaknesses,
                    "suggestions": evaluation.suggestions,
                    "iteration": iteration,
                })

                # Record
                record = GenerationRecord(
                    id=processed.id,
                    iteration=iteration,
                    timestamp=datetime.now().isoformat(),
                    prompt=prompt,
                    plan=asdict(plan),
                    token_count=gen_result.token_count,
                    num_notes=processed.num_notes,
                    duration_seconds=processed.duration_seconds,
                    features=features.to_dict(),
                    evaluation=asdict(evaluation),
                    action_taken=evaluation.verdict,
                    text_description=gen_result.text_description,
                )
                memory.add_record(record)

                # Track best
                if evaluation.score > best_score:
                    best_score = evaluation.score
                    best_result = {
                        "generation_id": processed.id,
                        "text_description": gen_result.text_description,
                        "midi_url": f"/api/outputs/{processed.id}/midi",
                        "features": features.to_dict(),
                        "score": evaluation.score,
                    }

                # Decide
                if evaluation.verdict == "accept":
                    await send_json(websocket, {
                        "type": "complete",
                        "message": "Composition accepted!",
                        "success": True,
                        "generation_id": processed.id,
                        "text_description": gen_result.text_description,
                        "midi_url": f"/api/outputs/{processed.id}/midi",
                        "features": features.to_dict(),
                        "iterations": iteration,
                    })
                    return

                elif evaluation.verdict == "refine":
                    plan = planner.plan_refinement(
                        evaluation, plan, memory.get_history()
                    )
                    await send_json(websocket, {
                        "type": "status",
                        "message": "Refining based on feedback...",
                        "iteration": iteration,
                    })
                else:
                    plan = planner.plan_initial(intent)
                    await send_json(websocket, {
                        "type": "status",
                        "message": "Rejected. Trying a fresh approach...",
                        "iteration": iteration,
                    })

            # Max iterations reached — return best
            if best_result:
                await send_json(websocket, {
                    "type": "complete",
                    "message": "Max iterations reached. Returning best result.",
                    "success": True,
                    **best_result,
                    "iterations": max_iterations,
                })
            else:
                await send_json(websocket, {
                    "type": "complete",
                    "message": "All attempts failed.",
                    "success": False,
                    "iterations": max_iterations,
                })

        finally:
            await client.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send_json(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
