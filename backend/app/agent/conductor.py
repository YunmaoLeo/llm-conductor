"""Conductor Agent: the brain of the orchestration system."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Optional

from app.agent.evaluator import Evaluator
from app.agent.memory import EvaluationResult, GenerationRecord, Memory
from app.agent.planner import Planner
from app.config import settings
from app.core.feature_extractor import FeatureExtractor, MusicFeatures
from app.core.ollama_client import OllamaClient
from app.core.prompt_builder import GenerationPlan, PromptBuilder
from app.core.token_processor import ProcessedGeneration, TokenProcessor


@dataclass
class AgentLogEntry:
    """A single entry in the agent's reasoning log."""

    iteration: int
    action: str  # "plan", "generate", "evaluate", "decide"
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class CompositionResult:
    """Final result of a composition session."""

    success: bool
    generation_id: str = ""
    text_description: str = ""
    midi_path: Optional[str] = None
    audio_path: Optional[str] = None
    features: Optional[dict] = None
    iterations: int = 0
    agent_log: list[dict] = field(default_factory=list)
    error: str = ""


class Conductor:
    """Orchestrates the generate-evaluate-refine loop.

    The Conductor interprets user intent, plans generations, evaluates
    results, and iteratively refines until an acceptable output is
    produced or max iterations are reached.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        token_processor: Optional[TokenProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        evaluator: Optional[Evaluator] = None,
        planner: Optional[Planner] = None,
        max_iterations: int = 0,
    ):
        self.client = ollama_client or OllamaClient()
        self.processor = token_processor or TokenProcessor()
        self.extractor = feature_extractor or FeatureExtractor()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.evaluator = evaluator or Evaluator()
        self.planner = planner or Planner()
        self.max_iterations = max_iterations or settings.max_iterations
        self.memory = Memory()
        self._log: list[AgentLogEntry] = []

    def _log_entry(self, iteration: int, action: str, message: str, **data):
        entry = AgentLogEntry(iteration=iteration, action=action, message=message, data=data)
        self._log.append(entry)

    async def compose(self, user_intent: str) -> CompositionResult:
        """Main entry point: run the full composition loop.

        1. Plan initial generation from user intent
        2. Generate MIDI tokens via Ollama
        3. Process tokens into MIDI
        4. Extract features
        5. Evaluate against user intent
        6. Decide: accept / refine / reject
        7. Repeat if needed (up to max_iterations)

        Args:
            user_intent: Natural language description of desired music.

        Returns:
            CompositionResult with the best generation and agent log.
        """
        self.memory.clear()
        self._log.clear()
        self.memory.user_intent = user_intent

        self._log_entry(0, "start", f"Received intent: {user_intent}")

        # Step 1: Initial plan
        plan = self.planner.plan_initial(user_intent)
        self._log_entry(0, "plan", "Created initial generation plan", plan=asdict(plan))

        for iteration in range(1, self.max_iterations + 1):
            self._log_entry(iteration, "generate", f"Starting generation (iteration {iteration}/{self.max_iterations})")

            # Step 2: Build prompt
            if iteration == 1:
                prompt = self.prompt_builder.build(plan)
            else:
                # Include evaluation feedback in the prompt for refinements
                latest = self.memory.get_latest()
                if latest and latest.evaluation.get("suggestions"):
                    feedback = ". ".join(latest.evaluation["suggestions"])
                    prompt = self.prompt_builder.build_refinement(plan, feedback)
                else:
                    prompt = self.prompt_builder.build(plan)

            self._log_entry(iteration, "prompt", f"Prompt: {prompt}")

            # Step 3: Generate
            try:
                gen_result = await self.client.generate(prompt=prompt)
            except Exception as e:
                self._log_entry(iteration, "error", f"Generation failed: {e}")
                continue

            if not gen_result.midi_token_ids:
                self._log_entry(iteration, "error", "No MIDI tokens in response")
                continue

            self._log_entry(
                iteration,
                "tokens",
                f"Generated {gen_result.token_count} tokens in {gen_result.generation_time:.1f}s",
                text_description=gen_result.text_description,
            )

            # Step 4: Process tokens
            processed = self.processor.process(gen_result.midi_token_ids)
            if not processed.is_valid:
                self._log_entry(
                    iteration, "error", f"Token processing failed: {processed.validation_message}"
                )
                continue

            self._log_entry(
                iteration,
                "process",
                f"Processed: {processed.num_notes} notes, {processed.duration_seconds:.1f}s, instruments: {processed.instruments_used}",
            )

            # Step 5: Extract features
            features = self.extractor.extract(processed)
            self._log_entry(
                iteration,
                "features",
                f"Density: {features.note_density}, pitch range: {features.pitch_range}",
            )

            # Step 6: Evaluate
            evaluation = self.evaluator.evaluate(
                features=features,
                user_intent=user_intent,
                history=self.memory.get_history(),
            )

            self._log_entry(
                iteration,
                "evaluate",
                f"Score: {evaluation.score:.3f}, verdict: {evaluation.verdict}",
                strengths=evaluation.strengths,
                weaknesses=evaluation.weaknesses,
                suggestions=evaluation.suggestions,
            )

            # Step 7: Record
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
            self.memory.add_record(record)

            # Step 8: Decide
            if evaluation.verdict == "accept":
                self._log_entry(iteration, "decide", "Accepted! Generation meets criteria.")
                return self._build_result(processed, gen_result.text_description, features, True)

            elif evaluation.verdict == "refine":
                self._log_entry(iteration, "decide", "Refining based on evaluation feedback.")
                plan = self.planner.plan_refinement(
                    evaluation, plan, self.memory.get_history()
                )

            else:  # reject
                self._log_entry(iteration, "decide", "Rejected. Starting fresh.")
                plan = self.planner.plan_initial(user_intent)

        # Max iterations reached — return best attempt
        self._log_entry(
            self.max_iterations,
            "fallback",
            "Max iterations reached. Returning best attempt.",
        )
        best = self.memory.get_best_record()
        if best:
            # Re-process best record to get the ProcessedGeneration
            # (We need the midi_path from it)
            return CompositionResult(
                success=True,
                generation_id=best.id,
                text_description=best.text_description,
                midi_path=f"/api/outputs/{best.id}/midi",
                features=best.features,
                iterations=self.memory.iteration_count,
                agent_log=[asdict(e) for e in self._log],
            )

        return CompositionResult(
            success=False,
            error="All generation attempts failed",
            iterations=self.memory.iteration_count,
            agent_log=[asdict(e) for e in self._log],
        )

    def _build_result(
        self,
        processed: ProcessedGeneration,
        text_description: str,
        features: MusicFeatures,
        success: bool,
    ) -> CompositionResult:
        return CompositionResult(
            success=success,
            generation_id=processed.id,
            text_description=text_description,
            midi_path=f"/api/outputs/{processed.id}/midi",
            features=features.to_dict(),
            iterations=self.memory.iteration_count,
            agent_log=[asdict(e) for e in self._log],
        )

    async def close(self):
        await self.client.close()
