"""Action planning for the Conductor Agent."""

from app.agent.memory import EvaluationResult, GenerationRecord
from app.core.prompt_builder import GenerationPlan


class Planner:
    """Decides what to generate based on user intent and evaluation feedback."""

    def plan_initial(self, user_intent: str) -> GenerationPlan:
        """Create the first generation plan from raw user intent.

        Passes through the user's text directly. The model is good at
        interpreting natural language descriptions.

        Args:
            user_intent: The user's original request.

        Returns:
            A GenerationPlan for the first generation attempt.
        """
        plan = GenerationPlan(raw_user_text=user_intent)
        self._infer_attributes(plan, user_intent)
        return plan

    def plan_refinement(
        self,
        evaluation: EvaluationResult,
        previous_plan: GenerationPlan,
        history: list[GenerationRecord],
    ) -> GenerationPlan:
        """Create a refined plan based on evaluation feedback.

        Incorporates suggestions from the evaluator to improve the next attempt.

        Args:
            evaluation: Evaluation of the previous generation.
            previous_plan: The plan that was used for the previous generation.
            history: Full generation history.

        Returns:
            A refined GenerationPlan.
        """
        # Start from the previous plan
        plan = GenerationPlan(
            raw_user_text=previous_plan.raw_user_text,
            style=previous_plan.style,
            instruments=previous_plan.instruments,
            energy=previous_plan.energy,
            mood=previous_plan.mood,
            tempo=previous_plan.tempo,
        )

        # Build constraints from evaluation suggestions
        constraints = list(previous_plan.constraints or [])
        for suggestion in evaluation.suggestions:
            constraints.append(suggestion)
        plan.constraints = constraints

        return plan

    def _infer_attributes(self, plan: GenerationPlan, intent: str):
        """Infer plan attributes from natural language intent."""
        intent_lower = intent.lower()

        # Infer energy
        if any(w in intent_lower for w in ["energetic", "fast", "upbeat", "lively", "powerful"]):
            plan.energy = "high"
        elif any(w in intent_lower for w in ["gentle", "calm", "soft", "peaceful", "relaxing"]):
            plan.energy = "low"

        # Infer tempo
        if any(w in intent_lower for w in ["fast", "quick", "rapid", "allegro"]):
            plan.tempo = "fast"
        elif any(w in intent_lower for w in ["slow", "adagio", "largo"]):
            plan.tempo = "slow"

        # Infer mood
        mood_keywords = {
            "happy": ["happy", "cheerful", "joyful", "bright"],
            "sad": ["sad", "melancholy", "somber", "dark"],
            "dramatic": ["dramatic", "intense", "epic", "powerful"],
            "peaceful": ["peaceful", "serene", "tranquil", "calm"],
        }
        for mood, keywords in mood_keywords.items():
            if any(w in intent_lower for w in keywords):
                plan.mood = mood
                break

        # Infer style
        style_keywords = {
            "jazz": ["jazz", "swing", "bebop"],
            "classical": ["classical", "baroque", "romantic period"],
            "rock": ["rock", "metal", "punk"],
            "pop": ["pop", "contemporary"],
            "electronic": ["electronic", "edm", "techno", "synth"],
        }
        for style, keywords in style_keywords.items():
            if any(w in intent_lower for w in keywords):
                plan.style = style
                break
