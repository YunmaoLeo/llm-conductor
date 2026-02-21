"""MIDI Example Database: Stores successful generations for few-shot retrieval.

Implements a simple tag-based retrieval system. When a generation passes the quality
gate, its tokens and metadata are saved. On future generations, similar examples are
retrieved and prepended to the MIDI-LLM prompt as few-shot context.

Research basis: Jonason et al. (2023) "Retrieval Augmented Generation of Symbolic Music
with LLMs" — few-shot examples selected by tag similarity improve generation quality.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum tokens per example to include in prompt (to avoid blowing context window)
MAX_EXAMPLE_TOKENS = 90  # 30 token triples
# Maximum examples to include in prompt
MAX_EXAMPLES = 2


@dataclass
class MIDIExample:
    """A stored example of a successful MIDI generation."""

    tags: list[str]  # e.g., ["piano", "melody", "calm", "C_major", "slow"]
    token_ids: list[int]  # MIDI token IDs (truncated for storage efficiency)
    description: str  # Short description (e.g., "Calm piano melody in C major")
    role: str  # melody, harmony, bass, rhythm
    quality_score: float  # Score from evaluator at time of acceptance
    note_density: float = 0.0
    key: str = ""  # e.g., "C major"
    tempo: float = 0.0


class MIDIExampleDB:
    """Simple file-backed database of MIDI examples for few-shot retrieval."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or os.getenv("EXAMPLE_DB_PATH", "./outputs/example_db.json"))
        self.examples: list[MIDIExample] = []
        self._load()

    def _load(self):
        """Load examples from disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.examples = [MIDIExample(**ex) for ex in data]
                logger.info(f"Loaded {len(self.examples)} MIDI examples from {self.db_path}")
            except Exception as e:
                logger.warning(f"Failed to load example DB: {e}")
                self.examples = []

    def _save(self):
        """Persist examples to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(ex) for ex in self.examples]
            self.db_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save example DB: {e}")

    def add_example(
        self,
        token_ids: list[int],
        instruction: str,
        role: str,
        quality_score: float,
        note_density: float = 0.0,
        key: str = "",
        tempo: float = 0.0,
    ):
        """Add a successful generation as a retrievable example.

        Only stores a truncated token sequence (first MAX_EXAMPLE_TOKENS tokens)
        to keep storage and prompt size manageable.

        Args:
            token_ids: Full MIDI token sequence
            instruction: The instruction that produced this generation
            role: Track role
            quality_score: Evaluator score at acceptance
            note_density: Notes per second
            key: Detected key (e.g., "C major")
            tempo: Detected tempo in BPM
        """
        # Only store high-quality examples
        if quality_score < 0.60:
            return

        # Truncate tokens for storage (keep first N tokens, aligned to triples)
        truncated = token_ids[:MAX_EXAMPLE_TOKENS]
        remainder = len(truncated) % 3
        if remainder:
            truncated = truncated[:len(truncated) - remainder]

        if len(truncated) < 9:  # Need at least 3 triples
            return

        # Extract tags from instruction
        tags = self._extract_tags(instruction, role, key, tempo)

        # Build short description
        description = instruction[:80] if len(instruction) > 80 else instruction

        example = MIDIExample(
            tags=tags,
            token_ids=truncated,
            description=description,
            role=role,
            quality_score=quality_score,
            note_density=note_density,
            key=key,
            tempo=tempo,
        )

        self.examples.append(example)

        # Cap total examples to prevent unbounded growth
        if len(self.examples) > 200:
            # Keep the highest-quality examples
            self.examples.sort(key=lambda x: x.quality_score, reverse=True)
            self.examples = self.examples[:150]

        self._save()
        logger.info(f"Added example: tags={tags}, score={quality_score:.2f}, tokens={len(truncated)}")

    def query(
        self,
        role: str,
        instruction: str = "",
        key: str = "",
        tempo: float = 0.0,
        top_k: int = MAX_EXAMPLES,
    ) -> list[MIDIExample]:
        """Retrieve the most similar examples by tag matching.

        Uses Jaccard similarity between query tags and stored example tags.

        Args:
            role: Track role to match
            instruction: Generation instruction (tags extracted from it)
            key: Target key (e.g., "C major")
            tempo: Target tempo in BPM
            top_k: Number of examples to return

        Returns:
            List of up to top_k best-matching examples
        """
        if not self.examples:
            return []

        query_tags = set(self._extract_tags(instruction, role, key, tempo))
        if not query_tags:
            return []

        scored = []
        for ex in self.examples:
            ex_tags = set(ex.tags)
            intersection = len(query_tags & ex_tags)
            union = len(query_tags | ex_tags)
            if union == 0:
                continue
            jaccard = intersection / union

            # Bonus for matching role (most important)
            if ex.role == role:
                jaccard += 0.3

            scored.append((jaccard, ex))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Only return examples with meaningful similarity
        return [ex for score, ex in scored[:top_k] if score > 0.15]

    def format_examples_for_prompt(self, examples: list[MIDIExample]) -> str:
        """Format retrieved examples as a prompt prefix for MIDI-LLM.

        Args:
            examples: Retrieved examples from query()

        Returns:
            Formatted string to prepend to the generation prompt
        """
        if not examples:
            return ""

        parts = []
        for i, ex in enumerate(examples, 1):
            token_str = "".join(f"<|midi_token_{t}|>" for t in ex.token_ids)
            parts.append(
                f"[EXAMPLE {i}: {ex.description}]\n{token_str}\n"
            )

        return "\n".join(parts) + "\n[YOUR COMPOSITION]\n"

    def _extract_tags(
        self, instruction: str, role: str, key: str = "", tempo: float = 0.0
    ) -> list[str]:
        """Extract searchable tags from an instruction and metadata."""
        tags = [role]
        text = instruction.lower()

        # Instrument tags
        instruments = [
            "piano", "guitar", "bass", "strings", "violin", "cello",
            "drums", "flute", "trumpet", "saxophone", "organ", "harp",
            "brass", "choir", "pad", "synth",
        ]
        for inst in instruments:
            if inst in text:
                tags.append(inst)

        # Mood/energy tags
        moods = {
            "calm": ["calm", "peaceful", "gentle", "soft", "relaxing", "serene"],
            "energetic": ["energetic", "fast", "upbeat", "lively", "powerful", "dynamic"],
            "sad": ["sad", "melancholy", "somber", "dark", "minor"],
            "happy": ["happy", "cheerful", "joyful", "bright", "major"],
            "dramatic": ["dramatic", "intense", "epic"],
        }
        for mood, keywords in moods.items():
            if any(kw in text for kw in keywords):
                tags.append(mood)

        # Style tags
        styles = ["jazz", "classical", "rock", "pop", "blues", "folk", "electronic"]
        for style in styles:
            if style in text:
                tags.append(style)

        # Key tag
        if key:
            tags.append(key.replace(" ", "_"))

        # Tempo range tag
        if tempo > 0:
            if tempo < 80:
                tags.append("slow")
            elif tempo < 120:
                tags.append("moderate")
            else:
                tags.append("fast")

        return tags
