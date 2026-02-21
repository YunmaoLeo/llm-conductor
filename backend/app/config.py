"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI (GPT-4o Conductor)
    openai_api_key: str = ""  # REQUIRED - set in .env

    # Ollama (MIDI-LLM Musicians)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "midi-llm-q6"
    ollama_timeout: float = 300.0

    # Generation defaults
    default_temperature: float = 0.85
    default_top_p: float = 0.92
    default_max_tokens: int = 2046
    max_iterations: int = 5

    # Role-specific temperature overrides (lower = more coherent)
    role_temperature: dict[str, float] = {
        "melody": 0.88,
        "harmony": 0.75,
        "bass": 0.70,
        "rhythm": 0.78,
    }

    # Paths
    output_dir: str = "./outputs"
    soundfont_path: str = "./soundfonts/FluidR3_GM.sf2"

    # MIDI-LLM
    system_prompt: str = (
        "You are a professional composer generating single-instrument MIDI music. "
        "Rules: "
        "1) Generate ONLY the one instrument requested — never add extra instruments. "
        "2) Use clear phrase structure: organize notes into 4-bar or 8-bar phrases, "
        "with brief rests between phrases for natural breathing. "
        "3) Use melodic motifs: introduce a short melodic idea early and develop it "
        "through repetition, transposition, or variation. "
        "4) Match the specified key precisely — the majority of notes should be on "
        "scale degrees of the requested key. "
        "5) Match the specified tempo — use note durations that create the requested rhythmic feel. "
        "6) Keep polyphony under 4 simultaneous notes. "
        "7) Use dynamic variation — vary velocity for musical expression, avoid flat dynamics. "
        "8) Avoid: random pitch sequences, excessively long notes with no movement, "
        "dense clusters with no rhythmic pattern, or notes outside the specified register. "
        "Compose music for: "
    )

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
    )


settings = Settings()
