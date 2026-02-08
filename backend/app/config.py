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
    default_temperature: float = 1.0
    default_top_p: float = 0.98
    default_max_tokens: int = 2046
    max_iterations: int = 5

    # Paths
    output_dir: str = "./outputs"
    soundfont_path: str = "./soundfonts/FluidR3_GM.sf2"

    # MIDI-LLM
    system_prompt: str = (
        "You are a world-class composer. "
        "Please compose some music according to the following description: "
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
