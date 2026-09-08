"""
Configuration — loaded from environment variables + .env file.
Defaults to local Ollama (qwen2.5-coder:7b) at http://localhost:11434/v1.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Config(BaseModel):
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5-coder:7b"
    llm_base_url: str = "http://localhost:11434/v1"
    max_experiments: int = 8
    max_recoveries: int = 6
    max_recoveries_per_subtask: int = 2
    runs_dir: Path = Path("runs")
    use_stub_tools: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "ollama") or "ollama",
            openai_model=(
                os.getenv("LLM_MODEL")
                or os.getenv("OPENAI_MODEL")
                or "qwen2.5-coder:7b"
            ),
            llm_base_url=(
                os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "http://localhost:11434/v1"
            ),
            max_experiments=int(os.getenv("MAX_EXPERIMENTS", "8")),
            max_recoveries=int(os.getenv("MAX_RECOVERIES", "6")),
            max_recoveries_per_subtask=int(
                os.getenv("MAX_RECOVERIES_PER_SUBTASK", "2")
            ),
        )
