"""Prompt loader helpers for externalized prompts."""
from __future__ import annotations

from typing import Dict, Any
from config.settings import get_settings


def load_prompt_text(name: str) -> str:
    """Load a prompt text file from config/prompts/{agent}/ by basename (without extension)."""
    settings = get_settings()
    # Extract agent prefix (agent_b, agent_c, agent_z) from the prompt name
    parts = name.split("_")
    if len(parts) >= 2:
        agent_prefix = f"{parts[0]}_{parts[1]}"
        path = settings.prompts_dir / agent_prefix / f"{name}.txt"
    else:
        # Fallback for non-standard names
        path = settings.prompts_dir / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, values: Dict[str, Any]) -> str:
    """Replace {placeholders} without touching other braces."""
    rendered = template
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, str(value))
    return rendered
