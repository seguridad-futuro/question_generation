"""LLM Factory - Crea instancias de LLM OpenAI.

Este módulo proporciona una factory para crear LLMs de OpenAI de forma consistente.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config.settings import get_settings


_LOGGING_CONFIGURED = False


def _silence_external_loggers() -> None:
    """Reduce el ruido de logs externos (HTTP/API)."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


_silence_external_loggers()


def create_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> BaseChatModel:
    """Factory para crear LLM de OpenAI.

    Args:
        model: Modelo específico (None = usar el configurado)
        temperature: Temperatura para generación (0-1)
            - 0: Determinístico, consistente
            - 0.7: Balance creatividad/consistencia
            - 1: Máxima creatividad
        max_tokens: Máximo de tokens en la respuesta
        **kwargs: Argumentos adicionales para ChatOpenAI

    Returns:
        Instancia de BaseChatModel configurada

    Examples:
        >>> llm = create_llm()
        >>> llm = create_llm(model="gpt-5-mini", temperature=0.3, max_tokens=2000)
    """
    settings = get_settings()

    if model is None:
        model = settings.openai_model

    return _create_openai_llm(model, temperature, max_tokens, **kwargs)


def _create_openai_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> BaseChatModel:
    """Crea LLM de OpenAI.

    Args:
        model: Modelo OpenAI (gpt-5-mini, gpt-4o-mini, etc.)
        temperature: Temperatura
        max_tokens: Max tokens
        **kwargs: Argumentos adicionales para ChatOpenAI

    Returns:
        ChatOpenAI instance

    Raises:
        ValueError: Si OPENAI_API_KEY no está configurada
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY no configurada. "
            "Añade OPENAI_API_KEY en .env"
        )

    # El parámetro "timeout" se pasa directamente a ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.openai_api_key,
        **kwargs
    )


# ==========================================
# PRESETS ESPECÍFICOS
# ==========================================

def create_llm_for_generation() -> BaseChatModel:
    """LLM optimizado para generación de preguntas.

    Configuración:
    - Modelo: configurable via GENERATION_MODEL (fallback a OPENAI_MODEL)
    - Temperatura alta (0.7) para creatividad
    - max_tokens=16000 para acomodar reasoning tokens
    - timeout=120 para evitar cuelgues

    Returns:
        LLM configurado para generación
    """
    settings = get_settings()

    # Modelo para generación (override con GENERATION_MODEL en .env)
    generation_model = settings.generation_model or settings.openai_model

    extra_kwargs = {}
    if settings.generation_reasoning_effort and generation_model.startswith("gpt-5"):
        extra_kwargs["reasoning"] = {"effort": settings.generation_reasoning_effort}

    return create_llm(
        model=generation_model,
        temperature=0.7,
        max_tokens=16000,  # Alto para acomodar reasoning tokens de gpt-5
        timeout=120,
        **extra_kwargs
    )


def create_llm_for_evaluation() -> BaseChatModel:
    """LLM optimizado para evaluación (razonamiento).

    Configuración:
    - Temperatura baja (0) para consistencia
    - max_tokens=16000 para acomodar reasoning tokens
    - timeout=120 para evitar cuelgues

    Returns:
        LLM configurado para evaluación
    """
    settings = get_settings()
    timeout_seconds = getattr(settings, "agent_c_timeout_seconds", 120)
    return create_llm(
        model=settings.openai_model,
        temperature=0,
        max_tokens=16000,
        timeout=timeout_seconds
    )


def create_llm_for_agents() -> BaseChatModel:
    """LLM optimizado para agentes (tool calling).

    Configuración:
    - Temperatura baja (0) para razonamiento consistente
    - max_tokens=10000 para acomodar reasoning overhead
    - timeout=60 para evitar cuelgues

    Returns:
        LLM configurado para agentes
    """
    settings = get_settings()
    return create_llm(
        model=settings.openai_model,
        temperature=0,
        max_tokens=10000,
        timeout=60  # Timeout de 60 segundos
    )


def create_llm_for_quick_evaluation(max_tokens: int = 16000) -> BaseChatModel:
    """LLM rapido para validacion basica de preguntas."""
    settings = get_settings()
    timeout_seconds = getattr(settings, "agent_c_timeout_seconds", 120)
    return create_llm(
        model=settings.openai_model,
        temperature=0,
        max_tokens=max_tokens,
        timeout=timeout_seconds
    )


# ==========================================
# HELPERS
# ==========================================

def get_current_provider() -> str:
    """Retorna el proveedor LLM configurado actualmente."""
    return "openai"


def get_current_model() -> str:
    """Retorna el modelo LLM configurado actualmente."""
    settings = get_settings()
    return settings.openai_model


def print_llm_info():
    """Imprime información sobre la configuración LLM actual."""
    settings = get_settings()

    print("🤖 CONFIGURACIÓN LLM")
    print("=" * 50)
    print("Proveedor: openai")
    print(f"Modelo: {settings.openai_model}")
    print(f"API Key: {settings.openai_api_key[:10]}..." if settings.openai_api_key else "❌ No configurada")
    print("=" * 50)


if __name__ == "__main__":
    # Demo del factory
    print_llm_info()

    print("\n📝 Probando creación de LLMs:")

    try:
        # Auto
        print("\n1. LLM (desde settings):")
        llm_auto = create_llm()
        print(f"   ✅ Creado: {type(llm_auto).__name__}")

        # Generation preset
        print("\n2. LLM para generación:")
        llm_gen = create_llm_for_generation()
        print(f"   ✅ Creado: {type(llm_gen).__name__}")

        # Evaluation preset
        print("\n3. LLM para evaluación:")
        llm_eval = create_llm_for_evaluation()
        print(f"   ✅ Creado: {type(llm_eval).__name__}")

        # Agents preset
        print("\n4. LLM para agentes:")
        llm_agent = create_llm_for_agents()
        print(f"   ✅ Creado: {type(llm_agent).__name__}")

        print("\n✅ Todos los LLMs creados correctamente!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de tener configurado:")
        print("- OPENAI_API_KEY")
