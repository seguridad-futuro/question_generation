"""GraphState definition for LangGraph workflow.

Este módulo define el estado compartido que se pasa entre todos los agentes
en el pipeline Map-Reduce de generación de preguntas.
"""

from typing import TypedDict, List, Dict, Optional, Any
from models.question import Question
from models.chunk import Chunk
from models.quality_metrics import QualityMetrics


class GraphState(TypedDict):
    """Estado compartido entre todos los agentes del pipeline.

    Este estado se pasa y modifica a través del grafo LangGraph,
    manteniendo toda la información del pipeline de generación.

    Flujo del estado:
    1. Input → metadata, input_docs
    2. Agente A → chunks
    3. Map Node → generated_questions (por chunk)
    4. Agente C (per chunk) → validated_questions, failed_questions, quality_scores
    5. Retry logic → retry_counts
    6. Reduce Node → consolidación
    7. Agente D → persisted_ids, dedup_stats
    """

    # ==================
    # INPUT
    # ==================
    input_docs: List[str]  # Paths a documentos PDF/TXT/MD
    metadata: Dict[str, Any]  # {academy, topic, num_questions, llm_model, batch_name}

    # ==================
    # AGENTE A: CHUNKING
    # ==================
    chunks: List[Chunk]  # Chunks procesados con metadata

    # ==================
    # AGENTE B: GENERATION (Map Node)
    # ==================
    # Preguntas generadas por chunk (antes de quality gate)
    # Key: chunk_id, Value: lista de Question objects
    generated_questions: Dict[str, List[Question]]

    # ==================
    # AGENTE C: QUALITY GATE (per chunk)
    # ==================
    # Preguntas que PASARON quality gate (auto-pass)
    validated_questions: List[Question]

    # Preguntas en zona gris (requieren revisión manual)
    # Scores entre auto_pass y auto_fail thresholds
    manual_review_questions: List[Question]

    # Preguntas que FALLARON quality gate (auto-fail, para retry)
    # Key: chunk_id, Value: lista de {question, feedback}
    failed_questions: Dict[str, List[Dict[str, Any]]]

    # Métricas de calidad por pregunta
    # Key: question_id (temporal), Value: QualityMetrics
    quality_scores: Dict[str, QualityMetrics]

    # ==================
    # RETRY TRACKING
    # ==================
    # Contador de retries por chunk
    # Key: chunk_id, Value: número de intentos
    retry_counts: Dict[str, int]

    # Máximo de retries permitidos (default: 3)
    max_retries: int

    # ==================
    # AGENTE D: PERSISTENCE
    # ==================
    # IDs de preguntas persistidas en SQLite
    persisted_ids: List[int]

    # Estadísticas de deduplicación
    # {unique: X, duplicates: Y, total: Z}
    dedup_stats: Dict[str, int]

    # ==================
    # ERROR HANDLING
    # ==================
    # Lista de errores encontrados durante el pipeline
    # {node: "B", chunk_id: "1", error: "..."}
    errors: List[Dict[str, str]]

    # ==================
    # PROGRESS TRACKING
    # ==================
    # Track progreso del pipeline
    current_step: Optional[str]  # "chunking", "generation", "quality", "persistence"
    completed_chunks: List[str]  # chunk_ids procesados completamente
    total_questions_generated: int
    total_questions_validated: int

    # ==================
    # MULTI-CHUNK SYSTEM (Agente B Mejorado)
    # ==================
    # Indica si el servicio de retrieval de chunks está inicializado
    chunk_retriever_initialized: bool

    # Contextos enriquecidos por chunk (cuando se usa multi-chunk)
    # Key: chunk_id, Value: {primary_content, related_chunks[], total_tokens}
    enriched_contexts: Dict[str, Dict]

    # IDs de chunks filtrados (no sustantivos)
    # Chunks marcados como metadatos/índices que no generan preguntas
    filtered_chunk_ids: List[str]

    # Estadísticas del sistema multi-chunk
    # {total_chunks, substantive_chunks, filtered_chunks,
    #  multi_context_questions, single_context_questions}
    multi_chunk_stats: Dict[str, int]


def create_initial_state(
    input_docs: List[str],
    academy: int,
    topic: int,
    num_questions: int,
    llm_model: Optional[str] = None,
    batch_name: Optional[str] = None,
    max_retries: int = 3
) -> GraphState:
    """Crea el estado inicial del grafo con los parámetros de entrada.

    Args:
        input_docs: Paths a documentos a procesar
        academy: ID de la academia
        topic: ID del topic
        num_questions: Número total de preguntas a generar
        llm_model: Modelo LLM a usar
        batch_name: Nombre del batch (se genera automático si None)
        max_retries: Máximo de retries por chunk

    Returns:
        GraphState inicial para comenzar el pipeline
    """
    from datetime import datetime

    if batch_name is None:
        batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if llm_model is None:
        from config.settings import get_settings
        settings = get_settings()
        llm_model = settings.generation_model or settings.openai_model

    return GraphState(
        # Input
        input_docs=input_docs,
        metadata={
            "academy": academy,
            "topic": topic,
            "num_questions": num_questions,
            "llm_model": llm_model,
            "batch_name": batch_name,
        },

        # Agente A
        chunks=[],

        # Agente B
        generated_questions={},

        # Agente C
        validated_questions=[],
        manual_review_questions=[],
        failed_questions={},
        quality_scores={},

        # Retry
        retry_counts={},
        max_retries=max_retries,

        # Agente D
        persisted_ids=[],
        dedup_stats={"unique": 0, "duplicates": 0, "total": 0},

        # Errors
        errors=[],

        # Progress
        current_step="initial",
        completed_chunks=[],
        total_questions_generated=0,
        total_questions_validated=0,

        # Multi-chunk system
        chunk_retriever_initialized=False,
        enriched_contexts={},
        filtered_chunk_ids=[],
        multi_chunk_stats={
            "total_chunks": 0,
            "substantive_chunks": 0,
            "filtered_chunks": 0,
            "multi_context_questions": 0,
            "single_context_questions": 0
        },
    )
