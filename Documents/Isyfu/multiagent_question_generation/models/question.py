"""Question model - compatible with SQLite schema."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class Question(BaseModel):
    """Modelo de pregunta para oposiciones de Policía Nacional.

    Compatible con schema SQLite y con tracking de calidad/deduplicación.
    """
    id: Optional[int] = None
    academy: int
    question: str
    answer1: str
    answer2: str
    answer3: str
    answer4: Optional[str] = None
    solution: int  # 1, 2, 3, or 4 (if answer4 exists)
    tip: Optional[str] = None
    article: Optional[str] = None  # Texto del artículo legal citado
    topic: int
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    question_prompt: Optional[str] = None
    llm_model: Optional[str] = None
    order_num: Optional[int] = None
    by_llm: bool = True

    # Quality tracking
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    source_chunk_id: Optional[str] = None
    source_document: Optional[str] = None
    source_chunk: Optional[str] = None  # Texto completo del chunk fuente (para PDFs)
    generation_time: Optional[float] = None  # Tiempo de generación en segundos

    # Deduplication & retry
    is_duplicate: bool = False
    duplicate_of: Optional[int] = None
    retry_count: int = 0
    needs_manual_review: bool = False
    review_time: Optional[float] = None  # Tiempo de revisión (Agente C) en segundos
    manual_review_reason: Optional[str] = None  # Motivo breve para revisión
    manual_review_details: Optional[str] = None  # Detalle/razonamiento de revisión
    review_comment: Optional[str] = None  # Comentario breve de evaluación C
    review_details: Optional[str] = None  # Detalle adicional de evaluación C

    # Difficulty (evaluated by Agent C)
    difficulty: Optional[str] = None  # fácil, medio, difícil
    difficulty_reason: Optional[str] = None  # Razón del nivel asignado por Agente C

    # Multi-chunk context (when multiple articles are used)
    source_chunk_ids: Optional[List[str]] = None  # Lista de IDs de chunks usados
    context_strategy: Optional[str] = None  # "single_context" o "multi_context"
    num_chunks_used: Optional[int] = None  # Número de chunks/artículos relacionados usados

    @validator('solution')
    def validate_solution(cls, v, values):
        """Validar que solution sea válido según las opciones disponibles."""
        if v < 1 or v > 4:
            raise ValueError('solution must be between 1 and 4')
        # Si no hay answer4, solution debe ser 1, 2, or 3
        if 'answer4' in values and values['answer4'] is None and v == 4:
            raise ValueError('solution cannot be 4 when answer4 is None')
        return v

    @validator('faithfulness_score', 'relevancy_score')
    def validate_scores(cls, v):
        """Validar que los scores estén entre 0 y 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('score must be between 0 and 1')
        return v

    def to_dict(self, exclude_none: bool = False):
        """Convierte a dict para SQLite."""
        data = self.model_dump(exclude_none=exclude_none)
        # Convertir datetime a string para SQLite
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        return data

    def get_text_for_embedding(self) -> str:
        """Texto para deduplicación semántica.

        Combina la pregunta con la respuesta correcta para crear
        un texto único que represente la pregunta completa.
        """
        correct_answer = getattr(self, f"answer{self.solution}")
        return f"{self.question} {correct_answer}"

    def get_correct_answer(self) -> str:
        """Obtiene el texto de la respuesta correcta."""
        return getattr(self, f"answer{self.solution}")

    def get_all_answers(self) -> list[str]:
        """Obtiene lista de todas las respuestas (incluyendo None si no hay answer4)."""
        answers = [self.answer1, self.answer2, self.answer3]
        if self.answer4:
            answers.append(self.answer4)
        return answers

    def is_high_quality(self, faithfulness_threshold: float = 0.85, relevancy_threshold: float = 0.85) -> bool:
        """Verifica si la pregunta cumple con los thresholds de calidad."""
        if self.faithfulness_score is None or self.relevancy_score is None:
            return False
        return (
            self.faithfulness_score >= faithfulness_threshold and
            self.relevancy_score >= relevancy_threshold
        )

    def should_retry(self, faithfulness_fail: float = 0.60, relevancy_fail: float = 0.60) -> bool:
        """Verifica si la pregunta debe ser regenerada (auto-fail)."""
        if self.faithfulness_score is None or self.relevancy_score is None:
            return True
        return (
            self.faithfulness_score < faithfulness_fail or
            self.relevancy_score < relevancy_fail
        )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
