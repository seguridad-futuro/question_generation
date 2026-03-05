"""Question model V2 - Compatible con estructura Supabase."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from .question_option import QuestionOption


class Question(BaseModel):
    """Modelo de pregunta para oposiciones de Guardia Civil.

    V2: Compatible con estructura Supabase que separa questions y question_options.
    Las opciones de respuesta están en una relación 1-N con la pregunta.
    """
    # ==================== ID Y CONTENIDO ====================
    id: Optional[int] = None
    question: str
    tip: Optional[str] = None  # Feedback/explicación detallada
    article: Optional[str] = None  # Artículo o referencia legal

    # ==================== CLASIFICACIÓN ====================
    topic: int  # FK a topics
    academy_id: int = 1
    classification_category_id: Optional[int] = None  # Clasificación cruzada
    classification_topic_id: Optional[int] = None  # Clasificación cruzada

    # ==================== ORDEN Y VISUALIZACIÓN ====================
    order_num: int = 0
    published: bool = True
    shuffled: Optional[bool] = None  # Si las opciones deben mezclarse

    # ==================== MULTIMEDIA ====================
    question_image_url: str = ''
    retro_image_url: str = ''
    retro_audio_enable: bool = False
    retro_audio_text: str = ''
    retro_audio_url: str = ''

    # ==================== ESTADÍSTICAS DE USO ====================
    num_answered: int = 0
    num_fails: int = 0
    num_empty: int = 0
    # difficult_rate se calcula: num_answered / (num_answered + num_fails + num_empty)

    # ==================== METADATA DE CREACIÓN ====================
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    created_by: Optional[str] = None  # UUID del usuario
    created_by_cms_user_id: Optional[int] = None

    # ==================== CHALLENGES ====================
    challenge_by_tutor: bool = False
    challenge_reason: Optional[str] = None

    # ==================== GENERACIÓN LLM ====================
    llm_model: Optional[str] = None
    by_llm: bool = True
    question_prompt: Optional[str] = None
    generation_method: Optional[str] = None  # 'advanced_prompt', 'simple', etc.

    # ==================== QUALITY TRACKING ====================
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    source_chunk_id: Optional[str] = None
    source_document: Optional[str] = None
    source_page: Optional[int] = None

    # ==================== DEDUPLICATION & RETRY ====================
    is_duplicate: bool = False
    duplicate_of: Optional[int] = None
    retry_count: int = 0
    needs_manual_review: bool = False
    review_time: Optional[float] = None
    manual_review_reason: Optional[str] = None
    manual_review_details: Optional[str] = None
    review_comment: Optional[str] = None
    review_details: Optional[str] = None

    # ==================== TÉCNICAS DE DISTRACCIÓN ====================
    distractor_techniques: Optional[str] = None  # JSON array de técnicas

    # ==================== OPCIONES DE RESPUESTA (RELACIÓN 1-N) ====================
    options: List[QuestionOption] = Field(default_factory=list)

    @field_validator('faithfulness_score', 'relevancy_score')
    @classmethod
    def validate_scores(cls, v):
        """Validar que los scores estén entre 0 y 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('score must be between 0 and 1')
        return v

    def to_dict(self, exclude_none: bool = False, include_options: bool = True):
        """Convierte a dict para SQLite.

        Args:
            exclude_none: Si True, excluye campos None
            include_options: Si True, incluye las opciones (para inserts relacionales)
        """
        data = self.model_dump(exclude_none=exclude_none, exclude={'options'})

        # Convertir datetimes a string para SQLite
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat()

        # Incluir opciones si se solicita
        if include_options:
            data['options'] = [opt.to_dict(exclude_none=exclude_none) for opt in self.options]

        return data

    def get_correct_option(self) -> Optional[QuestionOption]:
        """Obtiene la opción correcta."""
        for option in self.options:
            if option.is_correct:
                return option
        return None

    def get_correct_answer_text(self) -> Optional[str]:
        """Obtiene el texto de la respuesta correcta."""
        correct = self.get_correct_option()
        return correct.answer if correct else None

    def get_correct_answer_order(self) -> Optional[int]:
        """Obtiene el orden (1-4) de la respuesta correcta."""
        correct = self.get_correct_option()
        return correct.option_order if correct else None

    def get_all_answers(self) -> List[str]:
        """Obtiene lista de todas las respuestas ordenadas."""
        return [opt.answer for opt in sorted(self.options, key=lambda x: x.option_order)]

    def get_text_for_embedding(self) -> str:
        """Texto para deduplicación semántica.

        Combina la pregunta con la respuesta correcta para crear
        un texto único que represente la pregunta completa.
        """
        correct_answer = self.get_correct_answer_text() or ""
        return f"{self.question} {correct_answer}"

    def calculate_difficult_rate(self) -> float:
        """Calcula la tasa de dificultad de la pregunta."""
        total = self.num_answered + self.num_fails + self.num_empty
        if total == 0:
            return 0.0
        return round(self.num_answered / total, 3)

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

    # ==================== MÉTODOS DE COMPATIBILIDAD LEGACY ====================

    @classmethod
    def from_legacy_dict(cls, data: dict) -> "Question":
        """Crea una instancia desde el formato legacy (answer1-4).

        Args:
            data: Dict con campos answer1, answer2, answer3, answer4 (opcional), solution

        Returns:
            Question con opciones creadas desde answer1-4
        """
        # Extraer opciones del formato legacy
        options = []
        for i in range(1, 5):
            answer_key = f"answer{i}"
            if answer_key in data and data[answer_key]:
                options.append(QuestionOption(
                    question_id=data.get('id', 0),
                    answer=data[answer_key],
                    is_correct=(data.get('solution') == i),
                    option_order=i
                ))

        # Remover campos legacy del dict
        legacy_fields = ['answer1', 'answer2', 'answer3', 'answer4', 'solution']
        clean_data = {k: v for k, v in data.items() if k not in legacy_fields}

        # Crear instancia con opciones
        clean_data['options'] = options
        return cls(**clean_data)

    def to_legacy_dict(self, exclude_none: bool = False) -> dict:
        """Convierte a formato legacy (answer1-4, solution).

        Útil para compatibilidad con código antiguo.
        """
        data = self.to_dict(exclude_none=exclude_none, include_options=False)

        # Agregar campos legacy
        for i, option in enumerate(sorted(self.options, key=lambda x: x.option_order), 1):
            data[f'answer{i}'] = option.answer
            if option.is_correct:
                data['solution'] = i

        # Asegurar que siempre haya 4 opciones (rellenar con None si faltan)
        for i in range(len(self.options) + 1, 5):
            data[f'answer{i}'] = None

        return data

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
