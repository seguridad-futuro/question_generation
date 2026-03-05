"""Quality metrics model for Ragas evaluation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal


class QualityMetrics(BaseModel):
    """Métricas de calidad de Ragas para una pregunta.

    Usado por el Agente C (Quality Gate) para evaluar preguntas generadas.
    """
    question_id: Optional[str] = None
    faithfulness: float  # ¿La respuesta es fiel al contexto?
    answer_relevancy: float  # ¿La respuesta es relevante a la pregunta?
    context_precision: Optional[float] = None  # ¿El contexto es preciso?

    # Metadata adicional
    context: Optional[str] = None  # Contexto usado para evaluación
    feedback: Optional[str] = None  # Feedback para retry si falla

    @validator('faithfulness', 'answer_relevancy', 'context_precision')
    def validate_score_range(cls, v):
        """Validar que los scores estén entre 0 y 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError('metric score must be between 0 and 1')
        return v

    def get_classification(
        self,
        auto_pass_faithfulness: float = 0.85,
        auto_pass_relevancy: float = 0.85,
        auto_fail_faithfulness: float = 0.60,
        auto_fail_relevancy: float = 0.60
    ) -> Literal["auto_pass", "auto_fail", "manual_review"]:
        """Clasifica la pregunta según thresholds.

        Returns:
            - "auto_pass": Alta calidad, aprobar automáticamente
            - "auto_fail": Baja calidad, regenerar
            - "manual_review": Zona gris, requiere revisión manual
        """
        if (self.faithfulness >= auto_pass_faithfulness and
            self.answer_relevancy >= auto_pass_relevancy):
            return "auto_pass"
        elif (self.faithfulness < auto_fail_faithfulness or
              self.answer_relevancy < auto_fail_relevancy):
            return "auto_fail"
        else:
            return "manual_review"

    def to_dict(self):
        """Convierte a dict para serialización."""
        return self.model_dump(exclude_none=True)

    def __str__(self):
        """Representación string para logs."""
        classification = self.get_classification()
        return (f"QualityMetrics(faithfulness={self.faithfulness:.3f}, "
                f"relevancy={self.answer_relevancy:.3f}, "
                f"classification={classification})")
