"""Quality thresholds and deduplication configuration."""

from pydantic import BaseModel
from typing import Literal


class QualityThresholds(BaseModel):
    """Thresholds para clasificación de calidad de preguntas.

    Usado por el Agente C (Quality Gate) para decidir si una pregunta:
    - auto_pass: Alta calidad, aprobar automáticamente
    - auto_fail: Baja calidad, regenerar
    - manual_review: Zona gris, requiere revisión humana
    """

    # Auto-pass thresholds (alta calidad)
    auto_pass_faithfulness: float = 0.85
    auto_pass_relevancy: float = 0.85

    # Auto-fail thresholds (baja calidad, regenerar)
    auto_fail_faithfulness: float = 0.60
    auto_fail_relevancy: float = 0.60

    def classify(self, faithfulness: float, relevancy: float) -> Literal["auto_pass", "auto_fail", "manual_review"]:
        """Clasifica una pregunta según sus métricas de calidad.

        Args:
            faithfulness: Score de faithfulness (0-1)
            relevancy: Score de relevancy (0-1)

        Returns:
            Clasificación: "auto_pass", "auto_fail", o "manual_review"
        """
        if (faithfulness >= self.auto_pass_faithfulness and
            relevancy >= self.auto_pass_relevancy):
            return "auto_pass"
        elif (faithfulness < self.auto_fail_faithfulness or
              relevancy < self.auto_fail_relevancy):
            return "auto_fail"
        else:
            return "manual_review"

    def should_retry(self, faithfulness: float, relevancy: float) -> bool:
        """Determina si una pregunta debe ser regenerada (auto-fail)."""
        return self.classify(faithfulness, relevancy) == "auto_fail"


class DeduplicationConfig(BaseModel):
    """Configuración para deduplicación semántica.

    Usado por el Agente D (Persistence) para detectar duplicados.
    """

    # Threshold de similitud coseno para considerar duplicado
    similarity_threshold: float = 0.85

    # Modelo de embeddings para similitud (OpenAI)
    # Opciones: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    embedding_model: str = "text-embedding-3-small"

    # Batch size para procesamiento de embeddings
    batch_size: int = 32

    def is_duplicate(self, similarity_score: float) -> bool:
        """Determina si dos preguntas son duplicadas según similitud."""
        return similarity_score >= self.similarity_threshold


# Instancias por defecto
DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()
DEFAULT_DEDUP_CONFIG = DeduplicationConfig()
