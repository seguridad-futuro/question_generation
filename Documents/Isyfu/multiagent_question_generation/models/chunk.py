"""Chunk model for document processing."""

from pydantic import BaseModel, Field
from typing import Optional, List


class Chunk(BaseModel):
    """Modelo de chunk de documento con metadata de trazabilidad.

    Representa un fragmento de documento procesado por el Agente A.

    Incluye campos para el sistema multi-chunk inteligente:
    - Filtrado de metadatos de PDF
    - Indexación de embeddings
    - Relación con chunks similares
    """
    chunk_id: str
    content: str
    source_document: str  # Path del documento original
    page: Optional[int] = None  # Número de página (si aplica)
    start_char: Optional[int] = None  # Posición de inicio en el documento
    end_char: Optional[int] = None  # Posición final en el documento
    token_count: Optional[int] = None  # Número estimado de tokens
    metadata: dict = Field(default_factory=dict)  # Metadata adicional

    # ==========================================
    # CAMPOS MULTI-CHUNK (Agente B Mejorado)
    # ==========================================

    # Indica si el chunk contiene contenido sustantivo (legal/normativo)
    # False si es principalmente metadatos, índice, etc.
    is_substantive: bool = True

    # Ratio de contenido sustantivo vs. metadatos (0-1)
    # Calculado por PDFMetadataFilter
    substantive_ratio: float = 1.0

    # Contenido limpio sin metadatos (emails, URLs, etc.)
    # None si no se ha procesado con filtro de metadatos
    clean_content: Optional[str] = None

    # Indica si el chunk ha sido indexado en el sistema de embeddings
    embedding_indexed: bool = False

    # IDs de chunks relacionados semánticamente
    # Populado por ChunkRetrieverService
    related_chunk_ids: List[str] = Field(default_factory=list)

    def to_dict(self):
        """Convierte a dict para serialización."""
        return self.model_dump()

    def get_content_for_generation(self) -> str:
        """Retorna el mejor contenido para generación de preguntas.

        Prioriza clean_content si está disponible, sino usa content original.

        Returns:
            Contenido óptimo para generación.
        """
        if self.clean_content and self.clean_content.strip():
            return self.clean_content
        return self.content

    def get_effective_token_count(self) -> int:
        """Estima tokens del contenido efectivo (limpio o original).

        Returns:
            Estimación de tokens.
        """
        content = self.get_content_for_generation()
        # Estimación simple: ~4 caracteres por token en español
        return len(content) // 4

    def __str__(self):
        """Representación string para logs."""
        substantive_flag = "✓" if self.is_substantive else "✗"
        return (
            f"Chunk({self.chunk_id}, source={self.source_document}, "
            f"page={self.page}, tokens~{self.token_count}, "
            f"substantive={substantive_flag})"
        )
