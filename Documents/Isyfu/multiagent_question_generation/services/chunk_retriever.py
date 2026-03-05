"""Servicio de retrieval inteligente de chunks para el Agente B.

Este servicio proporciona:
- Inicialización con filtrado de chunks no sustantivos
- Indexación de embeddings para búsqueda semántica
- Recuperación de chunks relacionados
- Construcción de contexto enriquecido combinando múltiples chunks
- Evaluación de suficiencia de contexto
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from models.chunk import Chunk
from utils.metadata_filter import PDFMetadataFilter, MetadataFilterResult
from utils.embeddings import ChunkEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ContextSufficiencyResult:
    """Resultado de evaluación de suficiencia de contexto."""

    is_sufficient: bool
    reason: str
    confidence: float  # 0-1
    recommended_additional_chunks: int
    detected_topics: List[str] = field(default_factory=list)
    complexity_level: str = "medium"  # low, medium, high


@dataclass
class EnrichedContext:
    """Contexto enriquecido con múltiples chunks."""

    primary_chunk_id: str
    primary_content: str
    related_chunks: List[Dict[str, Any]]  # [{chunk_id, content, similarity}]
    combined_content: str
    total_tokens: int
    num_chunks_used: int
    topics_covered: List[str] = field(default_factory=list)


class ChunkRetrieverService:
    """Servicio de retrieval inteligente para el Agente B.

    Características:
    - Filtrado automático de chunks no sustantivos
    - Indexación de embeddings con FAISS
    - Búsqueda de chunks semánticamente relacionados
    - Construcción de contextos enriquecidos
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.7,
        max_related_chunks: int = 3,
        max_context_tokens: int = 2000,
        min_substantive_ratio: float = 0.3,
        filter_metadata: bool = True,
        enable_embeddings: bool = True
    ):
        """Inicializa el servicio de retrieval.

        Args:
            embedding_model: Modelo de sentence-transformers para embeddings.
            similarity_threshold: Umbral mínimo de similitud (0-1).
            max_related_chunks: Máximo de chunks adicionales a recuperar.
            max_context_tokens: Límite de tokens en contexto combinado.
            min_substantive_ratio: Ratio mínimo de contenido sustantivo.
            filter_metadata: Si aplicar filtrado de metadatos.
        """
        self.similarity_threshold = similarity_threshold
        self.max_related_chunks = max_related_chunks
        self.max_context_tokens = max_context_tokens
        self.min_substantive_ratio = min_substantive_ratio
        self.filter_metadata = filter_metadata

        # Servicios internos
        self._metadata_filter = PDFMetadataFilter(min_substantive_ratio)
        self._embeddings_enabled = enable_embeddings
        self._embedding_service = None
        if enable_embeddings:
            from config.settings import get_settings
            settings = get_settings()
            self._embedding_service = ChunkEmbeddingService(
                model_name=embedding_model,
                similarity_threshold=similarity_threshold,
                provider=settings.embedding_provider,
                openai_model=settings.openai_embedding_model
            )

        # Estado
        self._chunks: Dict[str, Chunk] = {}
        self._is_initialized = False
        self._filtered_chunk_ids: List[str] = []
        self._substantive_chunk_ids: List[str] = []

    def initialize(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Inicializa el servicio con una lista de chunks.

        Proceso:
        1. Filtra metadatos de cada chunk
        2. Marca chunks como sustantivos o no
        3. Indexa embeddings de chunks sustantivos

        Args:
            chunks: Lista de chunks a procesar.

        Returns:
            Dict con estadísticas de inicialización.
        """
        logger.info(f"Inicializando ChunkRetrieverService con {len(chunks)} chunks")

        stats = {
            "total_chunks": len(chunks),
            "substantive_chunks": 0,
            "filtered_chunks": 0,
            "indexed_chunks": 0,
            "avg_substantive_ratio": 0.0
        }

        if not chunks:
            logger.warning("No hay chunks para inicializar")
            return stats

        substantive_ratios = []
        chunk_ids_to_index = []
        contents_to_index = []
        metadata_to_index = []

        for chunk in chunks:
            # Guardar referencia
            self._chunks[chunk.chunk_id] = chunk

            # Filtrar metadatos si está habilitado
            if self.filter_metadata:
                filter_result = self._metadata_filter.filter_chunk(chunk.content)

                # Actualizar chunk con resultados del filtrado
                chunk.is_substantive = filter_result.is_substantive
                chunk.substantive_ratio = filter_result.substantive_ratio
                chunk.clean_content = filter_result.clean_content

                substantive_ratios.append(filter_result.substantive_ratio)

                if filter_result.is_substantive:
                    self._substantive_chunk_ids.append(chunk.chunk_id)
                    chunk_ids_to_index.append(chunk.chunk_id)
                    contents_to_index.append(filter_result.clean_content)
                    metadata_to_index.append({
                        "source_document": chunk.source_document,
                        "page": chunk.page,
                        "token_count": chunk.token_count
                    })
                else:
                    self._filtered_chunk_ids.append(chunk.chunk_id)
                    if filter_result.filter_reasons:
                        logger.debug(
                            f"Chunk {chunk.chunk_id} filtrado: {filter_result.filter_reasons}"
                        )
            else:
                # Sin filtrado, todos son sustantivos
                self._substantive_chunk_ids.append(chunk.chunk_id)
                chunk_ids_to_index.append(chunk.chunk_id)
                contents_to_index.append(chunk.content)
                metadata_to_index.append({
                    "source_document": chunk.source_document,
                    "page": chunk.page,
                    "token_count": chunk.token_count
                })

        # Indexar embeddings de chunks sustantivos (si están habilitados)
        if self._embeddings_enabled and self._embedding_service is not None:
            if chunk_ids_to_index:
                num_indexed = self._embedding_service.index_chunks(
                    chunk_ids=chunk_ids_to_index,
                    chunk_contents=contents_to_index,
                    chunk_metadata=metadata_to_index
                )

                # Marcar chunks como indexados
                for chunk_id in chunk_ids_to_index:
                    self._chunks[chunk_id].embedding_indexed = True

                stats["indexed_chunks"] = num_indexed
        else:
            logger.info("Embeddings desactivados; se omite indexado semántico")

        # Calcular estadísticas
        stats["substantive_chunks"] = len(self._substantive_chunk_ids)
        stats["filtered_chunks"] = len(self._filtered_chunk_ids)

        if substantive_ratios:
            stats["avg_substantive_ratio"] = sum(substantive_ratios) / len(substantive_ratios)

        self._is_initialized = True

        logger.info(
            f"Inicialización completa: {stats['substantive_chunks']} sustantivos, "
            f"{stats['filtered_chunks']} filtrados, {stats['indexed_chunks']} indexados"
        )

        return stats

    def assess_context_sufficiency(
        self,
        chunk_id: str,
        question_type: Optional[str] = None
    ) -> ContextSufficiencyResult:
        """Evalúa si un chunk tiene suficiente contexto para generar preguntas.

        Criterios de evaluación:
        - Longitud del contenido
        - Complejidad del tema
        - Presencia de referencias a otros conceptos
        - Tipo de pregunta deseada

        Args:
            chunk_id: ID del chunk a evaluar.
            question_type: Tipo de pregunta (opcional): factual, conceptual, application

        Returns:
            ContextSufficiencyResult con la evaluación.
        """
        if not self._is_initialized:
            raise RuntimeError("El servicio no está inicializado. Llama a initialize() primero.")

        if chunk_id not in self._chunks:
            return ContextSufficiencyResult(
                is_sufficient=False,
                reason=f"Chunk no encontrado: {chunk_id}",
                confidence=1.0,
                recommended_additional_chunks=0
            )

        chunk = self._chunks[chunk_id]

        # Si no es sustantivo, no es suficiente
        if not chunk.is_substantive:
            return ContextSufficiencyResult(
                is_sufficient=False,
                reason="Chunk marcado como no sustantivo (metadatos/índice)",
                confidence=0.9,
                recommended_additional_chunks=0
            )

        content = chunk.get_content_for_generation()
        token_count = chunk.get_effective_token_count()

        # Detectar temas y complejidad
        detected_topics = self._detect_topics(content)
        complexity = self._assess_complexity(content)

        # Evaluar suficiencia
        reasons = []
        is_sufficient = True
        recommended_chunks = 0

        # Criterio 1: Longitud mínima
        if token_count < 100:
            is_sufficient = False
            reasons.append("Contenido muy corto")
            recommended_chunks = max(recommended_chunks, 2)
        elif token_count < 200:
            reasons.append("Contenido relativamente corto")
            recommended_chunks = max(recommended_chunks, 1)

        # Criterio 2: Referencias externas
        external_refs = self._detect_external_references(content)
        if external_refs:
            reasons.append(f"Referencias a otros artículos/secciones: {external_refs}")
            recommended_chunks = max(recommended_chunks, len(external_refs))

        # Criterio 3: Complejidad del tema
        if complexity == "high":
            reasons.append("Tema de alta complejidad")
            recommended_chunks = max(recommended_chunks, 2)
        elif complexity == "medium":
            recommended_chunks = max(recommended_chunks, 1)

        # Criterio 4: Tipo de pregunta
        if question_type == "application":
            reasons.append("Preguntas de aplicación requieren contexto amplio")
            recommended_chunks = max(recommended_chunks, 2)
        elif question_type == "conceptual":
            recommended_chunks = max(recommended_chunks, 1)

        # Ajustar límite
        recommended_chunks = min(recommended_chunks, self.max_related_chunks)

        # Calcular confianza basada en criterios cumplidos
        confidence = 0.9 if not reasons else 0.7

        return ContextSufficiencyResult(
            is_sufficient=is_sufficient and recommended_chunks == 0,
            reason="; ".join(reasons) if reasons else "Contexto suficiente",
            confidence=confidence,
            recommended_additional_chunks=recommended_chunks,
            detected_topics=detected_topics,
            complexity_level=complexity
        )

    def retrieve_related_chunks(
        self,
        chunk_id: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Recupera chunks semánticamente relacionados.

        Args:
            chunk_id: ID del chunk de referencia.
            k: Número de chunks a recuperar (usa max_related_chunks si None).
            threshold: Umbral de similitud (usa similarity_threshold si None).

        Returns:
            Lista de dicts con {chunk_id, content, similarity_score, metadata}.
        """
        if not self._is_initialized:
            raise RuntimeError("El servicio no está inicializado.")
        if not self._embeddings_enabled or self._embedding_service is None:
            return []

        if chunk_id not in self._chunks:
            logger.warning(f"Chunk no encontrado: {chunk_id}")
            return []

        chunk = self._chunks[chunk_id]

        # Si no está indexado, no podemos buscar similares
        if not chunk.embedding_indexed:
            return []

        k = k if k is not None else self.max_related_chunks
        threshold = threshold if threshold is not None else self.similarity_threshold

        # Buscar similares
        try:
            results = self._embedding_service.search_similar_to_chunk(
                chunk_id=chunk_id,
                k=k,
                threshold=threshold
            )
        except Exception as e:
            logger.error(f"Error buscando chunks similares: {e}")
            return []

        # Formatear resultados
        related = []
        for result in results:
            related_chunk = self._chunks.get(result.chunk_id)
            if related_chunk:
                related.append({
                    "chunk_id": result.chunk_id,
                    "content": related_chunk.get_content_for_generation(),
                    "similarity_score": result.similarity_score,
                    "metadata": result.metadata,
                    "source_document": related_chunk.source_document,
                    "page": related_chunk.page
                })

        # Actualizar chunk con IDs relacionados
        chunk.related_chunk_ids = [r["chunk_id"] for r in related]

        return related

    def build_enriched_context(
        self,
        chunk_id: str,
        max_additional_chunks: Optional[int] = None
    ) -> EnrichedContext:
        """Construye un contexto enriquecido combinando múltiples chunks.

        Args:
            chunk_id: ID del chunk principal.
            max_additional_chunks: Máximo de chunks adicionales (usa config si None).

        Returns:
            EnrichedContext con el contexto combinado.
        """
        if chunk_id not in self._chunks:
            raise ValueError(f"Chunk no encontrado: {chunk_id}")

        primary_chunk = self._chunks[chunk_id]
        primary_content = primary_chunk.get_content_for_generation()
        primary_tokens = primary_chunk.get_effective_token_count()

        max_additional = max_additional_chunks or self.max_related_chunks

        # Recuperar chunks relacionados
        related = self.retrieve_related_chunks(chunk_id, k=max_additional)

        # Construir contexto combinado respetando límite de tokens
        combined_parts = [f"[CONTEXTO PRINCIPAL - {chunk_id}]\n{primary_content}"]
        total_tokens = primary_tokens
        chunks_used = 1
        selected_related = []

        for rel in related:
            rel_tokens = len(rel["content"]) // 4  # Estimación

            if total_tokens + rel_tokens <= self.max_context_tokens:
                combined_parts.append(
                    f"\n\n[CONTEXTO RELACIONADO - {rel['chunk_id']} "
                    f"(similitud: {rel['similarity_score']:.2f})]\n{rel['content']}"
                )
                total_tokens += rel_tokens
                chunks_used += 1
                selected_related.append(rel)
            else:
                logger.debug(
                    f"Chunk {rel['chunk_id']} excede límite de tokens "
                    f"({total_tokens + rel_tokens} > {self.max_context_tokens})"
                )
                break

        combined_content = "\n".join(combined_parts)

        # Detectar temas cubiertos
        topics = self._detect_topics(combined_content)

        return EnrichedContext(
            primary_chunk_id=chunk_id,
            primary_content=primary_content,
            related_chunks=selected_related,
            combined_content=combined_content,
            total_tokens=total_tokens,
            num_chunks_used=chunks_used,
            topics_covered=topics
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Obtiene un chunk por su ID.

        Args:
            chunk_id: ID del chunk.

        Returns:
            Chunk o None si no existe.
        """
        return self._chunks.get(chunk_id)

    def get_substantive_chunks(self) -> List[Chunk]:
        """Obtiene todos los chunks sustantivos.

        Returns:
            Lista de chunks marcados como sustantivos.
        """
        return [self._chunks[cid] for cid in self._substantive_chunk_ids]

    def get_filtered_chunk_ids(self) -> List[str]:
        """Obtiene IDs de chunks filtrados (no sustantivos).

        Returns:
            Lista de IDs de chunks filtrados.
        """
        return self._filtered_chunk_ids.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio.

        Returns:
            Dict con estadísticas.
        """
        return {
            "is_initialized": self._is_initialized,
            "total_chunks": len(self._chunks),
            "substantive_chunks": len(self._substantive_chunk_ids),
            "filtered_chunks": len(self._filtered_chunk_ids),
            "embedding_stats": self._embedding_service.get_stats() if self._is_initialized else None,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_related_chunks": self.max_related_chunks,
                "max_context_tokens": self.max_context_tokens,
                "min_substantive_ratio": self.min_substantive_ratio,
                "filter_metadata": self.filter_metadata
            }
        }

    def _detect_topics(self, content: str) -> List[str]:
        """Detecta temas en el contenido.

        Args:
            content: Texto a analizar.

        Returns:
            Lista de temas detectados.
        """
        topics = []
        content_lower = content.lower()

        topic_keywords = {
            "delitos_vida": ["homicidio", "asesinato", "muerte", "matar"],
            "delitos_patrimonio": ["robo", "hurto", "estafa", "apropiación"],
            "delitos_integridad": ["lesiones", "daño corporal", "agresión"],
            "constitucion": ["constitución", "constitucional", "derechos fundamentales"],
            "procedimiento": ["procedimiento", "proceso", "trámite", "recurso"],
            "organización_policial": ["guardia civil", "policía", "cuerpos seguridad"],
            "administrativo": ["administración", "administrativo", "funcionario"],
            "penal": ["código penal", "delito", "pena", "prisión"]
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)

        return topics if topics else ["general"]

    def _assess_complexity(self, content: str) -> str:
        """Evalúa la complejidad del contenido.

        Args:
            content: Texto a evalizar.

        Returns:
            "low", "medium" o "high"
        """
        # Indicadores de alta complejidad
        high_complexity_indicators = [
            "sin perjuicio de",
            "no obstante",
            "salvo que",
            "excepto cuando",
            "siempre que",
            "a menos que",
            "en virtud de",
            "de conformidad con"
        ]

        content_lower = content.lower()
        word_count = len(content.split())

        # Contar indicadores de complejidad
        complexity_count = sum(
            1 for indicator in high_complexity_indicators
            if indicator in content_lower
        )

        # Evaluar
        if complexity_count >= 3 or word_count > 500:
            return "high"
        elif complexity_count >= 1 or word_count > 250:
            return "medium"
        else:
            return "low"

    def _detect_external_references(self, content: str) -> List[str]:
        """Detecta referencias a otros artículos o secciones.

        Args:
            content: Texto a analizar.

        Returns:
            Lista de referencias detectadas.
        """
        import re

        references = []

        # Patrones de referencia
        patterns = [
            r'(?:artículo|art\.?)\s*(\d+(?:\.\d+)?)',
            r'(?:sección|secc\.?)\s*(\d+)',
            r'(?:capítulo|cap\.?)\s*(\d+)',
            r'(?:título)\s*([IVXLCDM]+|\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)

        return list(set(references))[:5]  # Limitar a 5


# Instancia singleton global
_retriever_service: Optional[ChunkRetrieverService] = None


def get_chunk_retriever_service(
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.7,
    max_related_chunks: int = 3,
    max_context_tokens: int = 2000,
    min_substantive_ratio: float = 0.3,
    filter_metadata: bool = True,
    enable_embeddings: bool = True,
    force_new: bool = False
) -> ChunkRetrieverService:
    """Obtiene o crea el servicio singleton de chunk retrieval.

    Args:
        embedding_model: Modelo de embeddings.
        similarity_threshold: Umbral de similitud.
        max_related_chunks: Máximo de chunks relacionados.
        max_context_tokens: Máximo de tokens en contexto.
        min_substantive_ratio: Ratio mínimo sustantivo.
        filter_metadata: Si filtrar metadatos.
        force_new: Si forzar nueva instancia.

    Returns:
        ChunkRetrieverService singleton.
    """
    global _retriever_service

    if _retriever_service is None or force_new:
        _retriever_service = ChunkRetrieverService(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_related_chunks=max_related_chunks,
            max_context_tokens=max_context_tokens,
            min_substantive_ratio=min_substantive_ratio,
            filter_metadata=filter_metadata,
            enable_embeddings=enable_embeddings
        )

    return _retriever_service


# ==========================================
# EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Crear chunks de ejemplo
    test_chunks = [
        Chunk(
            chunk_id="chunk_001",
            content="""Artículo 138 del Código Penal:
            El que matare a otro será castigado, como reo de homicidio,
            con la pena de prisión de diez a quince años.""",
            source_document="codigo_penal.pdf",
            page=45,
            token_count=50
        ),
        Chunk(
            chunk_id="chunk_002",
            content="""Artículo 139 del Código Penal:
            Será castigado con la pena de prisión de quince a veinticinco años,
            como reo de asesinato, el que matare a otro concurriendo alguna
            de las circunstancias siguientes: alevosía, precio o ensañamiento.""",
            source_document="codigo_penal.pdf",
            page=46,
            token_count=60
        ),
        Chunk(
            chunk_id="chunk_003",
            content="""La Constitución Española de 1978 establece en su artículo 1
            que España se constituye en un Estado social y democrático de Derecho,
            que propugna como valores superiores de su ordenamiento jurídico
            la libertad, la justicia, la igualdad y el pluralismo político.""",
            source_document="constitucion.pdf",
            page=1,
            token_count=55
        ),
        Chunk(
            chunk_id="chunk_metadata",
            content="""
            Manual de Derecho Penal
            Autor: Dr. García López
            Email: garcia@universidad.es
            ISBN: 978-84-1234-567-8
            Editorial Jurídica, 2024
            """,
            source_document="manual.pdf",
            page=1,
            token_count=30
        )
    ]

    # Inicializar servicio
    service = ChunkRetrieverService()
    stats = service.initialize(test_chunks)

    print(f"\nEstadísticas de inicialización:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Sustantivos: {stats['substantive_chunks']}")
    print(f"  Filtrados: {stats['filtered_chunks']}")

    # Evaluar suficiencia de contexto
    print(f"\nEvaluación de contexto para chunk_001:")
    sufficiency = service.assess_context_sufficiency("chunk_001")
    print(f"  Suficiente: {sufficiency.is_sufficient}")
    print(f"  Razón: {sufficiency.reason}")
    print(f"  Chunks adicionales recomendados: {sufficiency.recommended_additional_chunks}")

    # Recuperar chunks relacionados
    print(f"\nChunks relacionados con chunk_001:")
    related = service.retrieve_related_chunks("chunk_001", k=2)
    for rel in related:
        print(f"  - {rel['chunk_id']} (similitud: {rel['similarity_score']:.3f})")

    # Construir contexto enriquecido
    print(f"\nContexto enriquecido para chunk_001:")
    enriched = service.build_enriched_context("chunk_001")
    print(f"  Chunks usados: {enriched.num_chunks_used}")
    print(f"  Tokens totales: {enriched.total_tokens}")
    print(f"  Temas: {enriched.topics_covered}")
