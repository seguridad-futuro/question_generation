"""Sistema de embeddings con FAISS para búsqueda semántica de chunks.

Este módulo proporciona:
- Generación de embeddings con sentence-transformers
- Índice FAISS para búsqueda eficiente por similitud
- Búsqueda de k chunks similares con threshold configurable
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSearchResult:
    """Resultado de una búsqueda de similitud."""

    chunk_id: str
    content: str
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkEmbeddingService:
    """Servicio de embeddings para chunks usando sentence-transformers y FAISS.

    Características:
    - Modelo multilingüe optimizado para español
    - Índice FAISS con Inner Product (coseno normalizado)
    - Búsqueda de k chunks más similares con threshold
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.7,
        use_gpu: bool = False,
        provider: str = "sentence_transformers",
        openai_model: str = "text-embedding-3-small"
    ):
        """Inicializa el servicio de embeddings.

        Args:
            model_name: Nombre del modelo de sentence-transformers.
            similarity_threshold: Umbral mínimo de similitud (0-1).
            use_gpu: Si usar GPU para embeddings (requiere faiss-gpu).
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.provider = provider
        self.openai_model = openai_model

        # Estado interno
        self._model = None
        self._index = None
        self._chunk_ids: List[str] = []
        self._chunk_contents: List[str] = []
        self._chunk_metadata: List[Dict] = []
        self._embedding_dim: Optional[int] = None
        self._is_initialized = False

    @property
    def model(self):
        """Carga lazy del modelo de embeddings."""
        if self._model is None:
            if self.provider == "openai":
                try:
                    from langchain_openai import OpenAIEmbeddings
                    from config.settings import get_settings
                    settings = get_settings()
                    logger.info(f"Usando OpenAI embeddings: {self.openai_model}")
                    self._model = OpenAIEmbeddings(
                        model=self.openai_model,
                        api_key=settings.openai_api_key
                    )
                except ImportError:
                    raise ImportError(
                        "langchain-openai no está instalado. "
                        "Instala con: pip install langchain-openai"
                    )
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Cargando modelo de embeddings: {self.model_name}")
                    self._model = SentenceTransformer(self.model_name)
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
                    logger.info(f"Modelo cargado. Dimensión de embeddings: {self._embedding_dim}")
                except ImportError:
                    raise ImportError(
                        "sentence-transformers no está instalado. "
                        "Instala con: pip install sentence-transformers"
                    )
        return self._model

    def _get_faiss_index(self, dim: int):
        """Crea un índice FAISS con Inner Product.

        Args:
            dim: Dimensión de los embeddings.

        Returns:
            Índice FAISS configurado.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS no está instalado. "
                "Instala con: pip install faiss-cpu (o faiss-gpu para GPU)"
            )

        # Inner Product para similitud de coseno (vectores normalizados)
        index = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            try:
                # Intentar usar GPU
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Usando FAISS con GPU")
            except Exception as e:
                logger.warning(f"No se pudo usar GPU para FAISS: {e}. Usando CPU.")

        return index

    def compute_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Calcula embeddings para una lista de textos.

        Args:
            texts: Lista de textos a embedir.
            batch_size: Tamaño del batch para procesamiento.
            show_progress: Si mostrar barra de progreso.

        Returns:
            Array numpy de embeddings normalizados (N x dim).
        """
        if not texts:
            return np.array([])

        logger.info(f"Calculando embeddings para {len(texts)} textos...")

        if self.provider == "openai":
            embeddings_list = []
            total = len(texts)
            for start in range(0, total, batch_size):
                batch = texts[start:start + batch_size]
                batch_embeddings = self.model.embed_documents(batch)
                embeddings_list.extend(batch_embeddings)
            embeddings = np.array(embeddings_list, dtype="float32")
            # Normalizar L2
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            return embeddings

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalizar para similitud de coseno
            convert_to_numpy=True
        )

        return embeddings.astype('float32')

    def index_chunks(
        self,
        chunk_ids: List[str],
        chunk_contents: List[str],
        chunk_metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> int:
        """Indexa chunks para búsqueda por similitud.

        Args:
            chunk_ids: IDs únicos de cada chunk.
            chunk_contents: Contenido de texto de cada chunk.
            chunk_metadata: Metadata opcional para cada chunk.
            batch_size: Tamaño del batch para embeddings.

        Returns:
            Número de chunks indexados.
        """
        if len(chunk_ids) != len(chunk_contents):
            raise ValueError("chunk_ids y chunk_contents deben tener la misma longitud")

        if not chunk_contents:
            logger.warning("No hay chunks para indexar")
            return 0

        # Calcular embeddings
        embeddings = self.compute_embeddings(chunk_contents, batch_size)

        # Crear índice FAISS
        dim = embeddings.shape[1]
        self._index = self._get_faiss_index(dim)
        self._embedding_dim = dim

        # Agregar al índice
        self._index.add(embeddings)

        # Guardar referencias
        self._chunk_ids = list(chunk_ids)
        self._chunk_contents = list(chunk_contents)
        self._chunk_metadata = list(chunk_metadata) if chunk_metadata else [{} for _ in chunk_ids]

        self._is_initialized = True

        logger.info(f"Indexados {len(chunk_ids)} chunks en FAISS")
        return len(chunk_ids)

    def search_similar(
        self,
        query_text: str,
        k: int = 5,
        threshold: Optional[float] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[EmbeddingSearchResult]:
        """Busca los k chunks más similares a un texto de consulta.

        Args:
            query_text: Texto de consulta.
            k: Número máximo de resultados.
            threshold: Umbral de similitud (usa el default si None).
            exclude_ids: IDs de chunks a excluir de resultados.

        Returns:
            Lista de EmbeddingSearchResult ordenada por similitud.
        """
        if not self._is_initialized:
            raise RuntimeError("El índice no está inicializado. Llama a index_chunks primero.")

        if not query_text.strip():
            return []

        threshold = threshold if threshold is not None else self.similarity_threshold
        exclude_ids = set(exclude_ids) if exclude_ids else set()

        # Calcular embedding de la consulta
        query_embedding = self.compute_embeddings([query_text], show_progress=False)

        # Buscar más resultados de los necesarios para filtrar
        search_k = min(k * 3, self._index.ntotal)  # Buscar más para compensar filtros
        distances, indices = self._index.search(query_embedding, search_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS retorna -1 si no hay más resultados
                continue

            chunk_id = self._chunk_ids[idx]

            # Filtrar por exclusión
            if chunk_id in exclude_ids:
                continue

            # Filtrar por threshold
            if dist < threshold:
                continue

            results.append(EmbeddingSearchResult(
                chunk_id=chunk_id,
                content=self._chunk_contents[idx],
                similarity_score=float(dist),
                rank=len(results) + 1,
                metadata=self._chunk_metadata[idx]
            ))

            if len(results) >= k:
                break

        return results

    def search_similar_to_chunk(
        self,
        chunk_id: str,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[EmbeddingSearchResult]:
        """Busca chunks similares a un chunk existente.

        Args:
            chunk_id: ID del chunk de referencia.
            k: Número máximo de resultados.
            threshold: Umbral de similitud.

        Returns:
            Lista de EmbeddingSearchResult (excluyendo el chunk de referencia).
        """
        if chunk_id not in self._chunk_ids:
            raise ValueError(f"Chunk ID no encontrado: {chunk_id}")

        idx = self._chunk_ids.index(chunk_id)
        chunk_content = self._chunk_contents[idx]

        return self.search_similar(
            query_text=chunk_content,
            k=k,
            threshold=threshold,
            exclude_ids=[chunk_id]  # Excluir el propio chunk
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Obtiene información de un chunk por su ID.

        Args:
            chunk_id: ID del chunk.

        Returns:
            Dict con content y metadata, o None si no existe.
        """
        if chunk_id not in self._chunk_ids:
            return None

        idx = self._chunk_ids.index(chunk_id)
        return {
            "chunk_id": chunk_id,
            "content": self._chunk_contents[idx],
            "metadata": self._chunk_metadata[idx]
        }

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calcula la similitud de coseno entre dos textos.

        Args:
            text1: Primer texto.
            text2: Segundo texto.

        Returns:
            Similitud de coseno (0-1).
        """
        embeddings = self.compute_embeddings([text1, text2], show_progress=False)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def batch_compute_similarities(
        self,
        query_texts: List[str],
        reference_texts: List[str]
    ) -> np.ndarray:
        """Calcula matriz de similitudes entre dos listas de textos.

        Args:
            query_texts: Lista de textos de consulta.
            reference_texts: Lista de textos de referencia.

        Returns:
            Matriz de similitudes (len(query) x len(reference)).
        """
        query_embeddings = self.compute_embeddings(query_texts, show_progress=False)
        ref_embeddings = self.compute_embeddings(reference_texts, show_progress=False)

        # Producto punto = similitud de coseno (ya están normalizados)
        similarities = np.dot(query_embeddings, ref_embeddings.T)
        return similarities

    @property
    def num_indexed(self) -> int:
        """Número de chunks indexados."""
        return len(self._chunk_ids)

    @property
    def is_initialized(self) -> bool:
        """Si el índice está inicializado."""
        return self._is_initialized

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio.

        Returns:
            Dict con estadísticas del índice.
        """
        return {
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "num_indexed": self.num_indexed,
            "embedding_dim": self._embedding_dim,
            "is_initialized": self._is_initialized,
            "use_gpu": self.use_gpu
        }

    def reset(self):
        """Reinicia el índice, eliminando todos los chunks."""
        self._index = None
        self._chunk_ids = []
        self._chunk_contents = []
        self._chunk_metadata = []
        self._is_initialized = False
        logger.info("Índice de embeddings reiniciado")


# Instancia global singleton
_embedding_service: Optional[ChunkEmbeddingService] = None


def get_embedding_service(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.7,
    force_new: bool = False
) -> ChunkEmbeddingService:
    """Obtiene o crea la instancia singleton del servicio de embeddings.

    Args:
        model_name: Nombre del modelo.
        similarity_threshold: Umbral de similitud.
        force_new: Si crear una nueva instancia.

    Returns:
        ChunkEmbeddingService singleton.
    """
    global _embedding_service

    if _embedding_service is None or force_new:
        _embedding_service = ChunkEmbeddingService(
            model_name=model_name,
            similarity_threshold=similarity_threshold
        )

    return _embedding_service


# ==========================================
# EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Crear servicio
    service = ChunkEmbeddingService()

    # Chunks de ejemplo
    chunk_ids = ["chunk_001", "chunk_002", "chunk_003", "chunk_004"]
    chunk_contents = [
        "Artículo 138. El que matare a otro será castigado como reo de homicidio con prisión de diez a quince años.",
        "Artículo 139. Será castigado con prisión de quince a veinticinco años como reo de asesinato el que matare con alevosía.",
        "La Constitución Española de 1978 establece los derechos fundamentales de los ciudadanos.",
        "El procedimiento administrativo común está regulado por la Ley 39/2015."
    ]

    # Indexar chunks
    num_indexed = service.index_chunks(chunk_ids, chunk_contents)
    print(f"Chunks indexados: {num_indexed}")

    # Buscar chunks similares
    query = "delitos contra la vida y penas de prisión por homicidio"
    results = service.search_similar(query, k=3, threshold=0.5)

    print(f"\nBúsqueda: '{query}'")
    print("-" * 60)
    for result in results:
        print(f"  #{result.rank} [{result.chunk_id}] (score: {result.similarity_score:.4f})")
        print(f"      {result.content[:80]}...")

    # Buscar similares a un chunk
    print(f"\nChunks similares a 'chunk_001':")
    similar = service.search_similar_to_chunk("chunk_001", k=2, threshold=0.5)
    for result in similar:
        print(f"  [{result.chunk_id}] score: {result.similarity_score:.4f}")

    # Estadísticas
    print(f"\nEstadísticas: {service.get_stats()}")
