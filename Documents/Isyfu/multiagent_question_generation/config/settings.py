"""Global settings for the question generation system."""

from pydantic_settings import BaseSettings
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Optional

from config.config_models import (
    OpenAIConfig,
    AgentZConfig,
    AgentBConfig,
    AgentCConfig,
    EmbeddingsConfig,
    DedupConfig,
    PathsConfig,
    ConfigBundle,
)

class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    # ==========================================
    # LLM CONFIGURATION (OpenAI)
    # ==========================================
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-mini"
    # Rewriter model (Agent Z, reescritura selectiva)
    rewriter_model: Optional[str] = "gpt-5-mini"
    rewriter_temperature: float = 0.0
    rewriter_max_tokens: int = 1200
    rewriter_max_input_chars: int = 40000
    rewriter_concurrency: int = 3
    rewriter_timeout_seconds: int = 300
    rewriter_doc_timeout_seconds: int = 0
    rewriter_partial_save_every: int = 1
    # Forzar reescritura de todos los chunks (modo reescritura ON)
    rewriter_force_rewrite: bool = False
    agent_llm_timeout_seconds: int = 120
    rewriter_coordination_mode: str = "llm"
    rewriter_coordination_cache_enable: bool = True
    rewriter_coordination_cache_every: int = 20
    agent_reasoning_effort: Optional[str] = None
    rewriter_reasoning_effort: Optional[str] = None
    # Learned regex patterns for metadata cleanup (Agent Z)
    rewriter_pattern_enable: bool = True
    rewriter_pattern_min_hits: int = 2
    rewriter_pattern_min_len: int = 6
    rewriter_pattern_max_len: int = 160
    rewriter_pattern_boundary_lines: int = 3
    rewriter_pattern_path: Optional[Path] = None

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_merge_steps: int = 8
    max_merged_chars: int = 60000

    # Quality Thresholds
    quality_threshold_faithfulness: float = 0.85
    quality_threshold_relevancy: float = 0.85
    quality_threshold_auto_fail_faithfulness: float = 0.60
    quality_threshold_auto_fail_relevancy: float = 0.60

    # Retry Configuration
    max_retries: int = 3

    # Agent C (Quality) speed controls
    agent_c_timeout_seconds: int = 120
    agent_c_fast_mode: bool = True
    agent_c_context_max_chars: int = 3000
    agent_c_difficulty_model: Optional[str] = None

    # Generation Configuration
    batch_size: int = 10
    num_questions_per_chunk: int = 5
    num_answer_options: int = 4
    num_correct_options: int = 1
    generation_model: Optional[str] = None
    generation_max_tokens: int = 2000
    generation_reasoning_effort: Optional[str] = None
    tip_min_words: int = 40  # Reducido de 60 para ser menos estricto
    tip_max_words: int = 200  # Aumentado de 180 para dar más margen
    bcde_questions_per_chunk: int = 1
    bcde_chunk_cache_path: Optional[Path] = None
    b_enable_guardian: bool = True
    b_enable_sequential_expansion: bool = True
    b_enable_semantic_expansion: bool = True
    b_enable_multi_chunk: bool = True
    b_parallel_agents: int = 1
    b_max_expansions: int = 2

    # Deduplication
    similarity_threshold: float = 0.85

    # ==========================================
    # MULTI-CHUNK & EMBEDDINGS (Agente B Mejorado)
    # ==========================================
    # Embeddings provider (openai|sentence_transformers)
    embedding_provider: str = "openai"
    # Modelo de embeddings para búsqueda semántica (sentence-transformers)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # OpenAI embeddings model
    openai_embedding_model: str = "text-embedding-3-small"

    # Umbral de similitud para recuperar chunks relacionados
    embedding_similarity_threshold: float = 0.7

    # Máximo de chunks relacionados a recuperar
    max_related_chunks: int = 3

    # Máximo de tokens en contexto combinado
    max_context_tokens: int = 2000

    # Ratio mínimo de contenido sustantivo (filtrado de metadatos)
    min_substantive_ratio: float = 0.3

    # Habilitar generación con múltiples chunks
    enable_multi_chunk: bool = True

    # Habilitar filtrado de metadatos de PDF
    filter_metadata: bool = True

    # Paths
    project_root: Path = Path(__file__).parent.parent
    input_docs_dir: Path = project_root / "input_docs"
    output_dir: Path = project_root / "output"
    database_dir: Path = project_root / "database"
    examples_dir: Path = project_root / "examples"
    prompts_dir: Path = project_root / "config" / "prompts"

    # Database
    database_path: Path = database_dir / "questions.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @cached_property
    def config(self) -> ConfigBundle:
        """Aggregated per-agent config models with inline docs."""
        return ConfigBundle(
            openai=OpenAIConfig(
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
            ),
            agent_z=AgentZConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_merge_steps=self.max_merge_steps,
                max_merged_chars=self.max_merged_chars,
                rewriter_model=self.rewriter_model,
                rewriter_temperature=self.rewriter_temperature,
                rewriter_max_tokens=self.rewriter_max_tokens,
                rewriter_max_input_chars=self.rewriter_max_input_chars,
                rewriter_concurrency=self.rewriter_concurrency,
                rewriter_timeout_seconds=self.rewriter_timeout_seconds,
                rewriter_doc_timeout_seconds=self.rewriter_doc_timeout_seconds,
                agent_llm_timeout_seconds=self.agent_llm_timeout_seconds,
                agent_reasoning_effort=self.agent_reasoning_effort,
                rewriter_reasoning_effort=self.rewriter_reasoning_effort,
                rewriter_partial_save_every=self.rewriter_partial_save_every,
                rewriter_coordination_mode=self.rewriter_coordination_mode,
                rewriter_coordination_cache_enable=self.rewriter_coordination_cache_enable,
                rewriter_coordination_cache_every=self.rewriter_coordination_cache_every,
                rewriter_pattern_enable=self.rewriter_pattern_enable,
                rewriter_pattern_min_hits=self.rewriter_pattern_min_hits,
                rewriter_pattern_min_len=self.rewriter_pattern_min_len,
                rewriter_pattern_max_len=self.rewriter_pattern_max_len,
                rewriter_pattern_boundary_lines=self.rewriter_pattern_boundary_lines,
                rewriter_pattern_path=self.rewriter_pattern_path,
                rewriter_force_rewrite=self.rewriter_force_rewrite,
                filter_metadata=self.filter_metadata,
                min_substantive_ratio=self.min_substantive_ratio,
            ),
            agent_b=AgentBConfig(
                batch_size=self.batch_size,
                num_questions_per_chunk=self.num_questions_per_chunk,
                num_answer_options=self.num_answer_options,
                num_correct_options=self.num_correct_options,
                generation_model=self.generation_model,
                generation_max_tokens=self.generation_max_tokens,
                generation_reasoning_effort=self.generation_reasoning_effort,
                tip_min_words=self.tip_min_words,
                tip_max_words=self.tip_max_words,
                bcde_questions_per_chunk=self.bcde_questions_per_chunk,
                bcde_chunk_cache_path=self.bcde_chunk_cache_path,
                b_enable_guardian=self.b_enable_guardian,
                b_enable_sequential_expansion=self.b_enable_sequential_expansion,
                b_enable_semantic_expansion=self.b_enable_semantic_expansion,
                b_enable_multi_chunk=self.b_enable_multi_chunk,
                enable_multi_chunk=self.enable_multi_chunk,
                b_parallel_agents=self.b_parallel_agents,
                b_max_expansions=self.b_max_expansions,
            ),
            agent_c=AgentCConfig(
                quality_threshold_faithfulness=self.quality_threshold_faithfulness,
                quality_threshold_relevancy=self.quality_threshold_relevancy,
                quality_threshold_auto_fail_faithfulness=self.quality_threshold_auto_fail_faithfulness,
                quality_threshold_auto_fail_relevancy=self.quality_threshold_auto_fail_relevancy,
                max_retries=self.max_retries,
                timeout_seconds=self.agent_c_timeout_seconds,
                fast_mode=self.agent_c_fast_mode,
                context_max_chars=self.agent_c_context_max_chars,
                difficulty_model=self.agent_c_difficulty_model,
            ),
            embeddings=EmbeddingsConfig(
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                openai_embedding_model=self.openai_embedding_model,
                embedding_similarity_threshold=self.embedding_similarity_threshold,
                max_related_chunks=self.max_related_chunks,
                max_context_tokens=self.max_context_tokens,
            ),
            dedup=DedupConfig(
                similarity_threshold=self.similarity_threshold,
            ),
            paths=PathsConfig(
                project_root=self.project_root,
                input_docs_dir=self.input_docs_dir,
                output_dir=self.output_dir,
                database_dir=self.database_dir,
                examples_dir=self.examples_dir,
                prompts_dir=self.prompts_dir,
                database_path=self.database_path,
            ),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
