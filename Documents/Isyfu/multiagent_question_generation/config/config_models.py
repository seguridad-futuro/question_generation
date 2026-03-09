"""Structured config models for all .env-configurable settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI / LLM configuration."""
    # API key used by all agents that call OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    # Default model for general tasks
    openai_model: str = Field(default="gpt-5-mini", description="Default model for general tasks")


class AgentZConfig(BaseModel):
    """Agent Z (chunking + rewriting) configuration."""
    # Target chunk size (chars). Larger -> fewer chunks, more context per chunk.
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    # Overlap between chunks (chars). Larger -> more redundancy.
    chunk_overlap: int = Field(default=200, description="Overlap size in characters")
    # Max merges per chunk during coordination. Larger -> more aggressive merges.
    max_merge_steps: int = Field(default=8, description="Max merges per chunk during coordination")
    # Hard limit for merged chunk length (chars).
    max_merged_chars: int = Field(default=60000, description="Hard limit for merged chunk size (chars)")

    # Rewrite model (Agent Z)
    rewriter_model: Optional[str] = Field(default="gpt-5-mini", description="LLM model for rewrite")
    # Rewrite temperature
    rewriter_temperature: float = Field(default=0.0, description="Rewrite temperature")
    # Max output tokens for rewrite
    rewriter_max_tokens: int = Field(default=1200, description="Max output tokens for rewrite")
    # Max input chars before autosplit
    rewriter_max_input_chars: int = Field(default=40000, description="Max input chars before autosplit")
    # Parallel rewrite workers per document (unused when running sequentially)
    rewriter_concurrency: int = Field(default=1, description="Parallel rewrite workers per document")
    # Max seconds without progress before aborting rewrite tasks
    rewriter_timeout_seconds: int = Field(default=300, description="Rewrite timeout seconds")
    # Per-document hard timeout (seconds) in run_rewriter; 0 disables
    rewriter_doc_timeout_seconds: int = Field(default=0, description="Hard timeout per document (seconds)")
    # Timeout for Agent Z LLM calls (coordinate/analyze)
    agent_llm_timeout_seconds: int = Field(default=120, description="LLM timeout for Agent Z calls")
    # Reasoning effort for Agent Z LLM calls (gpt-5 only)
    agent_reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort for Agent Z calls")
    # Reasoning effort for rewrite calls (gpt-5 only)
    rewriter_reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort for rewrite calls")
    # Save incremental cache every N chunks (0 disables)
    rewriter_partial_save_every: int = Field(default=1, description="Incremental save every N chunks")
    # Coordination mode for Agent Z (llm|heuristic)
    rewriter_coordination_mode: str = Field(default="llm", description="Coordination mode")
    # Cache coordination decisions
    rewriter_coordination_cache_enable: bool = Field(default=True, description="Enable coordination cache")
    # Save coordination cache every N decisions
    rewriter_coordination_cache_every: int = Field(default=20, description="Coord cache save frequency")
    # Enable learning regex cleanup patterns
    rewriter_pattern_enable: bool = Field(default=True, description="Enable rewrite pattern learning")
    # Min hits before a pattern is applied
    rewriter_pattern_min_hits: int = Field(default=2, description="Min hits before pattern applies")
    # Min line length eligible for pattern
    rewriter_pattern_min_len: int = Field(default=6, description="Min line length for pattern")
    # Max line length eligible for pattern
    rewriter_pattern_max_len: int = Field(default=160, description="Max line length for pattern")
    # Boundary lines scanned for learning
    rewriter_pattern_boundary_lines: int = Field(default=3, description="Boundary lines scanned for learning")
    # Override path for pattern registry JSON
    rewriter_pattern_path: Optional[Path] = Field(default=None, description="Override path for pattern registry JSON")
    # Force rewrite on all chunks (Agent Z)
    rewriter_force_rewrite: bool = Field(default=False, description="Force rewrite on all chunks")
    # Enable metadata filter for PDF chunks
    filter_metadata: bool = Field(default=True, description="Enable metadata filtering")
    # Minimum substantive ratio to keep a chunk
    min_substantive_ratio: float = Field(default=0.3, description="Min substantive ratio (0-1)")


class AgentBConfig(BaseModel):
    """Agent B (question generation) configuration."""
    # Questions generated per chunk (legacy)
    batch_size: int = Field(default=10, description="Questions per chunk (legacy)")
    # Default questions per chunk
    num_questions_per_chunk: int = Field(default=5, description="Default questions per chunk")
    # Number of answer options per question
    num_answer_options: int = Field(default=4, description="Number of answer options")
    # Expected number of correct options (current support: 1)
    num_correct_options: int = Field(default=1, description="Expected number of correct options")
    # LLM model for generation (optional override)
    generation_model: Optional[str] = Field(default=None, description="LLM model for generation")
    # Max output tokens for generation
    generation_max_tokens: int = Field(default=2000, description="Max output tokens for generation")
    # GPT-5 reasoning effort: low|medium|high
    generation_reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort for generation")
    # Tip minimum words
    tip_min_words: int = Field(default=60, description="Minimum words in tip")
    # Tip maximum words
    tip_max_words: int = Field(default=180, description="Maximum words in tip")
    # Questions per chunk in BCDE pipeline
    bcde_questions_per_chunk: int = Field(default=1, description="Questions per chunk in BCDE")
    # Optional path for BCDE chunk usage cache
    bcde_chunk_cache_path: Optional[Path] = Field(default=None, description="BCDE chunk cache path")
    # Enable guardian (evaluate step)
    b_enable_guardian: bool = Field(default=True, description="Enable guardian step")
    # Allow sequential prev/next expansion
    b_enable_sequential_expansion: bool = Field(default=True, description="Enable sequential expansion")
    # Allow semantic expansion (FAISS)
    b_enable_semantic_expansion: bool = Field(default=False, description="Enable semantic expansion")
    # Enable multi-chunk generation (Agent B)
    b_enable_multi_chunk: bool = Field(default=True, description="Enable multi-chunk for Agent B")
    # Global toggle for multi-chunk (shared)
    enable_multi_chunk: bool = Field(default=True, description="Global multi-chunk toggle")
    # Max expansions per chunk
    b_max_expansions: int = Field(default=2, description="Max expansions per chunk")


class AgentCConfig(BaseModel):
    """Agent C (quality evaluation) configuration."""
    # Optional model override for difficulty analysis
    difficulty_model: Optional[str] = Field(default=None, description="LLM model for difficulty analysis")
    # Auto-pass faithfulness threshold
    quality_threshold_faithfulness: float = Field(default=0.85, description="Auto-pass faithfulness threshold")
    # Auto-pass relevancy threshold
    quality_threshold_relevancy: float = Field(default=0.85, description="Auto-pass relevancy threshold")
    # Auto-fail faithfulness threshold
    quality_threshold_auto_fail_faithfulness: float = Field(default=0.60, description="Auto-fail faithfulness threshold")
    # Auto-fail relevancy threshold
    quality_threshold_auto_fail_relevancy: float = Field(default=0.60, description="Auto-fail relevancy threshold")
    # Max retries for generation/evaluation
    max_retries: int = Field(default=3, description="Max retries")
    # Timeout (seconds) for Agent C LLM calls
    timeout_seconds: int = Field(default=120, description="Timeout seconds for Agent C LLM calls")
    # Fast mode skips extra LLM checks (tip consistency, tip supports, context, distractors)
    fast_mode: bool = Field(default=True, description="Skip extra checks for speed")
    # Max chars of context passed to quick evaluation
    context_max_chars: int = Field(default=3000, description="Max context chars for quick evaluation")


class EmbeddingsConfig(BaseModel):
    """Embeddings and retrieval configuration."""
    # Embeddings provider
    embedding_provider: str = Field(default="openai", description="Embeddings provider: openai|sentence_transformers")
    # Embeddings model for semantic search
    embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", description="Embeddings model")
    # OpenAI embeddings model
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embeddings model")
    # Similarity threshold for retrieval
    embedding_similarity_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")
    # Max related chunks to retrieve
    max_related_chunks: int = Field(default=3, description="Max related chunks")
    # Max tokens in combined context
    max_context_tokens: int = Field(default=2000, description="Max tokens in combined context")


class DedupConfig(BaseModel):
    """Deduplication configuration."""
    # Similarity threshold for deduplication
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for dedup")


class PathsConfig(BaseModel):
    """Filesystem paths configuration."""
    project_root: Path
    input_docs_dir: Path
    output_dir: Path
    database_dir: Path
    examples_dir: Path
    prompts_dir: Path
    database_path: Path


class ConfigBundle(BaseModel):
    """Aggregated configuration bundle with per-agent models."""
    openai: OpenAIConfig
    agent_z: AgentZConfig
    agent_b: AgentBConfig
    agent_c: AgentCConfig
    embeddings: EmbeddingsConfig
    dedup: DedupConfig
    paths: PathsConfig
