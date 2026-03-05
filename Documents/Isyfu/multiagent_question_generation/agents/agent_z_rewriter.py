"""Agente Z: Document Rewriter con StateGraph y Agente ReAct

Arquitectura moderna usando LangGraph:
- StateGraph para flujo de nodos
- Análisis 100% LLM (sin heurísticas)
- Agente ReAct con herramientas para navegación de chunks
- Runtime context injection para capturar chunks en herramientas
- Análisis de coherencia semántica avanzada
- Sistema de caché para optimización de costos
- Logging detallado de decisiones
"""

import logging
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import uuid4

# LangChain / LangGraph Imports
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Imports del proyecto
from models.chunk import Chunk
from utils.loaders import load_document
from utils.rewrite_pattern_registry import RewritePatternRegistry
from config.settings import get_settings
from utils.prompt_loader import load_prompt_text, render_prompt


# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================

def setup_logger(name: str = "AgentZ", level: int = logging.INFO) -> logging.Logger:
    """Configura logger con formato detallado."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '\n%(asctime)s | %(name)s | %(levelname)s\n%(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger("AgentZ")


def log_separator(title: str = "", char: str = "=", width: int = 70):
    """Imprime separador visual."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


# ==========================================
# 1. MODELOS DE DATOS (PYDANTIC)
# ==========================================

class ChunkTypeEnum(str, Enum):
    """Tipos de chunk identificados."""
    COMPLETE_ARTICLE = "complete_article"
    COMPLETE_SECTION = "complete_section"
    PARTIAL_CONTENT = "partial_content"
    METADATA = "metadata"
    GARBAGE = "garbage"


class ChunkActionEnum(str, Enum):
    """Acciones sugeridas para coordinar contexto."""
    KEEP = "keep"
    MERGE_NEXT = "merge_next"
    MERGE_PREV = "merge_prev"
    DISCARD = "discard"


class CoherenceAnalysis(BaseModel):
    """Resultado del análisis de coherencia de un chunk."""
    is_coherent: bool = Field(description="Si el chunk tiene contexto suficiente para preguntas de calidad")
    chunk_type: ChunkTypeEnum = Field(description="Tipo de chunk identificado")
    confidence: float = Field(description="Confianza del análisis (0-1)", ge=0, le=1)
    reason: str = Field(description="Razón de la evaluación")
    starts_cleanly: bool = Field(description="Si empieza en un punto natural")
    ends_cleanly: bool = Field(description="Si termina en un punto natural")
    suggested_action: str = Field(description="Acción sugerida: keep, merge_next, merge_prev, discard")


class ChunkContextDecision(BaseModel):
    """Decisión de coordinación para asegurar contexto completo."""
    action: ChunkActionEnum = Field(description="Acción a tomar con el chunk")
    reasoning: str = Field(description="Razón breve de la decisión")
    has_article: bool = Field(description="Si el chunk contiene artículos legales")
    is_complete_article: bool = Field(description="Si contiene al menos un artículo completo")
    is_sufficient_context: bool = Field(description="Si el contexto es suficiente para una pregunta de calidad")
    needs_rewrite: bool = Field(
        default=False,
        description="Si el chunk necesita reescritura para limpiar o mejorar cohesión"
    )
    rewrite_reason: str = Field(
        default="",
        description="Razón de la reescritura solicitada"
    )
    rewrite_type: str = Field(
        default="none",
        description="Tipo de reescritura: clean, cohesion, both, none"
    )


class RewriteResult(BaseModel):
    """Resultado de limpieza y reescritura controlada."""
    cleaned_text: str = Field(description="Texto limpio, continuo y fiel al contenido legal")


class DocumentStructure(BaseModel):
    """Estructura detectada del documento."""
    has_articles: bool = Field(description="Si tiene artículos legales")
    has_chapters: bool = Field(description="Si tiene capítulos")
    has_sections: bool = Field(description="Si tiene secciones/títulos")
    article_count: int = Field(description="Número de artículos detectados")
    chapter_count: int = Field(description="Número de capítulos detectados")
    recommended_separators: List[str] = Field(description="Separadores recomendados")


class ChunkQualityMetrics(BaseModel):
    """Métricas de calidad de un chunk."""
    token_count: int
    word_count: int
    has_complete_sentences: bool
    has_legal_references: bool
    coherence_score: float = Field(ge=0, le=1)


class ChunkCutDecision(BaseModel):
    """Decisión sobre dónde cortar un chunk."""
    chars_to_remove: int = Field(description="Caracteres a eliminar del final del chunk", ge=0)
    reasoning: str = Field(description="Explicación de por qué se toma esta decisión")
    chunk_ends_cleanly: bool = Field(description="Si el chunk termina limpiamente sin cortar")
    recommended_action: str = Field(description="keep_as_is, cut_at_boundary, needs_review")


# ==========================================
# 2. ESTADO DEL AGENTE (STATE)
# ==========================================

class AgentZState(TypedDict):
    """Estado compartido entre nodos del grafo."""
    # Input
    document_path: Path
    topic: int

    # Procesamiento
    raw_content: str
    document_structure: Optional[DocumentStructure]
    preliminary_chunks: List[str]
    coherence_analyses: List[CoherenceAnalysis]
    rewrite_flags: List[bool]
    rewrite_reasons: List[str]
    rewrite_types: List[str]
    rewrite_errors: List[str]
    split_mode: str
    limit_chunks: int

    # Output
    final_chunks: List[Chunk]
    quality_metrics: Dict[str, Any]

    # Tracking
    decisions_log: List[Dict[str, Any]]
    node_count: int


# ==========================================
# 3. FUNCIONES DE UTILIDAD
# ==========================================

def get_llm(structured_output=None):
    """Obtiene LLM usando la factory del proyecto."""
    from utils.llm_factory import create_llm

    settings = get_settings()
    # SIEMPRE usar timeout para evitar cuelgues
    timeout = getattr(settings, "agent_llm_timeout_seconds", None) or 60
    extra_kwargs = {}
    if settings.agent_reasoning_effort and settings.openai_model.startswith("gpt-5"):
        extra_kwargs["reasoning"] = {"effort": settings.agent_reasoning_effort}

    llm = create_llm(
        model=settings.openai_model,
        temperature=0,
        max_tokens=10000,
        timeout=timeout,
        **extra_kwargs
    )

    if structured_output:
        return llm.with_structured_output(structured_output)
    return llm


def get_rewrite_llm(structured_output=None):
    """Obtiene LLM rápido para reescritura selectiva."""
    from utils.llm_factory import create_llm

    settings = get_settings()
    model = settings.rewriter_model or settings.openai_model
    extra_kwargs = {}
    if settings.rewriter_reasoning_effort and model.startswith("gpt-5"):
        extra_kwargs["reasoning"] = {"effort": settings.rewriter_reasoning_effort}
    llm = create_llm(
        model=model,
        temperature=settings.rewriter_temperature,
        max_tokens=settings.rewriter_max_tokens,
        timeout=settings.rewriter_timeout_seconds or None,
        **extra_kwargs
    )

    if structured_output:
        return llm.with_structured_output(structured_output)
    return llm


_REWRITE_PATTERN_REGISTRY: Optional[RewritePatternRegistry] = None


def get_rewrite_pattern_registry() -> Optional[RewritePatternRegistry]:
    """Carga el registro de patrones aprendidos para limpieza."""
    global _REWRITE_PATTERN_REGISTRY
    settings = get_settings()
    if not settings.rewriter_pattern_enable:
        return None
    if _REWRITE_PATTERN_REGISTRY is not None:
        return _REWRITE_PATTERN_REGISTRY

    registry_path = None
    if settings.rewriter_pattern_path:
        candidate = str(settings.rewriter_pattern_path).strip()
        if candidate not in ("", "."):
            registry_path = Path(candidate)
    if registry_path is None:
        registry_path = settings.output_dir / "patterns" / "agent_z_rewrite_patterns.json"

    _REWRITE_PATTERN_REGISTRY = RewritePatternRegistry(
        path=registry_path,
        min_hits=settings.rewriter_pattern_min_hits,
        min_len=settings.rewriter_pattern_min_len,
        max_len=settings.rewriter_pattern_max_len
    )
    return _REWRITE_PATTERN_REGISTRY


def add_decision_log(state: AgentZState, node: str, decision: str, details: Dict = None) -> List[Dict]:
    """Añade entrada al log de decisiones."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "node": node,
        "decision": decision,
        "details": details or {}
    }

    decisions_log = state.get("decisions_log", [])
    decisions_log.append(log_entry)
    return decisions_log


def _split_by_articles(text: str) -> List[str]:
    """Divide el texto por marcadores de articulo para mantenerlos completos."""
    article_pattern = re.compile(
        r'(?im)(?:^|\n)\s*((?:art[íi]culo|art\.?)\s+\d+[\wº°.-]*)'
    )
    matches = list(article_pattern.finditer(text))
    if not matches:
        return []

    starts = [m.start(1) for m in matches]
    segments = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

    # Si hay prefacio corto, lo adjuntamos al primer articulo
    prefix = text[:starts[0]].strip()
    if prefix and segments:
        max_prefix_chars = 400
        if len(prefix) <= max_prefix_chars:
            segments[0] = f"{prefix}\n\n{segments[0]}"
        else:
            segments.insert(0, prefix)

    return segments


def _count_article_headers(text: str) -> int:
    if not text:
        return 0
    pattern = re.compile(r'(?im)(?:^|\n)\s*(?:art[íi]culo|art\.?)\s+\d+[\wº°.-]*')
    return len(pattern.findall(text))


# ==========================================
# 3.1. DETECCIÓN DE LEYES EN CONTEXTO
# ==========================================

_LAW_TITLE_PATTERNS = [
    re.compile(
        r"(Constituci[oó]n\s+Espa[nñ]ola(?:\s+de\s+\d{4})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Ley\s+Org[aá]nica\s+\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Ley\s+\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Real\s+Decreto\s+Legislativo\s+\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Real\s+Decreto-ley\s+\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Real\s+Decreto\s+\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Reglamento\s*\(UE\)\s*\d+\s*/\s*\d{4}(?:\s*,\s*de\s+[^\n\.]{0,160})?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Ley\s+de\s+Enjuiciamiento\s+(?:Criminal|Civil))",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Código\s+(?:Penal|Civil|de\s+Comercio|Procesal|Tributario|Aduanero|Militar|Penal\s+Militar|de\s+Justicia\s+Militar))",
        flags=re.IGNORECASE
    ),
    # Tratados internacionales
    re.compile(
        r"(Carta\s+de\s+las\s+Naciones\s+Unidas)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Declaración\s+Universal\s+de\s+(?:los\s+)?Derechos\s+Humanos)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Convenio\s+(?:Europeo\s+)?para\s+la\s+Protección\s+de\s+(?:los\s+)?(?:DDHH|Derechos\s+Humanos)(?:\s+y\s+(?:las\s+)?[Ll]ibertades\s+[Ff]undamentales)?)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Carta\s+(?:de\s+los\s+)?Derechos\s+Fundamentales\s+de\s+la\s+Unión\s+Europea)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"(Pacto\s+Internacional\s+de\s+Derechos\s+(?:Civiles\s+y\s+Políticos|Económicos[,\s]+[Ss]ociales\s+y\s+[Cc]ulturales))",
        flags=re.IGNORECASE
    ),
]

_LAW_LABEL_PATTERN = re.compile(r"(?im)^\s*(ley|norma)\s+(referida|aplicable)\s*:")


def _normalize_law_title(title: str) -> str:
    if not title:
        return ""
    normalized = re.sub(r"\s*/\s*", "/", title)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.strip(" ,;:-")


def _find_law_title_in_text(text: str, max_chars: int = 0) -> Optional[str]:
    if not text:
        return None
    haystack = text[:max_chars] if max_chars and len(text) > max_chars else text
    for pattern in _LAW_TITLE_PATTERNS:
        match = pattern.search(haystack)
        if match:
            return _normalize_law_title(match.group(1))
    return None


def _find_law_title_in_filename(doc_path: Path) -> Optional[str]:
    if not doc_path:
        return None
    stem = doc_path.stem or ""
    if not stem:
        return None
    normalized = re.sub(r"[_-]+", " ", stem)
    normalized = re.sub(r"(?i)leyorganica", "ley organica", normalized)
    normalized = re.sub(r"(\d{1,3})\s+(\d{4})", r"\1/\2", normalized)
    normalized = re.sub(r"(\d{1,3})[-_](\d{4})", r"\1/\2", normalized)
    return _find_law_title_in_text(normalized)


def _detect_law_with_llm(content: str) -> Optional[str]:
    """Usa LLM para detectar la ley principal del documento."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from config.settings import get_settings

    try:
        settings = get_settings()
        # Usar el modelo configurado (rápido, sin reasoning)
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            max_tokens=100,  # Solo necesitamos el nombre de la ley
            api_key=settings.openai_api_key,
            timeout=30  # Timeout de 30 segundos para evitar cuelgues
        )

        excerpt = content[:1500]

        # Prompt muy simple y directo
        response = llm.invoke([
            HumanMessage(content=f"""Extrae SOLO el nombre de la ley principal de este texto.
Responde UNICAMENTE con el nombre (ej: "Ley Organica 3/2007" o "Carta de las Naciones Unidas").
Si no hay ley clara, responde "Desconocida".

Texto:
{excerpt}

Ley principal:""")
        ])

        law_name = response.content.strip().strip('"').strip("'")
        if law_name and law_name.lower() != "desconocida" and len(law_name) < 100:
            print(f"   🔍 LLM detectó ley: {law_name}")
            return law_name

    except Exception as e:
        print(f"   ⚠️ Error LLM detección ley: {e}")

    return None


# Caché para evitar llamadas repetidas a la API
_DOCUMENT_LAW_CACHE: Dict[str, Optional[str]] = {}

def _infer_document_law(doc_path: Path, raw_content: str = "") -> Optional[str]:
    """Infiere la ley principal del documento usando SOLO LLM (con caché)."""
    global _DOCUMENT_LAW_CACHE

    cache_key = str(doc_path) if doc_path else ""

    # Si ya está en caché, retornar sin llamar a la API
    if cache_key in _DOCUMENT_LAW_CACHE:
        print(f"   ⚡ Ley desde caché: {_DOCUMENT_LAW_CACHE[cache_key]}")
        return _DOCUMENT_LAW_CACHE[cache_key]

    # Solo llamar a la API una vez por documento
    if raw_content:
        print(f"   🔄 Detectando ley del documento (primera vez)...")
        result = _detect_law_with_llm(raw_content)
        _DOCUMENT_LAW_CACHE[cache_key] = result
        print(f"   ✅ Ley cacheada: {result}")
        return result

    return None


def _get_chunk_header_prefix(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""
    match = re.search(r'(?im)(?:^|\n)\s*(?:art[íi]culo|art\.?)\s+\d+[\wº°.-]*', text)
    if match:
        if match.start() == 0:
            return ""
        return text[:min(match.start(), max_chars)]
    return text[:max_chars]


def _extract_law_title_from_chunk_header(text: str, max_chars: int = 800) -> Optional[str]:
    header = _get_chunk_header_prefix(text, max_chars=max_chars)
    if not header.strip():
        return None
    return _find_law_title_in_text(header)


def _chunk_has_article(text: str) -> bool:
    return bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', text or ""))


def _prepend_law_reference(chunk_text: str, law_title: str) -> str:
    if not chunk_text or not law_title:
        return chunk_text
    if _LAW_LABEL_PATTERN.search(chunk_text):
        return chunk_text
    return f"Ley referida: {law_title}\n\n{chunk_text}".strip()


def _find_law_title_in_chunk(text: str, max_chars: int = 1800) -> Optional[str]:
    if not text:
        return None
    header_law = _extract_law_title_from_chunk_header(text, max_chars=min(max_chars, 1200))
    if header_law:
        return header_law
    snippet = text[:max_chars] if max_chars and len(text) > max_chars else text
    return _find_law_title_in_text(snippet)


def _split_article_chunks(chunks: List[str]) -> List[str]:
    """Re-split chunks si contienen múltiples artículos detectados."""
    normalized: List[str] = []
    for chunk in chunks:
        if _count_article_headers(chunk) > 1:
            split_chunks = _split_by_articles(chunk)
            if split_chunks:
                normalized.extend(split_chunks)
                continue
        normalized.append(chunk)
    return normalized


def _extract_article_number(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(
        r'(?im)(?:^|\n)\s*(?:art[íi]culo|art\.?)\s+(\d+)\b',
        text
    )
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _references_previous_article(current_text: str, prev_text: str) -> bool:
    if not current_text:
        return False
    # Referencia explícita al artículo anterior/precedente
    if re.search(
        r'(?i)art[íi]culo(?:s)?\s+(?:anterior(?:es)?|precedente(?:s)?|inmediatamente\s+anterior(?:es)?)',
        current_text
    ):
        return True

    current_num = _extract_article_number(current_text)
    prev_num = _extract_article_number(prev_text) if prev_text else None
    if current_num is None or prev_num is None:
        return False
    if prev_num != current_num - 1:
        return False

    return bool(re.search(rf'(?i)art[íi]culo\s+{prev_num}\b|art\.?\s+{prev_num}\b', current_text))


def _merge_short_chunks(chunks: List[str], min_chars: int, max_chars: int) -> List[str]:
    """Une chunks pequenos para asegurar contexto suficiente."""
    merged = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if not buffer:
            buffer = chunk
            continue

        if len(buffer) < min_chars:
            buffer = f"{buffer}\n\n{chunk}"
            continue

        if len(buffer) + 2 + len(chunk) <= max_chars:
            buffer = f"{buffer}\n\n{chunk}"
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def get_cache_path(doc_path: Path) -> Path:
    """Obtiene la ruta del archivo de caché para un documento.

    Args:
        doc_path: Path del documento original

    Returns:
        Path del archivo JSON de caché en input_docs/rewritten/
    """
    settings = get_settings()
    cache_dir = Path(settings.input_docs_dir) / "rewritten"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Usar el mismo nombre pero con extensión .json
    cache_file = cache_dir / f"{doc_path.stem}.json"
    return cache_file


def get_partial_cache_path(doc_path: Path) -> Path:
    """Ruta de caché parcial (in-progress) para un documento."""
    settings = get_settings()
    cache_dir = Path(settings.input_docs_dir) / "rewritten"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_path.stem}.partial.json"


def get_coord_cache_path(doc_path: Path) -> Path:
    """Ruta de caché para decisiones de coordinación."""
    settings = get_settings()
    cache_dir = Path(settings.input_docs_dir) / "rewritten"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_path.stem}.coord.json"


def load_coord_cache(doc_path: Path) -> Dict[str, dict]:
    """Carga caché de coordinación si existe."""
    cache_path = get_coord_cache_path(doc_path)
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("decisions", {}) if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"No se pudo cargar caché de coordinación: {e}")
        return {}


def save_coord_cache(doc_path: Path, decisions: Dict[str, dict]) -> None:
    """Guarda caché de coordinación."""
    cache_path = get_coord_cache_path(doc_path)
    payload = {
        "document_name": doc_path.name,
        "document_path": str(doc_path),
        "cached_at": datetime.now().isoformat(),
        "decisions": decisions
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_chunks_to_cache(
    chunks: List[Chunk],
    doc_path: Path,
    status: str = "complete"
) -> None:
    """Guarda chunks procesados en el JSON final (con estado)."""
    cache_path = get_cache_path(doc_path)

    # Convertir chunks a diccionarios serializables
    chunks_data = []
    for chunk in chunks:
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "source_document": chunk.source_document,
            "page": chunk.page,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata
        }
        chunks_data.append(chunk_dict)

    # Crear estructura completa con metadata
    cache_data = {
        "document_name": doc_path.name,
        "document_path": str(doc_path),
        "cached_at": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "cache_status": status,
        "chunks": chunks_data
    }

    # Guardar en JSON final
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    if status == "complete":
        partial_path = get_partial_cache_path(doc_path)
        if partial_path.exists():
            try:
                partial_path.unlink()
            except Exception as exc:
                logger.warning(f"No se pudo eliminar caché parcial: {exc}")

    print(f"\n💾 Chunks guardados en caché: {cache_path}")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • Estado: {status}")
    print(f"   • Tamaño: {cache_path.stat().st_size / 1024:.1f} KB\n")


def save_chunks_to_partial_cache(chunks: List[Chunk], doc_path: Path) -> None:
    """Guarda incrementalmente en el JSON final (sin archivos parciales)."""
    save_chunks_to_cache(chunks, doc_path, status="in_progress")


def load_chunks_from_cache(doc_path: Path) -> Optional[List[Chunk]]:
    """Carga chunks desde archivo JSON de caché si existe.

    Args:
        doc_path: Path del documento original

    Returns:
        Lista de chunks si existe caché, None en caso contrario
    """
    cache_path = get_cache_path(doc_path)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        cache_status = cache_data.get("cache_status", "complete")
        if cache_status != "complete":
            print(f"\n⚠️ Caché en progreso detectada ({cache_status}), ignorando: {cache_path}\n")
            return None

        print(f"\n📦 Chunks cargados desde caché: {cache_path}")
        print(f"   • Documento: {cache_data['document_name']}")
        print(f"   • Generado: {cache_data['cached_at']}")
        print(f"   • Total chunks: {cache_data['total_chunks']}")
        print(f"   ⚡ 0 llamadas LLM necesarias (ahorro 100%)\n")

        # Reconstruir objetos Chunk
        chunks = []
        for chunk_dict in cache_data['chunks']:
            chunk = Chunk(
                chunk_id=chunk_dict['chunk_id'],
                content=chunk_dict['content'],
                source_document=chunk_dict['source_document'],
                page=chunk_dict.get('page'),
                token_count=chunk_dict['token_count'],
                metadata=chunk_dict.get('metadata', {})
            )
            chunks.append(chunk)

        return chunks

    except Exception as e:
        logger.warning(f"Error cargando caché, regenerando chunks: {e}")
        return None


def generate_chunks_pdf(chunks: List[Chunk], doc_path: Path) -> Optional[Path]:
    """Genera un PDF coloreado con todos los chunks."""
    if not chunks:
        return None

    try:
        from utils.pdf_visualizer import add_colored_backgrounds_to_pdf
    except Exception as e:
        logger.warning(f"No se pudo importar pdf_visualizer: {e}")
        return None

    try:
        settings = get_settings()
        output_dir = settings.output_dir / "colored_chunks"
        output_path = add_colored_backgrounds_to_pdf(
            chunks,
            doc_path,
            output_dir=output_dir
        )
        print(f"📄 PDF de chunks generado: {output_path}")
        logger.info(f"✓ Colored PDF created: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"No se pudo generar PDF de chunks: {e}")
        print(f"⚠️ No se pudo generar PDF de chunks: {e}")
        return None


# ==========================================
# 4. NODOS DEL GRAFO
# ==========================================

def load_document_node(state: AgentZState) -> dict:
    """
    NODO 1: Carga el documento PDF.
    """
    log_separator("NODO: LOAD DOCUMENT", "═")

    doc_path = state["document_path"]

    print(f"""
📄 CARGANDO DOCUMENTO:
   • Path: {doc_path.name}
   • Tamaño: {doc_path.stat().st_size / 1024 / 1024:.2f} MB
""")

    try:
        doc = load_document(doc_path)
        content = doc["content"]

        word_count = len(content.split())
        char_count = len(content)

        print(f"""
✅ DOCUMENTO CARGADO:
   • Palabras: {word_count:,}
   • Caracteres: {char_count:,}
   • Tipo: {doc["metadata"].get("type", "unknown")}
""")

        return {
            "raw_content": content,
            "decisions_log": add_decision_log(state, "load_document", "success", {
                "word_count": word_count,
                "char_count": char_count
            }),
            "node_count": state.get("node_count", 0) + 1
        }

    except Exception as e:
        logger.error(f"Error cargando documento: {e}")
        return {
            "raw_content": "",
            "decisions_log": add_decision_log(state, "load_document", "error", {"error": str(e)}),
            "node_count": state.get("node_count", 0) + 1
        }


def analyze_structure_node(state: AgentZState) -> dict:
    """
    NODO 2: Analiza la estructura del documento usando LLM.
    Detecta artículos, capítulos, secciones.
    """
    log_separator("NODO: ANALYZE STRUCTURE", "═")

    content = state["raw_content"]

    print(f"""
🔍 ANALIZANDO ESTRUCTURA:
   • Modo: LLM
   • Contenido: {len(content)} caracteres
""")

    # Análisis con LLM
    structure_analyzer = get_llm(structured_output=DocumentStructure)

    prompt = render_prompt(
        load_prompt_text("agent_z_analyze_structure_user"),
        {"content_excerpt": content[:2000]}
    )

    try:
        structure = structure_analyzer.invoke([
            SystemMessage(content=load_prompt_text("agent_z_analyze_structure_system")),
            HumanMessage(content=prompt)
        ])
    except Exception as e:
        logger.error(f"Error en análisis LLM: {e}")
        raise

    print(f"""
📊 ESTRUCTURA DETECTADA:
   • Artículos: {structure.article_count}
   • Capítulos: {structure.chapter_count}
   • Separadores recomendados: {len(structure.recommended_separators)}
""")

    for i, sep in enumerate(structure.recommended_separators, 1):
        print(f"   {i}. {repr(sep)}")

    return {
        "document_structure": structure,
        "decisions_log": add_decision_log(state, "analyze_structure", "completed", {
            "articles": structure.article_count,
            "chapters": structure.chapter_count,
            "separators": len(structure.recommended_separators)
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def _create_chunk_tools(chunk_text: str):
    """
    Crea herramientas dinámicas que capturan el chunk actual.
    Patrón de runtime context injection.
    """
    def inspect_full(_input: str = "") -> str:
        """Inspecciona el chunk completo para entender su contenido."""
        return f"Chunk completo ({len(chunk_text)} caracteres):\n\n{chunk_text}"

    def inspect_end(_input: str = "") -> str:
        """Inspecciona los últimos 300 caracteres del chunk."""
        tail = chunk_text[-300:] if len(chunk_text) > 300 else chunk_text
        return f"Últimos {len(tail)} caracteres:\n\n{tail}"

    def find_boundary(_input: str = "") -> str:
        """Encuentra el último límite de frase limpio en el chunk."""
        import re
        # Buscar límites de frase
        boundaries = list(re.finditer(r'[.!?:]\s+|\n\n', chunk_text))

        if not boundaries:
            return "No se encontró ningún límite de frase en el chunk."

        last = boundaries[-1]
        chars_from_end = len(chunk_text) - last.end()
        text_after = chunk_text[last.end():][:80]

        return f"""Último límite encontrado:
- Posición: {last.end()} de {len(chunk_text)}
- Caracteres desde el final: {chars_from_end}
- Carácter límite: '{chunk_text[last.start():last.end()]}'
- Texto después del límite: '{text_after}...'"""

    # Crear herramientas con el chunk capturado
    tools = [
        Tool(
            name="inspect_chunk_full",
            description="Lee el chunk completo para entender el contexto. No requiere input.",
            func=inspect_full
        ),
        Tool(
            name="inspect_chunk_end",
            description="Lee los últimos 300 caracteres del chunk. No requiere input.",
            func=inspect_end
        ),
        Tool(
            name="find_last_sentence_boundary",
            description="Encuentra el último punto de corte limpio (punto, dos puntos, etc). No requiere input.",
            func=find_boundary
        )
    ]

    return tools


def _ensure_complete_sentences_with_agent(chunk: str, idx: int = 0, total: int = 0) -> str:
    """
    Usa un agente ReAct con herramientas para decidir dónde cortar el chunk.
    El agente puede navegar el chunk llamando herramientas según necesite.
    """
    if not chunk or len(chunk) < 50:
        print(f"      ⚠️ Chunk muy corto ({len(chunk)} chars), sin procesar")
        return chunk

    print(f"      🤖 Iniciando agente ReAct...")

    try:
        from utils.llm_factory import create_llm_for_agents

        # Crear herramientas con el chunk capturado
        tools = _create_chunk_tools(chunk)

        # Crear LLM y agente
        llm = create_llm_for_agents()
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=load_prompt_text("agent_z_chunk_boundary_react")
        )

        # Ejecutar agente
        tail = chunk[-300:] if len(chunk) > 300 else chunk
        result = agent.invoke({
            "messages": [(
                "user",
                f"Chunk {idx}/{total} ({len(chunk)} chars).\n\n"
                f"Últimos {len(tail)} caracteres del chunk:\n{tail}\n\n"
                "Analiza y responde con JSON: {{\"chars_to_remove\": N, \"reasoning\": \"...\"}}"
            )]
        })

        # Extraer respuesta
        final_message = result["messages"][-1].content
        messages = result.get("messages", [])
        if messages:
            print("      🧠 Razonamiento completo (ReAct):")
            for msg in messages:
                role = getattr(msg, "type", None)
                if role != "ai":
                    continue
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        name = call.get("name", "tool")
                        args = call.get("args", {})
                        print(f"      [tool_call] {name} {args}")
                content = getattr(msg, "content", None)
                if content:
                    for line in str(content).splitlines():
                        print(f"      [ai] {line}")

        # Parsear JSON
        import json
        import re

        json_match = re.search(r'\{[^{}]*"chars_to_remove"\s*:\s*\d+[^{}]*\}', final_message)
        if json_match:
            decision = json.loads(json_match.group())
            chars_to_remove = int(decision.get("chars_to_remove", 0))
            reasoning = decision.get("reasoning", "")

            print(f"      🎯 Eliminar: {chars_to_remove} caracteres")
            print(f"      📋 Razón: {reasoning}")

            if chars_to_remove == 0:
                print(f"      ✨ Chunk termina limpiamente")
                return chunk.strip()
            elif 0 < chars_to_remove < len(chunk):
                print(f"      ✂️ Cortando...")
                removed = chunk[-chars_to_remove:]
                print(f"      🗑️ Eliminado: '{removed[:50]}...'")
                return chunk[:-chars_to_remove].strip()

        print(f"      ⚠️ No se pudo parsear, manteniendo completo")
        return chunk.strip()

    except Exception as e:
        logger.error(f"Error en agente: {e}")
        print(f"      ❌ Error: {e}")
        print(f"      🔄 Manteniendo chunk original")
        return chunk.strip()


def split_coherently_node(state: AgentZState) -> dict:
    """
    NODO 3: Divide el documento en chunks coherentes.
    """
    log_separator("NODO: SPLIT COHERENTLY", "═")

    content = state["raw_content"]
    structure = state["document_structure"]
    settings = get_settings()

    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    article_chunks = _split_by_articles(content)
    if article_chunks:
        print(f"""
✂️ DIVIDIENDO EN CHUNKS:
   • Modo: artículos completos
   • Artículos detectados: {len(article_chunks)}
""")
        chunks = [c.strip() for c in article_chunks if c.strip()]
        split_mode = "articles"
    else:
        recommended_separators = structure.recommended_separators if structure else []
        default_separators = ["\n\n", "\n", ". ", " ", ""]
        separators = []
        for sep in recommended_separators + default_separators:
            if sep not in separators:
                separators.append(sep)

        print(f"""
✂️ DIVIDIENDO EN CHUNKS:
   • Modo: texto libre (sin artículos)
   • Chunk size: {chunk_size} caracteres
   • Overlap: {chunk_overlap} caracteres
   • Separadores: {len(separators)} (LLM: {len(recommended_separators)} + fallback)
""")

        # Crear text splitter con separadores optimizados + fallback
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )

        # Dividir
        raw_chunks = text_splitter.split_text(content)
        print(f"📋 Chunks crudos generados: {len(raw_chunks)}")
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        split_mode = "free"

    total_chunks = len(chunks)
    if chunks:
        avg_size = sum(len(c) for c in chunks) // len(chunks)
        min_size = min(len(c) for c in chunks)
        max_size = max(len(c) for c in chunks)
    else:
        avg_size = 0
        min_size = 0
        max_size = 0

    print(f"""
📦 CHUNKS PRELIMINARES:
   • Total: {total_chunks}
   • Promedio: {avg_size} caracteres
   • Mínimo: {min_size} caracteres
   • Máximo: {max_size} caracteres
""")

    return {
        "preliminary_chunks": chunks,
        "split_mode": split_mode,
        "decisions_log": add_decision_log(state, "split_coherently", "completed", {
            "total_chunks": total_chunks,
            "avg_size": avg_size
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def _decide_chunk_context_llm(
    current_chunk: str,
    prev_chunk: str = "",
    next_chunk: str = ""
) -> ChunkContextDecision:
    """Decide si el chunk tiene contexto suficiente o requiere merge."""
    trimmed = current_chunk.strip()
    if not trimmed:
        return ChunkContextDecision(
            action=ChunkActionEnum.DISCARD,
            reasoning="Chunk vacío",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )
    normalized = re.sub(r"[-_\s]+", "", trimmed, flags=re.UNICODE).lower()
    if re.fullmatch(r"p[aá]gina\d+", normalized) or re.fullmatch(r"page\d+", normalized):
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_NEXT if next_chunk else ChunkActionEnum.DISCARD,
            reasoning="Solo contiene marcador de página",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )
    if len(trimmed) < 80:
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_NEXT if next_chunk else ChunkActionEnum.KEEP,
            reasoning="Chunk demasiado corto para contexto suficiente",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=not bool(next_chunk),
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )
    dependency = None
    if prev_chunk:
        dependency = _detect_prev_context_dependency(current_chunk)
    if dependency:
        has_article = bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk))
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_PREV,
            reasoning=f"Referencia anafórica detectada: {dependency}",
            has_article=has_article,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    decider = get_llm(structured_output=ChunkContextDecision)

    prompt = render_prompt(
        load_prompt_text("agent_z_chunk_context_user"),
        {
            "current_chunk": current_chunk,
            "prev_chunk": prev_chunk if prev_chunk else "N/A",
            "next_chunk": next_chunk if next_chunk else "N/A"
        }
    )

    try:
        result = decider.invoke([
            SystemMessage(content=load_prompt_text("agent_z_chunk_context_system")),
            HumanMessage(content=prompt)
        ])
        return result
    except Exception as e:
        logger.error(f"Error en coordinación LLM de chunk: {e}")
        if next_chunk and not re.search(r"[.!?]\s*$", trimmed):
            return ChunkContextDecision(
                action=ChunkActionEnum.MERGE_NEXT,
                reasoning="Fallback por error LLM; texto parece incompleto",
                has_article=bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk)),
                is_complete_article=False,
                is_sufficient_context=False,
                needs_rewrite=False,
                rewrite_reason="",
                rewrite_type="none"
            )
        return ChunkContextDecision(
            action=ChunkActionEnum.KEEP,
            reasoning="Fallback por error LLM",
            has_article=bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk)),
            is_complete_article=False,
            is_sufficient_context=True,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )


def _coord_cache_key(current_chunk: str, prev_chunk: str = "", next_chunk: str = "") -> str:
    """Clave estable para cachear decisiones de coordinación."""
    payload = f"{prev_chunk}\n||\n{current_chunk}\n||\n{next_chunk}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _decide_chunk_context_heuristic(
    current_chunk: str,
    prev_chunk: str = "",
    next_chunk: str = ""
) -> ChunkContextDecision:
    """Heurística rápida para coordinar contexto sin LLM."""
    trimmed = current_chunk.strip()
    if not trimmed:
        return ChunkContextDecision(
            action=ChunkActionEnum.DISCARD,
            reasoning="Chunk vacío (heurístico)",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    normalized = re.sub(r"[-_\s]+", "", trimmed, flags=re.UNICODE).lower()
    if re.fullmatch(r"p[aá]gina\d+", normalized) or re.fullmatch(r"page\d+", normalized):
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_NEXT if next_chunk else ChunkActionEnum.DISCARD,
            reasoning="Marcador de página (heurístico)",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    dependency = _detect_prev_context_dependency(current_chunk) if prev_chunk else None
    if dependency:
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_PREV,
            reasoning=f"Referencia anafórica detectada: {dependency}",
            has_article=bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk)),
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    if len(trimmed) < 120 and next_chunk:
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_NEXT,
            reasoning="Chunk corto, fusionar para contexto (heurístico)",
            has_article=False,
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    ends_clean = bool(re.search(r"[.!?;:]\s*$|\n\n\s*$", trimmed))
    if next_chunk and not ends_clean:
        return ChunkContextDecision(
            action=ChunkActionEnum.MERGE_NEXT,
            reasoning="Final incompleto, fusionar con siguiente (heurístico)",
            has_article=bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk)),
            is_complete_article=False,
            is_sufficient_context=False,
            needs_rewrite=False,
            rewrite_reason="",
            rewrite_type="none"
        )

    return ChunkContextDecision(
        action=ChunkActionEnum.KEEP,
        reasoning="Contexto suficiente (heurístico)",
        has_article=bool(re.search(r'(?im)\b(?:art[íi]culo|art\.?)\s+\d', current_chunk)),
        is_complete_article=False,
        is_sufficient_context=True,
        needs_rewrite=False,
        rewrite_reason="",
        rewrite_type="none"
    )


def _detect_prev_context_dependency(chunk_text: str) -> Optional[str]:
    """Detecta referencias que sugieren dependencia del contexto anterior."""
    if not chunk_text:
        return None

    patterns = [
        r"\besta representación\b",
        r"\besta definici[óo]n\b",
        r"\besta relaci[óo]n\b",
        r"\besta ecuaci[óo]n\b",
        r"\besta fórmula\b",
        r"\besta expresi[óo]n\b",
        r"\besta notaci[óo]n\b",
        r"\besta tabla\b",
        r"\besta figura\b",
        r"\beste gr[aá]fico\b",
        r"\beste concepto\b",
        r"\beste caso\b",
        r"\besta sección\b",
        r"\beste apartado\b",
        r"\beste punto\b",
        r"\blo anterior\b",
        r"\blo anteriormente expuesto\b",
        r"\blo expuesto\b",
        r"\bcomo se ha (indicado|señalado|mencionado|descrito|explicado)\b",
        r"\bcomo se (indic[oó]|señal[oó]|mencion[oó]|describi[oó]|explic[oó])\b",
        r"\btal y como se ha (indicado|señalado|mencionado|descrito|explicado)\b",
        r"\bsegún lo (anterior|expues|indicado|mencionado|descrito|explicado)\b",
        r"\bdicha(s)? (representación|definici[óo]n|relaci[óo]n|ecuaci[óo]n|fórmula|tabla|figura|expresi[óo]n|notaci[óo]n)\b",
        r"\bdicho(s)? (concepto|caso|apartado|punto)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, chunk_text, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def _extract_removed_boundary_lines(
    original_text: str,
    rewritten_text: str,
    boundary_lines: int
) -> List[str]:
    """Extrae líneas removidas cerca de los bordes del chunk."""
    def normalize_lines(text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return lines

    original_lines = normalize_lines(original_text)
    rewritten_lines = set(normalize_lines(rewritten_text))

    if not original_lines:
        return []

    removed = [
        (idx, line)
        for idx, line in enumerate(original_lines)
        if line not in rewritten_lines
    ]

    if not removed:
        return []

    boundary = max(1, boundary_lines)
    last_idx = len(original_lines) - boundary
    boundary_removed = [
        line for idx, line in removed
        if idx < boundary or idx >= last_idx
    ]

    return boundary_removed


def coordinate_context_node(state: AgentZState) -> dict:
    """
    NODO 4: Coordina contexto suficiente usando LLM (merge/keep/discard).
    """
    log_separator("NODO: COORDINATE CONTEXT", "═")

    chunks = state["preliminary_chunks"]
    limit_chunks = int(state.get("limit_chunks") or 0)
    if limit_chunks > 0:
        chunks = chunks[:limit_chunks]
    split_mode = state.get("split_mode", "free")
    if split_mode == "articles":
        normalized_chunks = _split_article_chunks(chunks)
        if len(normalized_chunks) != len(chunks):
            print(f"   📎 Normalizados por artículos: {len(chunks)} → {len(normalized_chunks)}")
        chunks = normalized_chunks

    coordination_mode = (get_settings().rewriter_coordination_mode or "llm").lower()
    doc_path = state.get("document_path")
    settings = get_settings()
    coord_cache_enable = bool(getattr(settings, "rewriter_coordination_cache_enable", True))
    coord_cache_every = int(getattr(settings, "rewriter_coordination_cache_every", 20) or 20)
    partial_save_every = int(getattr(settings, "rewriter_partial_save_every", 5) or 0)
    coord_cache = load_coord_cache(doc_path) if coord_cache_enable and doc_path else {}
    coord_cache_dirty = False
    decision_count = 0

    print(f"""
🤖 COORDINANDO CONTEXTO:
   • Chunks de entrada: {len(chunks)}
   • Modo: {coordination_mode}
""")

    merged_chunks: List[str] = []
    merge_next_count = 0
    merge_prev_count = 0
    discard_count = 0
    settings = get_settings()
    max_merge_steps = settings.max_merge_steps
    max_merged_chars = settings.max_merged_chars
    rewrite_flags: List[bool] = []
    rewrite_reasons: List[str] = []
    rewrite_types: List[str] = []

    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        i += 1

        if not current:
            continue

        merged_current = current
        merge_steps = 0
        needs_rewrite = False
        rewrite_reason_parts: List[str] = []
        rewrite_type_parts: set[str] = set()

        while True:
            prev_chunk = merged_chunks[-1] if merged_chunks else ""
            next_chunk = chunks[i].strip() if i < len(chunks) else ""

            if coordination_mode == "heuristic":
                decision = _decide_chunk_context_heuristic(merged_current, prev_chunk, next_chunk)
            else:
                cache_key = _coord_cache_key(merged_current, prev_chunk, next_chunk)
                cached = coord_cache.get(cache_key) if coord_cache else None
                if cached:
                    decision = ChunkContextDecision(**cached)
                else:
                    decision = _decide_chunk_context_llm(merged_current, prev_chunk, next_chunk)
                    if coord_cache_enable:
                        coord_cache[cache_key] = decision.model_dump()
                        coord_cache_dirty = True
                        decision_count += 1
                        if coord_cache_every > 0 and decision_count % coord_cache_every == 0 and doc_path:
                            try:
                                save_coord_cache(doc_path, coord_cache)
                                coord_cache_dirty = False
                            except Exception as e:
                                logger.warning(f"No se pudo guardar caché de coordinación: {e}")
            action = decision.action
            if split_mode == "articles" and action in (
                ChunkActionEnum.MERGE_NEXT,
                ChunkActionEnum.MERGE_PREV
            ):
                allow_prev = action == ChunkActionEnum.MERGE_PREV and _references_previous_article(
                    merged_current,
                    prev_chunk
                )
                if not allow_prev:
                    print("      ⚠️ Modo artículos: no se fusiona salvo referencia al artículo anterior")
                    action = ChunkActionEnum.KEEP

            print(
                f"   → Acción: {action.value} | "
                f"Artículo completo: {decision.is_complete_article} | "
                f"Contexto suficiente: {decision.is_sufficient_context} | "
                f"Reescritura: {'sí' if decision.needs_rewrite else 'no'} ({decision.rewrite_type})"
            )
            print(f"      💬 Razón: {decision.reasoning}")
            if decision.needs_rewrite and decision.rewrite_reason:
                print(f"      ✍️ Reescritura: {decision.rewrite_reason}")

            if decision.needs_rewrite:
                needs_rewrite = True
                if decision.rewrite_reason:
                    rewrite_reason_parts.append(decision.rewrite_reason)
                if decision.rewrite_type and decision.rewrite_type != "none":
                    rewrite_type_parts.add(decision.rewrite_type)

            if action == ChunkActionEnum.MERGE_PREV:
                if not merged_chunks:
                    if next_chunk and merge_steps < max_merge_steps:
                        action = ChunkActionEnum.MERGE_NEXT
                    else:
                        action = ChunkActionEnum.KEEP
                elif max_merged_chars and (len(merged_chunks[-1]) + len(merged_current)) > max_merged_chars:
                    print("      ⚠️ Límite de tamaño alcanzado, manteniendo actual")
                    action = ChunkActionEnum.KEEP
                elif merge_steps >= max_merge_steps:
                    print("      ⚠️ Límite de merges alcanzado, manteniendo actual")
                    action = ChunkActionEnum.KEEP
                else:
                    merge_prev_count += 1
                    merged_current = f"{merged_chunks.pop()}\n\n{merged_current}"
                    prev_rewrite_flag = rewrite_flags.pop() if rewrite_flags else False
                    prev_rewrite_reason = rewrite_reasons.pop() if rewrite_reasons else ""
                    prev_rewrite_type = rewrite_types.pop() if rewrite_types else "none"
                    if prev_rewrite_flag:
                        needs_rewrite = True
                        if prev_rewrite_reason:
                            rewrite_reason_parts.append(prev_rewrite_reason)
                        if prev_rewrite_type and prev_rewrite_type != "none":
                            rewrite_type_parts.add(prev_rewrite_type)
                    merge_steps += 1
                    continue

            if action == ChunkActionEnum.MERGE_NEXT:
                if not next_chunk:
                    print("      ⚠️ Sin chunk siguiente, manteniendo actual")
                    action = ChunkActionEnum.KEEP
                elif max_merged_chars and (len(merged_current) + len(next_chunk)) > max_merged_chars:
                    print("      ⚠️ Límite de tamaño alcanzado, manteniendo actual")
                    action = ChunkActionEnum.KEEP
                elif merge_steps >= max_merge_steps:
                    print("      ⚠️ Límite de merges alcanzado, manteniendo actual")
                    action = ChunkActionEnum.KEEP
                else:
                    merge_next_count += 1
                    merged_current = f"{merged_current}\n\n{next_chunk}"
                    i += 1
                    merge_steps += 1
                    continue

            if action == ChunkActionEnum.DISCARD:
                discard_count += 1
                merged_current = ""
            else:
                merged_chunks.append(merged_current)
                rewrite_flags.append(needs_rewrite)
                rewrite_reasons.append("; ".join(rewrite_reason_parts).strip())
                rewrite_types.append("+".join(sorted(rewrite_type_parts)) if rewrite_type_parts else "none")
                if doc_path and partial_save_every > 0 and len(merged_chunks) % partial_save_every == 0:
                    try:
                        partial_chunks = _build_fallback_chunks(
                            preliminary_chunks=merged_chunks,
                            doc_path=doc_path,
                            topic=state.get("topic", 0),
                            analyses=state.get("coherence_analyses") or [],
                            rewrite_flags=rewrite_flags,
                            rewrite_reasons=rewrite_reasons,
                            rewrite_types=rewrite_types,
                            rewrite_errors=[],
                            raw_content=state.get("raw_content", "")
                        )
                        save_chunks_to_partial_cache(partial_chunks, doc_path)
                    except Exception as e:
                        logger.warning(f"No se pudo guardar caché parcial en coordinación: {e}")

            break

    if coord_cache_dirty and doc_path:
        try:
            save_coord_cache(doc_path, coord_cache)
        except Exception as e:
            logger.warning(f"No se pudo guardar caché de coordinación final: {e}")

    print(f"""
✅ COORDINACIÓN COMPLETADA:
   • Chunks finales: {len(merged_chunks)}
   • Merges hacia adelante: {merge_next_count}
   • Merges hacia atrás: {merge_prev_count}
   • Descartados: {discard_count}
""")

    return {
        "preliminary_chunks": merged_chunks,
        "rewrite_flags": rewrite_flags,
        "rewrite_reasons": rewrite_reasons,
        "rewrite_types": rewrite_types,
        "decisions_log": add_decision_log(state, "coordinate_context", "completed", {
            "input_chunks": len(chunks),
            "output_chunks": len(merged_chunks),
            "merge_next": merge_next_count,
            "merge_prev": merge_prev_count,
            "discarded": discard_count
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def _clean_chunk_metadata(chunk_text: str) -> str:
    """Limpia metadatos de PDF en el chunk usando filtro determinista."""
    settings = get_settings()
    if not settings.filter_metadata:
        return chunk_text

    registry = get_rewrite_pattern_registry()
    if registry:
        cleaned_text, matched = registry.apply(chunk_text)
        if matched:
            print(f"      🧹 Patrones aprendidos aplicados: {len(matched)}")
        chunk_text = cleaned_text

    try:
        from utils.metadata_filter import filter_metadata_from_chunk
        result = filter_metadata_from_chunk(chunk_text, min_ratio=settings.min_substantive_ratio)
    except Exception as e:
        logger.warning(f"Error limpiando metadatos, manteniendo original: {e}")
        return chunk_text

    if result.clean_content:
        return result.clean_content
    return chunk_text


def _rewrite_chunk_llm(chunk_text: str) -> RewriteResult:
    """Reescribe limpiando ruido sin alterar el contenido legal."""
    rewriter = get_rewrite_llm(structured_output=RewriteResult)

    prompt = render_prompt(
        load_prompt_text("agent_z_rewrite_chunk_user"),
        {"chunk_text": chunk_text}
    )

    result = rewriter.invoke([
        SystemMessage(content=load_prompt_text("agent_z_rewrite_chunk_system")),
        HumanMessage(content=prompt)
    ])
    return result


def _is_length_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "length limit" in message or "context length" in message


def _split_text_for_rewrite(text: str, max_chars: int) -> List[str]:
    """Divide texto en partes <= max_chars usando separadores naturales."""
    if max_chars <= 0:
        return [text] if text.strip() else []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
    return parts


def _rewrite_chunk_llm_autosplit(
    chunk_text: str,
    max_chars: int,
    min_chars: int = 2000
) -> tuple[str, List[tuple[str, str]], str, int]:
    """Reescribe con auto-split si el texto es largo o si el LLM se corta."""
    if not chunk_text.strip():
        return "", [], "", 0

    parts = [chunk_text]
    if max_chars > 0 and len(chunk_text) > max_chars:
        parts = _split_text_for_rewrite(chunk_text, max_chars)

    autosplit_used = len(parts) > 1
    rewritten_parts: List[str] = []
    pairs: List[tuple[str, str]] = []
    error_label = ""
    total_parts = 0

    for part in parts:
        total_parts += 1
        try:
            rewritten = _rewrite_chunk_llm(part).cleaned_text.strip()
            if not rewritten:
                rewritten = part.strip()
        except Exception as e:
            if _is_length_limit_error(e) and max_chars > min_chars and len(part) > min_chars:
                smaller = max(max_chars // 2, min_chars)
                nested, nested_pairs, nested_error, nested_parts = _rewrite_chunk_llm_autosplit(
                    part,
                    smaller,
                    min_chars
                )
                if nested_parts:
                    total_parts += nested_parts - 1
                if nested_pairs:
                    pairs.extend(nested_pairs)
                rewritten_parts.append(nested)
                if nested_error:
                    error_label = (
                        nested_error if not error_label else f"{error_label},{nested_error}"
                    )
                continue

            if _is_length_limit_error(e):
                error_label = "length_limit" if not error_label else f"{error_label},length_limit"
            else:
                error_label = "rewrite_error" if not error_label else f"{error_label},rewrite_error"
            rewritten = part.strip()

        rewritten_parts.append(rewritten)
        pairs.append((part, rewritten))

    combined = "\n\n".join([p for p in rewritten_parts if p.strip()])

    if autosplit_used:
        error_label = "autosplit" if not error_label else f"autosplit,{error_label}"

    return combined, pairs, error_label, total_parts


def clean_and_rewrite_node(state: AgentZState) -> dict:
    """
    NODO 5: Limpia metadatos sin reescritura.
    """
    log_separator("NODO: CLEAN & REWRITE", "═")

    chunks = state["preliminary_chunks"]
    limit_chunks = int(state.get("limit_chunks") or 0)
    if limit_chunks > 0:
        chunks = chunks[:limit_chunks]
        rewrite_flags = (state.get("rewrite_flags") or [])[:limit_chunks]
        rewrite_reasons = (state.get("rewrite_reasons") or [])[:limit_chunks]
        rewrite_types = (state.get("rewrite_types") or [])[:limit_chunks]
    else:
        rewrite_flags = state.get("rewrite_flags") or []
        rewrite_reasons = state.get("rewrite_reasons") or []
        rewrite_types = state.get("rewrite_types") or []
    doc_path = state.get("document_path")
    doc_id = doc_path.stem if doc_path else ""
    settings = get_settings()

    print(f"""
🧹 LIMPIANDO METADATOS (reescritura selectiva):
   • Chunks de entrada: {len(chunks)}
""")

    def _rewrite_worker(text: str) -> dict:
        try:
            rewritten, pairs, rewrite_note, parts = _rewrite_chunk_llm_autosplit(
                text,
                max_input_chars,
                min_split_chars
            )
            if not rewritten:
                rewritten = text.strip()
            return {
                "rewritten": rewritten.strip(),
                "pairs": pairs,
                "note": rewrite_note,
                "parts": parts
            }
        except Exception as exc:
            note = "length_limit" if _is_length_limit_error(exc) else "rewrite_error"
            return {
                "rewritten": text.strip(),
                "pairs": [],
                "note": note,
                "parts": 0,
                "error": str(exc)
            }

    rewritten_chunks: List[str] = []
    cleaned_count = 0
    rewritten_count = 0
    rewrite_errors: List[str] = []
    max_input_chars = settings.rewriter_max_input_chars
    if settings.rewriter_max_tokens:
        token_limit = settings.rewriter_max_tokens * 4
        if not max_input_chars or max_input_chars > token_limit:
            max_input_chars = token_limit
    min_split_chars = max(2000, (max_input_chars // 4) if max_input_chars else 2000)

    entries: List[dict] = []
    force_rewrite = bool(getattr(settings, "rewriter_force_rewrite", False))
    for idx, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue

        should_rewrite = rewrite_flags[idx - 1] if idx - 1 < len(rewrite_flags) else False
        rewrite_reason = rewrite_reasons[idx - 1] if idx - 1 < len(rewrite_reasons) else ""
        rewrite_type = rewrite_types[idx - 1] if idx - 1 < len(rewrite_types) else "none"
        if force_rewrite:
            should_rewrite = True
            if not rewrite_reason:
                rewrite_reason = "force_rewrite"
            if rewrite_type == "none":
                rewrite_type = "rewrite"

        rewrite_error = ""
        if should_rewrite:
            print(f"\n   ✍️ Reescribiendo chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
            if rewrite_reason:
                print(f"      📋 Motivo: {rewrite_reason} ({rewrite_type})")
        else:
            print(f"\n   🧽 Limpiando chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        cleaned = _clean_chunk_metadata(chunk)
        if cleaned != chunk:
            cleaned_count += 1
            print(f"      🧽 Metadatos limpiados ({len(cleaned)} chars)")

        entries.append({
            "index": idx,
            "cleaned": cleaned,
            "should_rewrite": should_rewrite,
            "rewrite_reason": rewrite_reason,
            "rewrite_type": rewrite_type
        })

    rewrite_results: Dict[int, dict] = {}
    rewrite_tasks = [entry for entry in entries if entry["should_rewrite"]]

    if rewrite_tasks:
        # Ejecutar en serie para evitar paralelismo
        for entry in rewrite_tasks:
            rewrite_results[entry["index"]] = _rewrite_worker(entry["cleaned"])

    partial_save_every = int(getattr(settings, "rewriter_partial_save_every", 5) or 0)

    for entry in entries:
        idx = entry["index"]
        cleaned = entry["cleaned"]
        should_rewrite = entry["should_rewrite"]
        rewrite_reason = entry["rewrite_reason"]
        rewrite_type = entry["rewrite_type"]
        rewrite_error = ""

        if should_rewrite:
            result = rewrite_results.get(idx, {})
            rewritten = (result.get("rewritten") or "").strip()
            pairs = result.get("pairs") or []
            rewrite_note = result.get("note") or ""
            parts = result.get("parts") or 0
            if rewrite_note:
                rewrite_error = rewrite_note
                if "autosplit" in rewrite_note:
                    print(f"      ✂️ Auto-split en {parts} partes")
                if "length_limit" in rewrite_note:
                    logger.warning(f"Reescritura truncada en chunk {idx}: {rewrite_note}")
                    print("      ⚠️ Reescritura truncada, usando auto-split")
                if "timeout" in rewrite_note:
                    logger.warning(f"Timeout en reescritura chunk {idx}: {rewrite_note}")
                    print("      ⚠️ Timeout en reescritura, usando limpio")
                if "rewrite_error" in rewrite_note:
                    logger.warning(f"Error en reescritura chunk {idx}: {rewrite_note}")
                    print("      ⚠️ Error en reescritura, usando limpio en partes")

            if rewritten:
                rewritten_chunks.append(rewritten)
                rewritten_count += 1
                print(f"      ✅ Reescrito ({len(rewritten)} chars)")
                if settings.rewriter_pattern_enable and (
                    "clean" in rewrite_type or rewrite_type == "both"
                ):
                    registry = get_rewrite_pattern_registry()
                    if registry:
                        learned_total = 0
                        for original, rewritten_part in pairs:
                            removed_lines = _extract_removed_boundary_lines(
                                original,
                                rewritten_part,
                                settings.rewriter_pattern_boundary_lines
                            )
                            learned_total += registry.learn_from_removed_lines(
                                removed_lines,
                                doc_id,
                                rewrite_reason
                            )
                        if learned_total:
                            print(f"      🧠 Patrones aprendidos: {learned_total}")
            else:
                rewritten_chunks.append(cleaned.strip())
                print("      ⚠️ Reescritura vacía, usando limpio")
        else:
            rewritten_chunks.append(cleaned.strip())
        rewrite_errors.append(rewrite_error)
        if doc_path and partial_save_every > 0 and len(rewritten_chunks) % partial_save_every == 0:
            try:
                partial_chunks = _build_fallback_chunks(
                    preliminary_chunks=rewritten_chunks,
                    doc_path=doc_path,
                    topic=state.get("topic", 0),
                    analyses=state.get("coherence_analyses") or [],
                    rewrite_flags=rewrite_flags,
                    rewrite_reasons=rewrite_reasons,
                    rewrite_types=rewrite_types,
                    rewrite_errors=rewrite_errors,
                    raw_content=state.get("raw_content", "")
                )
                save_chunks_to_partial_cache(partial_chunks, doc_path)
            except Exception as e:
                logger.warning(f"No se pudo guardar caché parcial incremental: {e}")

    print(f"""
✅ LIMPIEZA COMPLETADA (reescritura selectiva):
   • Chunks finales: {len(rewritten_chunks)}
   • Chunks con metadatos limpiados: {cleaned_count}
   • Chunks reescritos: {rewritten_count}
""")

    # Guardado parcial inmediato (in-progress)
    if doc_path and rewritten_chunks:
        try:
            partial_chunks = _build_fallback_chunks(
                preliminary_chunks=rewritten_chunks,
                doc_path=doc_path,
                topic=state.get("topic", 0),
                analyses=state.get("coherence_analyses") or [],
                rewrite_flags=rewrite_flags,
                rewrite_reasons=rewrite_reasons,
                rewrite_types=rewrite_types,
                rewrite_errors=rewrite_errors,
                raw_content=state.get("raw_content", "")
            )
            save_chunks_to_partial_cache(partial_chunks, doc_path)
        except Exception as e:
            logger.warning(f"No se pudo guardar caché parcial: {e}")

    return {
        "preliminary_chunks": rewritten_chunks,
        "rewrite_flags": rewrite_flags,
        "rewrite_reasons": rewrite_reasons,
        "rewrite_types": rewrite_types,
        "rewrite_errors": rewrite_errors,
        "decisions_log": add_decision_log(state, "clean_and_rewrite", "completed", {
            "input_chunks": len(chunks),
            "output_chunks": len(rewritten_chunks),
            "cleaned": cleaned_count,
            "rewritten": rewritten_count,
            "limit_chunks": limit_chunks
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def validate_chunks_node(state: AgentZState) -> dict:
    """
    NODO 6: Genera análisis básico sin LLM.
    """
    log_separator("NODO: VALIDATE CHUNKS", "═")

    chunks = state["preliminary_chunks"]

    print(f"""
✅ VALIDANDO COHERENCIA:
   • Chunks a validar: {len(chunks)}
   • Modo: SIN LLM
""")

    analyses = []

    for i, chunk_text in enumerate(chunks):
        is_empty = not chunk_text.strip()
        analysis = CoherenceAnalysis(
            is_coherent=not is_empty,
            chunk_type=ChunkTypeEnum.GARBAGE if is_empty else ChunkTypeEnum.PARTIAL_CONTENT,
            confidence=0.0,
            reason="LLM validation disabled",
            starts_cleanly=False,
            ends_cleanly=False,
            suggested_action="discard" if is_empty else "keep"
        )
        analyses.append(analysis)

    if chunks:
        coherent_count = sum(1 for a in analyses if a.is_coherent)
        complete_articles = sum(1 for a in analyses if a.chunk_type == ChunkTypeEnum.COMPLETE_ARTICLE)
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
        print(f"""
📊 RESULTADOS DE VALIDACIÓN:
   • Coherentes: {coherent_count}/{len(chunks)} ({coherent_count/len(chunks)*100:.1f}%)
   • Artículos completos: {complete_articles}
   • Promedio confianza: {avg_confidence:.2f}
""")
    else:
        coherent_count = 0
        complete_articles = 0
        avg_confidence = 0.0
        print(f"""
📊 RESULTADOS DE VALIDACIÓN:
   • Sin chunks para validar
""")

    return {
        "coherence_analyses": analyses,
        "decisions_log": add_decision_log(state, "validate_chunks", "completed", {
            "coherent_count": coherent_count,
            "total": len(chunks),
            "complete_articles": complete_articles
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def _analyze_chunk_coherence_llm(chunk_text: str) -> CoherenceAnalysis:
    """Análisis de coherencia usando LLM."""
    analyzer = get_llm(structured_output=CoherenceAnalysis)

    prompt = render_prompt(
        load_prompt_text("agent_z_coherence_user"),
        {"chunk_text": chunk_text}
    )

    try:
        print(f"\n   🤖 Analizando coherencia de chunk ({len(chunk_text)} chars)...")
        print(f"   📝 Texto: {chunk_text[:100]}...")

        result = analyzer.invoke([
            SystemMessage(content=load_prompt_text("agent_z_coherence_system")),
            HumanMessage(content=prompt)
        ])

        print(f"   ✅ Respuesta del LLM:")
        print(f"      • Coherente: {result.is_coherent}")
        print(f"      • Tipo: {result.chunk_type.value}")
        print(f"      • Confianza: {result.confidence:.2f}")
        print(f"      • Empieza limpio: {result.starts_cleanly}")
        print(f"      • Termina limpio: {result.ends_cleanly}")
        print(f"      • Acción: {result.suggested_action}")
        print(f"      • Razón: {result.reason}\n")

        return result
    except Exception as e:
        logger.error(f"Error en análisis LLM de chunk: {e}")
        raise


def create_final_chunks_node(state: AgentZState) -> dict:
    """
    NODO 7: Crea los chunks finales con metadata completa.
    """
    log_separator("NODO: CREATE FINAL CHUNKS", "═")

    preliminary_chunks = state["preliminary_chunks"]
    analyses = state["coherence_analyses"]
    doc_path = state["document_path"]
    topic = state["topic"]
    raw_content = state.get("raw_content", "")
    rewrite_flags = state.get("rewrite_flags") or []
    rewrite_reasons = state.get("rewrite_reasons") or []
    rewrite_types = state.get("rewrite_types") or []
    rewrite_errors = state.get("rewrite_errors") or []
    document_law = _infer_document_law(doc_path, raw_content)
    last_law_title = document_law

    print(f"""
🔨 CREANDO CHUNKS FINALES:
   • Chunks preliminares: {len(preliminary_chunks)}
   • Con metadata completa
""")

    final_chunks = []

    for i, (chunk_text, analysis) in enumerate(zip(preliminary_chunks, analyses)):
        current_law = _find_law_title_in_chunk(chunk_text)
        if current_law:
            last_law_title = current_law

        # Saltar chunks basura
        if analysis.suggested_action == "discard":
            continue

        has_article = _chunk_has_article(chunk_text)
        law_reference = None
        if has_article:
            if current_law:
                law_reference = current_law
            else:
                prev_law = (
                    _find_law_title_in_chunk(preliminary_chunks[i - 1], max_chars=1200)
                    if i > 0 else None
                )
                if prev_law:
                    law_reference = prev_law
                    last_law_title = prev_law
                else:
                    next_law = (
                        _find_law_title_in_chunk(preliminary_chunks[i + 1], max_chars=1200)
                        if i + 1 < len(preliminary_chunks) else None
                    )
                    if next_law:
                        law_reference = next_law
                    else:
                        law_reference = last_law_title or document_law

        if (
            has_article
            and law_reference
            and analysis.chunk_type not in (ChunkTypeEnum.METADATA, ChunkTypeEnum.GARBAGE)
            and not current_law
        ):
            chunk_text = _prepend_law_reference(chunk_text, law_reference)

        chunk_id = f"{doc_path.stem}_coherent_{uuid4().hex[:8]}"
        chunk_height = i / (len(preliminary_chunks) - 1) if len(preliminary_chunks) > 1 else 0.5

        rewritten_flag = rewrite_flags[i] if i < len(rewrite_flags) else False
        rewrite_reason = rewrite_reasons[i] if i < len(rewrite_reasons) else ""
        rewrite_type = rewrite_types[i] if i < len(rewrite_types) else "none"
        rewrite_error = rewrite_errors[i] if i < len(rewrite_errors) else ""

        metadata = {
            "topic": topic,
            "chunk_height": chunk_height,
            "chunk_index": i,
            "index": i,
            "total_chunks": len(preliminary_chunks),
            "doc_id": doc_path.stem,
            "file_name": doc_path.name,
            "rewritten": rewritten_flag,
            "coherent": analysis.is_coherent,
            "chunk_type": analysis.chunk_type.value,
            "coherence_score": analysis.confidence,
            "starts_cleanly": analysis.starts_cleanly,
            "ends_cleanly": analysis.ends_cleanly
        }
        if law_reference:
            metadata["law_reference"] = law_reference
        if rewritten_flag:
            metadata["rewrite_reason"] = rewrite_reason
            metadata["rewrite_type"] = rewrite_type
        if rewrite_error:
            metadata["rewrite_error"] = rewrite_error

        # Extraer página si está en el texto
        page = None
        if "--- Página" in chunk_text:
            try:
                page_marker = chunk_text.split("--- Página")[1].split("---")[0].strip()
                page = int(page_marker)
            except (IndexError, ValueError):
                pass

        token_count = len(chunk_text) // 4  # Estimación

        chunk = Chunk(
            chunk_id=chunk_id,
            content=chunk_text,
            source_document=str(doc_path),
            page=page,
            token_count=token_count,
            metadata=metadata
        )

        final_chunks.append(chunk)

    print(f"""
✅ CHUNKS FINALES CREADOS:
   • Total: {len(final_chunks)}
   • Coherentes: {sum(1 for c in final_chunks if c.metadata.get('coherent'))}
   • Descartados: {len(preliminary_chunks) - len(final_chunks)}
""")

    # Calcular métricas de calidad
    quality_metrics = {
        "total_chunks": len(final_chunks),
        "coherent_chunks": sum(1 for c in final_chunks if c.metadata.get("coherent")),
        "avg_coherence_score": sum(c.metadata.get("coherence_score", 0) for c in final_chunks) / len(final_chunks) if final_chunks else 0,
        "complete_articles": sum(1 for c in final_chunks if c.metadata.get("chunk_type") == "complete_article"),
        "avg_tokens": sum(c.token_count or 0 for c in final_chunks) / len(final_chunks) if final_chunks else 0
    }

    return {
        "final_chunks": final_chunks,
        "quality_metrics": quality_metrics,
        "decisions_log": add_decision_log(state, "create_final_chunks", "completed", quality_metrics),
        "node_count": state.get("node_count", 0) + 1
    }


def _build_fallback_chunks(
    preliminary_chunks: List[str],
    doc_path: Path,
    topic: int,
    analyses: Optional[List[CoherenceAnalysis]] = None,
    rewrite_flags: Optional[List[bool]] = None,
    rewrite_reasons: Optional[List[str]] = None,
    rewrite_types: Optional[List[str]] = None,
    rewrite_errors: Optional[List[str]] = None,
    raw_content: Optional[str] = None,
) -> List[Chunk]:
    """Construye chunks mínimos para guardado si falla el paso final."""
    final_chunks: List[Chunk] = []
    total = len(preliminary_chunks)
    analyses = analyses or []
    rewrite_flags = rewrite_flags or []
    rewrite_reasons = rewrite_reasons or []
    rewrite_types = rewrite_types or []
    rewrite_errors = rewrite_errors or []
    document_law = _infer_document_law(doc_path, raw_content or "")
    last_law_title = document_law

    for i, chunk_text in enumerate(preliminary_chunks):
        if not chunk_text.strip():
            continue

        analysis = analyses[i] if i < len(analyses) else None
        current_law = _find_law_title_in_chunk(chunk_text)
        if current_law:
            last_law_title = current_law

        has_article = _chunk_has_article(chunk_text)
        law_reference = None
        if has_article:
            if current_law:
                law_reference = current_law
            else:
                prev_law = (
                    _find_law_title_in_chunk(preliminary_chunks[i - 1], max_chars=1200)
                    if i > 0 else None
                )
                if prev_law:
                    law_reference = prev_law
                    last_law_title = prev_law
                else:
                    next_law = (
                        _find_law_title_in_chunk(preliminary_chunks[i + 1], max_chars=1200)
                        if i + 1 < len(preliminary_chunks) else None
                    )
                    if next_law:
                        law_reference = next_law
                    else:
                        law_reference = last_law_title or document_law

        if has_article and law_reference and not current_law:
            chunk_text = _prepend_law_reference(chunk_text, law_reference)

        chunk_id = f"{doc_path.stem}_coherent_{uuid4().hex[:8]}"
        chunk_height = i / (total - 1) if total > 1 else 0.5

        metadata = {
            "topic": topic,
            "chunk_height": chunk_height,
            "chunk_index": i,
            "index": i,
            "total_chunks": total,
            "doc_id": doc_path.stem,
            "file_name": doc_path.name,
            "rewritten": rewrite_flags[i] if i < len(rewrite_flags) else False,
            "coherent": analysis.is_coherent if analysis else True,
            "chunk_type": analysis.chunk_type.value if analysis else "partial_content",
            "coherence_score": analysis.confidence if analysis else 0.0,
            "starts_cleanly": analysis.starts_cleanly if analysis else False,
            "ends_cleanly": analysis.ends_cleanly if analysis else False,
            "fallback_saved": True
        }
        if law_reference:
            metadata["law_reference"] = law_reference
        if i < len(rewrite_reasons) and rewrite_reasons[i]:
            metadata["rewrite_reason"] = rewrite_reasons[i]
        if i < len(rewrite_types) and rewrite_types[i]:
            metadata["rewrite_type"] = rewrite_types[i]
        if i < len(rewrite_errors) and rewrite_errors[i]:
            metadata["rewrite_error"] = rewrite_errors[i]

        token_count = len(chunk_text) // 4

        chunk = Chunk(
            chunk_id=chunk_id,
            content=chunk_text,
            source_document=str(doc_path),
            page=None,
            token_count=token_count,
            metadata=metadata
        )
        final_chunks.append(chunk)

    return final_chunks


# ==========================================
# 5. CONSTRUCCIÓN DEL GRAFO
# ==========================================

def build_agent_z():
    """Construye y compila el grafo del Agente Z."""
    workflow = StateGraph(AgentZState)

    # Añadir nodos
    workflow.add_node("load_document", load_document_node)
    workflow.add_node("analyze_structure", analyze_structure_node)
    workflow.add_node("split_coherently", split_coherently_node)
    workflow.add_node("coordinate_context", coordinate_context_node)
    workflow.add_node("clean_and_rewrite", clean_and_rewrite_node)
    workflow.add_node("validate_chunks", validate_chunks_node)
    workflow.add_node("create_final_chunks", create_final_chunks_node)

    # Definir flujo lineal
    workflow.set_entry_point("load_document")
    workflow.add_edge("load_document", "analyze_structure")
    workflow.add_edge("analyze_structure", "split_coherently")
    workflow.add_edge("split_coherently", "coordinate_context")
    workflow.add_edge("coordinate_context", "clean_and_rewrite")
    workflow.add_edge("clean_and_rewrite", "validate_chunks")
    workflow.add_edge("validate_chunks", "create_final_chunks")
    workflow.add_edge("create_final_chunks", END)

    return workflow.compile()


# ==========================================
# 6. CLASE PRINCIPAL DEL AGENTE
# ==========================================

class RewriterAgent:
    """Agente Z: Reescribe documentos para generar chunks coherentes usando agentes ReAct."""

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        """Inicializa el agente.

        Args:
            model_name: Modelo de OpenAI (no usado actualmente, usa LLM factory)
            temperature: Temperatura (no usado actualmente)
        """
        self.settings = get_settings()
        self.graph = build_agent_z()

        logger.info(f"Initialized RewriterAgent (Modo: 100% LLM + ReAct)")

    def create_coherent_chunks(self, file_path: Path, topic: int, limit_chunks: int = 0) -> List[Chunk]:
        """Crea chunks coherentes usando el grafo.

        Sistema de caché:
        - Si existe input_docs/rewritten/{nombre}.json → carga desde caché (0 LLM calls)
        - Si no existe → procesa con LLM y guarda en caché

        Args:
            file_path: Path al documento
            topic: Número del tema

        Returns:
            Lista de chunks coherentes
        """
        log_separator(f"AGENTE Z: {file_path.name}", "█")

        # ==========================================
        # PASO 1: VERIFICAR CACHÉ
        # ==========================================
        print("\n🔍 Verificando caché de chunks...")
        cached_chunks = load_chunks_from_cache(file_path)

        if cached_chunks:
            print("✅ Usando chunks desde caché (sin procesamiento)")
            print(f"   ⚡ Ahorro: 100% de llamadas LLM")
            print(f"   📊 Chunks cargados: {len(cached_chunks)}\n")
            # DESHABILITADO: Generación de PDF tarda demasiado con muchos chunks
            # generate_chunks_pdf(cached_chunks, file_path)
            return cached_chunks

        print("❌ No existe caché, procesando documento con LLM...\n")

        # ==========================================
        # PASO 2: PROCESAR DOCUMENTO (PRIMERA VEZ)
        # ==========================================

        # Estado inicial
        initial_state = {
            "document_path": file_path,
            "topic": topic,
            "raw_content": "",
            "document_structure": None,
            "preliminary_chunks": [],
            "coherence_analyses": [],
            "final_chunks": [],
            "quality_metrics": {},
            "decisions_log": [],
            "node_count": 0,
            "split_mode": "free",
            "limit_chunks": limit_chunks
        }

        # Ejecutar grafo
        final_state = self.graph.invoke(initial_state)

        # Log de recorrido
        print("\n📊 RECORRIDO DEL GRAFO:")
        for log in final_state.get("decisions_log", []):
            print(f"   {log['timestamp'][-8:]} | {log['node']:25} | {log['decision']}")

        # Métricas finales
        metrics = final_state.get("quality_metrics", {})
        print(f"""
📊 MÉTRICAS DE CALIDAD:
   • Chunks totales: {metrics.get('total_chunks', 0)}
   • Chunks coherentes: {metrics.get('coherent_chunks', 0)}
   • Score promedio: {metrics.get('avg_coherence_score', 0):.2f}
   • Artículos completos: {metrics.get('complete_articles', 0)}
   • Tokens promedio: {metrics.get('avg_tokens', 0):.1f}
""")

        final_chunks = final_state.get("final_chunks", [])
        if not final_chunks:
            preliminary_chunks = final_state.get("preliminary_chunks") or []
            if preliminary_chunks:
                print("⚠️ No se generaron chunks finales. Guardando fallback con chunks preliminares.")
                final_chunks = _build_fallback_chunks(
                    preliminary_chunks=preliminary_chunks,
                    doc_path=file_path,
                    topic=topic,
                    analyses=final_state.get("coherence_analyses") or [],
                    rewrite_flags=final_state.get("rewrite_flags") or [],
                    rewrite_reasons=final_state.get("rewrite_reasons") or [],
                    rewrite_types=final_state.get("rewrite_types") or [],
                    rewrite_errors=final_state.get("rewrite_errors") or [],
                    raw_content=final_state.get("raw_content", "")
                )

        # ==========================================
        # PASO 3: GUARDAR EN CACHÉ PARA PRÓXIMAS VECES
        # ==========================================
        if final_chunks:
            try:
                save_chunks_to_cache(final_chunks, file_path)
                print("✅ Chunks guardados en caché para reutilización futura\n")
            except Exception as e:
                logger.error(f"Error guardando chunks en caché: {e}")
            # DESHABILITADO: Generación de PDF tarda demasiado con muchos chunks
            # try:
            #     generate_chunks_pdf(final_chunks, file_path)
            # except Exception as e:
            #     logger.warning(f"No se pudo generar PDF de chunks: {e}")

        return final_chunks

    def rewrite_document(
        self,
        input_path: Path,
        output_dir: Path,
        topic: int,
        limit_chunks: int = 0
    ) -> Path:
        """Reescribe un documento completo.

        Args:
            input_path: Path al documento original
            output_dir: Directorio de salida
            topic: Número del tema

        Returns:
            Path al documento reescrito
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / input_path.name

        # Crear chunks coherentes
        chunks = self.create_coherent_chunks(input_path, topic, limit_chunks=limit_chunks)

        # Copiar archivo original
        import shutil
        shutil.copy2(input_path, output_path)

        # Guardar metadata
        import json
        metadata_path = output_dir / f"{input_path.stem}_chunks_metadata.json"

        chunks_metadata = [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.metadata.get("chunk_index"),
                "index": c.metadata.get("index"),
                "doc_id": c.metadata.get("doc_id"),
                "total_chunks": c.metadata.get("total_chunks"),
                "page": c.page,
                "token_count": c.token_count,
                "coherent": c.metadata.get("coherent"),
                "chunk_type": c.metadata.get("chunk_type"),
                "coherence_score": c.metadata.get("coherence_score"),
                "law_reference": c.metadata.get("law_reference"),
                "content_preview": c.content[:100] + "..."
            }
            for c in chunks
        ]

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Document rewritten: {output_path}")
        logger.info(f"✓ Saved chunks metadata: {metadata_path}")

        return output_path

    def rewrite_all_documents(
        self,
        input_dir: Path = None,
        output_subdir: str = "rewritten",
        topic: int = 1
    ) -> Dict[str, Path]:
        """Reescribe todos los documentos de un directorio.

        Args:
            input_dir: Directorio de entrada
            output_subdir: Subdirectorio de salida
            topic: Número del tema por defecto

        Returns:
            Dict {original_path: rewritten_path}
        """
        if input_dir is None:
            input_dir = Path("input_docs")

        input_dir = Path(input_dir)
        output_dir = input_dir / output_subdir

        logger.info(f"Rewriting all documents from {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        results = {}
        errors = []

        pdf_files = list(input_dir.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing: {pdf_file.name}")
                logger.info(f"{'='*80}")

                rewritten_path = self.rewrite_document(pdf_file, output_dir, topic)
                results[str(pdf_file)] = rewritten_path

            except Exception as e:
                error_msg = f"Failed to rewrite {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                errors.append({"file": str(pdf_file), "error": str(e)})

        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"✓ Rewritten: {len(results)}")
        logger.info(f"✗ Failed: {len(errors)}")

        if errors:
            logger.warning(f"\nErrors:")
            for err in errors:
                logger.warning(f"  - {err['file']}: {err['error']}")

        return results


# ==========================================
# 7. FUNCIONES DE CONVENIENCIA
# ==========================================

def rewrite_documents_standalone(
    input_dir: str = "input_docs",
    output_subdir: str = "rewritten",
    topic: int = 1
):
    """Función standalone para reescribir documentos."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Rewriter Agent (Agent Z)")

    agent = RewriterAgent()
    results = agent.rewrite_all_documents(
        input_dir=Path(input_dir),
        output_subdir=output_subdir,
        topic=topic
    )

    logger.info(f"\n✓ Rewriting complete. {len(results)} documents processed.")

    return results


__all__ = [
    'RewriterAgent',
    'rewrite_documents_standalone',
]
