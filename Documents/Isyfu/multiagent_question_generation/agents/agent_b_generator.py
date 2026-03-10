"""Agente B: Question Generator con StateGraph y Multi-Chunk Inteligente.

Arquitectura:
- StateGraph para flujo de nodos
- Integración con FAISS para búsqueda semántica
- Prompts avanzados desde JSON
- LLM Factory para múltiples proveedores
- Logging detallado de cada paso
"""

import os
import json
import logging
import random
import re
from datetime import datetime
from typing import List, Optional, TypedDict, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- Imports del proyecto ---
from models.chunk import Chunk
from config.settings import get_settings
from utils.prompt_loader import load_prompt_text, render_prompt

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================

def setup_logger(name: str = "AgentB", level: int = logging.INFO) -> logging.Logger:
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

logger = setup_logger("AgentB")


def log_separator(title: str = "", char: str = "=", width: int = 70):
    """Imprime separador visual."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def log_question(q: 'QuestionData', title: str = "PREGUNTA GENERADA"):
    """Log formateado de una pregunta completa."""
    log_separator(title, "─")
    correct_letter = "?"
    if 1 <= q.correct <= 4:
        correct_letter = ['A', 'B', 'C', 'D'][q.correct - 1]
    chunk_info = ""
    if q.source_chunk_ids:
        preview_ids = q.source_chunk_ids[:3]
        suffix = f" +{len(q.source_chunk_ids) - 3}" if len(q.source_chunk_ids) > 3 else ""
        chunk_info = f"{', '.join(preview_ids)}{suffix}"
    print(f"""
📝 Pregunta: {q.question}

   A) {q.answer1}
   B) {q.answer2}
   C) {q.answer3}
   D) {q.answer4}

   ✓ Correcta: {q.correct} ({correct_letter})
   💡 Tip: {q.tip[:200] + '...' if len(q.tip) > 200 else q.tip}
   🧠 Razón (resumen): {(q.reasoning_summary[:200] + '...') if q.reasoning_summary and len(q.reasoning_summary) > 200 else (q.reasoning_summary or 'N/A')}
   📄 Artículo: {q.article or 'No especificado'}
   🔧 Método: {q.generation_method}
   🧩 Chunks: {chunk_info or 'N/A'}
   {'❌ Error: ' + q.error if q.error else ''}
""")
    log_separator("", "─")


def _strip_option_prefix(text: str) -> str:
    """Elimina prefijos tipo 'A)', 'B.', '1)' o '✓' al inicio de una opción."""
    if not text:
        return text
    cleaned = text.strip()
    cleaned = re.sub(r"^[✓✔•\-\*]+\s*", "", cleaned)
    label_re = re.compile(r"^\s*(?:\(?[A-Da-d]|[1-4]\)?)[.)\-:]+\s*")
    while True:
        updated = label_re.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned


def _normalize_answers(question: 'QuestionData') -> 'QuestionData':
    """Normaliza las opciones eliminando etiquetas de letra."""
    question.answer1 = _strip_option_prefix(question.answer1)
    question.answer2 = _strip_option_prefix(question.answer2)
    question.answer3 = _strip_option_prefix(question.answer3)
    if question.answer4 is not None:
        question.answer4 = _strip_option_prefix(question.answer4)
    return question


# ==========================================
# 1. MODELOS DE DATOS (PYDANTIC)
# ==========================================

class ActionEnum(str, Enum):
    """Acciones del evaluador de chunks."""
    DISCARD = "discard"
    NEEDS_PREV = "needs_prev"
    NEEDS_NEXT = "needs_next"
    NEEDS_CONTEXT = "needs_context"
    PROCEED = "proceed"


class ChunkEvaluation(BaseModel):
    """Decisión del LLM sobre la calidad del chunk."""
    action: ActionEnum = Field(description="Acción a tomar")
    reason: str = Field(description="Razón de la decisión")
    missing_context_clue: Optional[str] = Field(
        default=None,
        description="Pista de qué falta"
    )
    complexity_level: str = Field(
        default="medium",
        description="Nivel de complejidad: low, medium, high"
    )


class QuestionData(BaseModel):
    """Estructura final de la pregunta."""
    question: str = Field(description="Enunciado de la pregunta")
    answer1: str = Field(description="Opción 1 (sin prefijo A))")
    answer2: str = Field(description="Opción 2 (sin prefijo B))")
    answer3: str = Field(description="Opción 3 (sin prefijo C))")
    answer4: str = Field(description="Opción 4 (sin prefijo D))")
    correct: int = Field(description="Índice correcto (1-4)", ge=1, le=4)
    tip: str = Field(default="", description="Explicación didáctica con formato enriquecido: **negrita**, _cursiva_, ^subrayado^, #encabezado, - viñetas")
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Resumen breve del razonamiento (1-2 frases, sin pasos internos)"
    )
    article: Optional[str] = Field(
        default=None,
        description=(
            "Referencia legal profesional con formato enriquecido. REGLAS ESTRICTAS:\n"
            "1. Formato: '^Artículo X de la [Ley/Código completo]^: _[texto legal relevante]_'\n"
            "2. Usa ^subrayado^ para la referencia normativa y _cursiva_ para el texto literal\n"
            "3. Cita SOLO los apartados específicos que fundamentan la respuesta correcta\n"
            "4. El texto legal debe ser LITERAL pero limpio (sin marcadores de página, sin errores de OCR)\n"
            "5. NO copies párrafos enteros - extrae solo las líneas clave (máximo 3-4 líneas)\n"
            "6. Si el artículo es largo, cita el apartado concreto: '^Artículo 14.1 CE^: _..._'\n"
            "7. SIEMPRE especifica la norma completa (CE, CP, LECrim, LOPD, etc.)\n"
            "8. NUNCA incluyas referencias a libros, editoriales, páginas o URLs\n"
            "Ejemplo: '^Artículo 138.1 del Código Penal^: _El que matare a otro será castigado, "
            "como reo de homicidio, con la pena de prisión de diez a quince años._'"
        )
    )
    generation_method: str = Field(default="single_context", description="Método de generación")
    generation_time: Optional[float] = Field(default=None, description="Tiempo de generación en segundos")
    error: Optional[str] = Field(default=None, description="Error si la validación falla")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs de chunks usados")
    source_chunk_indices: List[int] = Field(default_factory=list, description="Índices de chunks usados")
    source_doc_ids: List[str] = Field(default_factory=list, description="Doc IDs de los chunks usados")
    context_strategy: Optional[str] = Field(default=None, description="Estrategia de contexto aplicada")


class ValidationResult(BaseModel):
    """Resultado de validación de pregunta."""
    is_valid: bool
    feedback: str
    errors: List[str] = Field(default_factory=list)


# ==========================================
# 2. ESTADO DEL AGENTE (STATE)
# ==========================================

class AgentState(TypedDict):
    """Estado compartido entre nodos del grafo."""
    original_chunk: Chunk
    retriever_service: Any
    enable_multi_chunk: bool

    current_content: str
    related_contents: List[str]
    expansion_count: int
    retry_count: int

    context_chunk_ids: List[str]
    context_chunk_indices: List[int]
    context_doc_ids: List[str]
    context_strategy: Optional[str]

    evaluation: Optional[ChunkEvaluation]
    generated_question: Optional[QuestionData]
    validation_result: Optional[ValidationResult]

    source_document: str
    topic: str
    page_number: int
    law_reference: Optional[str]  # Ley/norma a la que pertenece el contenido

    # Tracking para logs
    decisions_log: List[Dict[str, Any]]

    # Dificultad (selección aleatoria por pregunta)
    difficulty_label: Optional[str]
    difficulty_criteria: Optional[List[str]]
    difficulty_roll: Optional[int]

    # Feedback externo del Agente C (si existe)
    review_feedback: Optional[str]


# ==========================================
# 3. FUNCIONES DE UTILIDAD
# ==========================================

def get_llm(structured_output=None, for_generation: bool = False):
    """Obtiene LLM usando la factory del proyecto."""
    from utils.llm_factory import create_llm_for_generation, create_llm_for_agents

    if for_generation:
        llm = create_llm_for_generation()
    else:
        llm = create_llm_for_agents()

    if structured_output:
        return llm.with_structured_output(structured_output)
    return llm


def load_question_styles() -> dict:
    """Carga los estilos de pregunta desde el archivo JSON."""
    settings = get_settings()
    styles_path = settings.prompts_dir / "assets" / "question_styles.json"
    try:
        with open(styles_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Estilos de pregunta no encontrados: {styles_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Error parseando estilos JSON: {e}")
        return {}


def select_question_style() -> dict:
    """Selecciona un estilo de pregunta aleatoriamente usando pesos configurados."""
    styles_config = load_question_styles()
    if not styles_config or "styles" not in styles_config:
        # Fallback: estilo por defecto
        return {
            "id": 1,
            "name": "correcta_directa",
            "type": "positive",
            "template": "Señale la opción CORRECTA",
            "instruction": "Genera una pregunta donde UNA opción sea correcta y las otras 3 sean falsas.",
            "correct_option_is": "verdadera"
        }

    styles = styles_config["styles"]
    weights = styles_config.get("distribution_weights", {})

    # Construir lista de pesos en orden
    weight_list = []
    for style in styles:
        weight_key = f"{style['id']}_{style['name']}"
        weight_list.append(weights.get(weight_key, 10))  # Default 10 si no está definido

    # Seleccionar usando random.choices con pesos
    selected = random.choices(styles, weights=weight_list, k=1)[0]
    return selected


def get_forbidden_phrases() -> list:
    """Obtiene la lista de frases prohibidas del JSON de estilos."""
    styles_config = load_question_styles()
    return styles_config.get("forbidden_phrases", [])


def get_option_rules() -> dict:
    """Obtiene las reglas de opciones del JSON de estilos."""
    styles_config = load_question_styles()
    return styles_config.get("option_rules", {"max_words": 20})


class LawDetectionResult(BaseModel):
    """Resultado de la detección de ley por LLM."""
    law_name: str = Field(description="Nombre completo de la ley principal (ej: 'Ley Orgánica 3/2007, para la igualdad efectiva de mujeres y hombres')")
    law_short: str = Field(description="Nombre corto para citar (ej: 'Ley Orgánica 3/2007' o 'Constitución Española')")
    confidence: str = Field(description="Nivel de confianza: 'alta', 'media', 'baja'")
    reasoning: str = Field(description="Breve explicación de por qué esta es la ley principal del contenido")


def detect_law_with_llm(content: str, metadata_law: Optional[str] = None) -> Optional[str]:
    """Usa el LLM para detectar la ley principal del chunk de forma inteligente.

    El LLM analiza el contenido y determina cuál es la ley PRINCIPAL sobre la que
    trata el texto, no simplemente cualquier ley mencionada.

    Args:
        content: Contenido del chunk a analizar
        metadata_law: Ley sugerida por el metadata (puede ser incorrecta)

    Returns:
        Nombre de la ley principal para citar en las preguntas
    """
    if not content:
        return metadata_law

    # Usar solo las primeras 1500 chars para eficiencia
    excerpt = content[:1500]

    system_prompt = """Eres un experto en legislacion espanola. Tu tarea es identificar la LEY PRINCIPAL
sobre la que trata un fragmento de texto legal.

REGLAS CRITICAS (en orden de prioridad):
1. BUSCA PRIMERO el titulo de la ley al inicio del documento:
   - "LEY ORGANICA X/YYYY" = esa es la ley principal
   - "REAL DECRETO X/YYYY" = esa es la ley principal
   - "LEY X/YYYY" = esa es la ley principal

2. Si encuentras "Articulo X" y antes aparece el nombre de una ley especifica,
   ese articulo pertenece a ESA ley, NO a la Constitucion.

3. La Constitucion Espanola (articulos 1-169) SOLO es la ley principal si:
   - El texto cita LITERALMENTE "Constitucion Espanola, Articulo X"
   - NO hay ninguna otra ley mencionada como titulo del documento

4. PRIORIDAD ABSOLUTA: Leyes Organicas > Leyes Ordinarias > Reales Decretos > Constitucion

5. PISTAS de que NO es la Constitucion:
   - "Ley Organica 3/2007" aparece en el texto = es Ley Organica 3/2007
   - "para la igualdad efectiva" = es Ley Organica 3/2007
   - El articulo tiene mas de 5 puntos/apartados = probablemente NO es la Constitucion
   - Se habla de "Poderes Publicos" con criterios de actuacion = Ley Organica 3/2007

6. El nombre corto debe ser citeable (ej: "Ley Organica 3/2007", "CEDH", "Carta de la ONU")

EJEMPLOS:
- "LEY ORGANICA 3/2007... El articulo 14 de la Constitucion proclama..." = "Ley Organica 3/2007"
- "Articulo 14. Criterios generales de actuacion de los Poderes Publicos..." = "Ley Organica 3/2007" (la Constitucion no tiene esto)
- "Carta de las Naciones Unidas... La Asamblea General..." = "Carta de las Naciones Unidas"
- "Segun el articulo 1 de la Constitucion Espanola..." (sin otra ley) = "Constitucion Espanola"
"""

    user_prompt = f"""Analiza este fragmento legal e identifica la LEY PRINCIPAL:

---
{excerpt}
---

Ley sugerida por metadata: {metadata_law or 'No disponible'}

INSTRUCCIONES:
1. Busca titulo de ley al principio (ej: LEY ORGANICA 3/2007). Si existe, ESA es la ley principal.
2. Si los articulos tienen muchos apartados (mas de 5), probablemente NO es la Constitucion.
3. Si menciona igualdad efectiva de mujeres y hombres = Ley Organica 3/2007
4. Si menciona Naciones Unidas u ONU = Carta de las Naciones Unidas
5. La Constitucion solo tiene articulos cortos (1-3 apartados generalmente).

Determina cual es la ley PRINCIPAL del contenido. Ignora la sugerida si no coincide con el contenido real."""

    try:
        llm = get_llm(structured_output=LawDetectionResult)
        result = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        if result and result.law_short:
            print(f"   🔍 LLM detectó ley: {result.law_short} (confianza: {result.confidence})")
            print(f"   📝 Razón: {result.reasoning[:100]}..." if len(result.reasoning) > 100 else f"   📝 Razón: {result.reasoning}")
            if result.confidence != "baja":
                return result.law_short
            else:
                print(f"   ⚠️ Confianza baja, usando metadata como fallback")

    except Exception as e:
        logger.warning(f"Error en detección de ley con LLM: {e}")
        print(f"   ❌ Error LLM detección ley: {e}")

    # Fallback al metadata
    return metadata_law


def load_advanced_prompt() -> dict:
    """Carga el prompt avanzado desde el archivo JSON."""
    settings = get_settings()
    prompt_path = settings.prompts_dir / "agent_b" / "agent_b_generation.json"
    config: Dict[str, Any] = {}
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Prompt no encontrado: {prompt_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"Error parseando prompt JSON: {e}")

    try:
        config["system_role"] = load_prompt_text("agent_b_generation_system")
    except FileNotFoundError:
        if "system_role" not in config:
            logger.warning("Prompt system de generación no encontrado")

    try:
        config["generation_prompt_template"] = load_prompt_text("agent_b_generation_user")
    except FileNotFoundError:
        if "generation_prompt_template" not in config:
            logger.warning("Prompt de usuario de generación no encontrado")

    return config


def render_prompt_template(template: str, values: dict) -> str:
    """Rellena placeholders {key} en el template."""
    rendered = template
    for key, value in values.items():
        placeholder = "{" + key + "}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered


def _estimate_tokens(text: str) -> int:
    """Estimación simple de tokens (~4 caracteres por token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _count_words(text: str) -> int:
    """Cuenta palabras aproximadas para validación de longitud."""
    if not text:
        return 0
    return len(text.split())


NOISE_KEYWORDS = [
    "aspirantes.es",
    "ingreso al cuerpo de la guardia civil",
    "horas de estudio",
    "test de verificación de nivel",
    "hoja de respuestas",
    "guía del ciudadano",
    "ccn-cert",
    "al final del tema",
    "observaciones",
    "resultado",
    "primera vuelta",
    "segunda vuelta",
    "tercera vuelta",
    "1ª vuelta",
    "2ª vuelta",
    "3ª vuelta",
    "--- página",
    "página",
]


def _detect_noise_reason(text: str) -> Optional[str]:
    """Detecta ruido/guía/academia no sustantiva en el contenido."""
    if not text:
        return None
    lower = text.lower()
    for kw in NOISE_KEYWORDS:
        if kw in lower:
            return f"keyword:{kw}"
    if re.search(r'tema\\s*\\d+.*aspirantes\\.es', lower):
        return "pattern:tema_aspirantes"
    return None

DIFFICULTY_RULES = {
    1: {
        "label": "Nivel 1 (Recuperacion de datos)",
        "criteria": [
            "Objetivo: Comprobar memoria literal de un dato basico.",
            "Foco: Un dato unico (fecha, organo, cifra, definicion corta).",
            "Distractores: Error de Categoria Plausible (mismo tipo de dato).",
            "Regla de oro: PROHIBIDO usar distractores absurdos o deducibles por logica moral.",
            "Si la correcta es fecha/organo/cifra: usar otras fechas/organos/cifras reales del temario.",
            "Estructura: Enunciado corto y directo."
        ],
    },
    2: {
        "label": "Nivel 2 (Comprension y relacion)",
        "criteria": [
            "Objetivo: Comprobar comprension de estructura y jerarquia de la norma.",
            "Foco: Relacion entre organo y funcion (Organo A hace Funcion B).",
            "Tecnica: Confusion funcional entre organos similares.",
            "Distractores: Intercambio de competencias (p. ej., Consejo vs Asamblea).",
            "Estructura: Preferencia por preguntas negativas o condicionales, sin perder literalidad."
        ],
    },
    3: {
        "label": "Nivel 3 (Precision juridica)",
        "criteria": [
            "Objetivo: Discriminacion fina y literalidad estricta (trampa legal).",
            "Modificador: Cambiar 'podra' por 'debera' o 'fomentara' por 'garantizara'.",
            "Alcance: Introducir 'en todos los casos' o 'sin excepcion' cuando el texto tiene salvedades.",
            "Procedimiento: Anadir requisitos verosimiles no previstos (p. ej., 'previo informe vinculante').",
            "Distractores: Sintacticamente identicos a la correcta; solo cambia una palabra clave.",
            "Requiere: Distinguir matices modales y procedimentales."
        ],
    },
}


def _format_difficulty_criteria(criteria: Any) -> str:
    """Formatea criterios de dificultad para el prompt."""
    if not criteria:
        return ""
    if isinstance(criteria, str):
        return criteria.strip()
    return "\n".join(f"- {item}" for item in criteria)


def _format_review_feedback(feedback: Any) -> str:
    """Formatea el feedback externo para el prompt."""
    if not feedback:
        return ""
    if isinstance(feedback, str):
        return feedback.strip()
    if isinstance(feedback, dict):
        lines = []
        for key, value in feedback.items():
            if value is None:
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines).strip()
    if isinstance(feedback, (list, tuple)):
        return "\n".join(f"- {item}" for item in feedback if item).strip()
    return str(feedback).strip()


def _select_difficulty(state: Optional[AgentState] = None) -> Dict[str, Any]:
    """Selecciona dificultad con distribución ponderada o reutiliza la existente del estado.

    Distribucion: 25% Nivel 1, 50% Nivel 2, 25% Nivel 3
    """
    state = state or {}
    label = state.get("difficulty_label")
    criteria = state.get("difficulty_criteria")
    roll = state.get("difficulty_roll")
    if label and criteria:
        return {"label": label, "criteria": criteria, "roll": roll}

    # Distribucion ponderada: 25% Nivel 1, 50% Nivel 2, 25% Nivel 3
    roll = random.randint(1, 100)
    if roll <= 25:  # 1-25 = 25% Nivel 1
        difficulty_level = 1
    elif roll <= 75:  # 26-75 = 50% Nivel 2
        difficulty_level = 2
    else:  # 76-100 = 25% Nivel 3
        difficulty_level = 3

    choice = DIFFICULTY_RULES.get(difficulty_level, DIFFICULTY_RULES[2])
    return {"label": choice["label"], "criteria": choice["criteria"], "roll": roll}


def _normalize_article_id(article_id: str) -> str:
    return re.sub(r"\s+", " ", article_id.strip().lower())


def _extract_article_id(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(
        r'(?i)art[íi]culo\s+(\d+(?:\s*(?:bis|ter|quater))?)',
        text
    )
    if not match:
        match = re.search(
            r'(?i)\bart\.?\s+(\d+(?:\s*(?:bis|ter|quater))?)',
            text
        )
    if match:
        return _normalize_article_id(match.group(1))
    return None


def _extract_full_article_from_context(
    context: str,
    target_article_id: Optional[str] = None
) -> Optional[str]:
    if not context:
        return None

    header_pattern = re.compile(
        r'(?im)(?:^|\n)\s*(?:art[íi]culo|art\.?)\s+'
        r'(\d+(?:\s*(?:bis|ter|quater))?)\b.*'
    )
    matches = list(header_pattern.finditer(context))
    if not matches:
        return None

    selected_index = None
    if target_article_id:
        for idx, match in enumerate(matches):
            candidate = _normalize_article_id(match.group(1))
            if candidate == target_article_id:
                selected_index = idx
                break

    if selected_index is None:
        if len(matches) == 1:
            selected_index = 0
        else:
            return None

    start = matches[selected_index].start()
    end = matches[selected_index + 1].start() if selected_index + 1 < len(matches) else len(context)
    article_text = context[start:end].strip()
    return article_text if article_text else None


def _clean_article_text(text: str) -> str:
    """Remove common OCR/book artifacts from extracted article text."""
    if not text:
        return text
    # Remove page markers
    text = re.sub(r"---\s*Página\s*\d+\s*---", "", text)
    text = re.sub(r"---\s*Pág\.?\s*\d+\s*---", "", text)
    # Remove chapter headers
    text = re.sub(r"\n\s*(?:CAPITULO|Capítulo)\s+[IVXLC]+\s*\n.*?\n", "\n", text)
    # Remove editorial/book references
    text = re.sub(r"Aspirantes\.es.*", "", text)
    text = re.sub(r"info@\S+", "", text)
    # Fix common OCR errors
    text = re.sub(r"\bE1\b", "El", text)
    text = re.sub(r"(?<!\d)l\.\s+", "1. ", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    # Truncate if too long
    if len(text) > 600:
        # Try to cut at a sentence boundary
        cut = text[:600].rfind(".")
        if cut > 300:
            text = text[:cut + 1] + " (...)"
        else:
            text = text[:600] + " (...)"
    return text


def _get_chunk_index(chunk: Chunk) -> Optional[int]:
    """Obtiene índice de chunk desde metadata."""
    if not chunk or not chunk.metadata:
        return None
    idx = chunk.metadata.get("index")
    if idx is None:
        idx = chunk.metadata.get("chunk_index")
    if idx is None:
        return None
    try:
        return int(idx)
    except (TypeError, ValueError):
        return None


def _dedupe_preserve_order(values: List) -> List:
    """Elimina duplicados preservando orden."""
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_context(
    chunk_ids: List[str],
    chunk_indices: List[int],
    doc_ids: List[str]
) -> tuple[List[str], List[int], List[str]]:
    """Deduplica contexto preservando alineación."""
    dedup_ids = []
    dedup_indices = []
    dedup_docs = []
    seen = set()

    for i, chunk_id in enumerate(chunk_ids):
        index_val = chunk_indices[i] if i < len(chunk_indices) else None
        doc_val = doc_ids[i] if i < len(doc_ids) else None
        key = (chunk_id, index_val, doc_val)
        if key in seen:
            continue
        seen.add(key)
        dedup_ids.append(chunk_id)
        if index_val is not None:
            dedup_indices.append(index_val)
        if doc_val is not None:
            dedup_docs.append(doc_val)

    return dedup_ids, dedup_indices, dedup_docs


def add_decision_log(state: AgentState, node: str, decision: str, details: Dict = None) -> List[Dict]:
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


def _find_chunk_by_metadata(service: Any, doc_id: str, index: int) -> Optional[Chunk]:
    """Busca un chunk por metadata cuando el ID no es determinista."""
    if service is None:
        return None

    candidates = None
    if hasattr(service, "get_substantive_chunks"):
        try:
            candidates = service.get_substantive_chunks()
        except Exception:
            candidates = None

    if candidates is None and hasattr(service, "_chunks"):
        candidates = list(getattr(service, "_chunks").values())

    if not candidates:
        return None

    for candidate in candidates:
        meta = candidate.metadata or {}
        cand_doc_id = meta.get("doc_id")
        cand_index = meta.get("index", meta.get("chunk_index"))
        if cand_doc_id == doc_id and cand_index == index:
            return candidate

    return None


def shuffle_question_answers(question: QuestionData) -> QuestionData:
    """Baraja las opciones de respuesta para evitar sesgos.

    Args:
        question: Pregunta con opciones en orden original

    Returns:
        Pregunta con opciones barajadas y correct actualizado
    """
    # Crear lista de opciones con sus índices originales
    options = [
        (1, question.answer1),
        (2, question.answer2),
        (3, question.answer3),
        (4, question.answer4)
    ]

    # Guardar opciones originales para referencias en el tip
    original_options = {
        1: question.answer1,
        2: question.answer2,
        3: question.answer3,
        4: question.answer4
    }

    # Barajar aleatoriamente
    random.shuffle(options)

    # Crear mapeo de índices originales a nuevos
    # Ejemplo: si la opción 3 ahora está en posición 1, mapping[3] = 1
    mapping = {}
    for new_idx, (original_idx, _) in enumerate(options, 1):
        mapping[original_idx] = new_idx

    # Actualizar opciones
    question.answer1 = options[0][1]
    question.answer2 = options[1][1]
    question.answer3 = options[2][1]
    question.answer4 = options[3][1]

    # Actualizar índice de la respuesta correcta
    original_correct = question.correct
    question.correct = mapping[original_correct]

    # CRÍTICO: Eliminar referencias a letras/números en el tip
    if question.tip:
        question.tip = _update_tip_references(question.tip, original_options)

    return question


def _update_tip_references(tip: str, original_options: dict) -> str:
    """
    Reemplaza referencias a letras/números de opciones en el tip por el texto literal.

    Ejemplo:
        "La respuesta correcta es la B" → "La respuesta correcta es: [texto opción]"
    """
    import re

    # Mapeo de letras/números a texto literal
    options = {
        "A": _strip_option_prefix(original_options.get(1, "")),
        "B": _strip_option_prefix(original_options.get(2, "")),
        "C": _strip_option_prefix(original_options.get(3, "")),
        "D": _strip_option_prefix(original_options.get(4, "")),
        "1": _strip_option_prefix(original_options.get(1, "")),
        "2": _strip_option_prefix(original_options.get(2, "")),
        "3": _strip_option_prefix(original_options.get(3, "")),
        "4": _strip_option_prefix(original_options.get(4, "")),
    }

    # Patrones para detectar referencias con letras o números
    patterns = [
        r'(\brespuesta\s+correcta\s+es\s+la\s+)([ABCD])\b',
        r'(\brespuesta\s+correcta\s+es\s+la\s+)([1-4])\b',
        r'(\bopci[oó]n\s+)([ABCD])\b',
        r'(\bopci[oó]n\s+)([1-4])\b',
        r'(\brespuesta\s+)([ABCD])\b',
        r'(\brespuesta\s+)([1-4])\b',
        r'(\bla\s+)([ABCD])\b',
        r'(\bla\s+)([1-4])\b',
    ]

    updated_tip = tip

    def replace_ref(match):
        prefix = match.group(1)
        key = match.group(2).upper()
        text = options.get(key)
        if text:
            return f"{prefix}{text}"
        return match.group(0)

    for pattern in patterns:
        updated_tip = re.sub(pattern, replace_ref, updated_tip, flags=re.IGNORECASE)

    return updated_tip


# ==========================================
# 4. NODOS DEL GRAFO (CON LOGS DETALLADOS)
# ==========================================

def evaluate_node(state: AgentState) -> dict:
    """
    NODO 1: El Guardián.
    Analiza si el texto es basura, si está cortado, o si necesita más contexto.
    """
    log_separator("NODO: EVALUATE (Guardián)", "═")

    content = state['current_content']
    word_count = len(content.split())
    settings = get_settings()

    noise_reason = _detect_noise_reason(content)
    if noise_reason:
        # No descartar el chunk, pero marcar que tiene ruido para que el LLM lo evite
        print(f"   ⚠️ Ruido detectado ({noise_reason}), continuando pero evitando esas partes")
        # Agregar instrucción al state para que el LLM evite generar preguntas de artefactos
        state['has_noise'] = True
        state['noise_reason'] = noise_reason

    if not settings.b_enable_guardian:
        response = ChunkEvaluation(
            action=ActionEnum.PROCEED,
            reason="Guardian disabled via env",
            complexity_level="medium"
        )
        print("   ⚠️ Guardián desactivado: pasando a generación")
        decisions_log = add_decision_log(state, "evaluate", response.action.value, {
            "reason": response.reason,
            "complexity": response.complexity_level,
            "word_count": word_count
        })
        return {
            "evaluation": response,
            "expansion_count": state.get("expansion_count", 0),
            "retry_count": state.get("retry_count", 0),
            "decisions_log": decisions_log
        }

    print(f"   📄 Chunk {state['original_chunk'].chunk_id}: {word_count} palabras, pág.{state['page_number']}")
    print("   🤖 Evaluando chunk...")
    evaluator = get_llm(structured_output=ChunkEvaluation)

    system_prompt = load_prompt_text("agent_b_evaluate_system")
    human_prompt = render_prompt(
        load_prompt_text("agent_b_evaluate_user"),
        {"content": content}
    )
    response = evaluator.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    # Log de la decisión
    action_icons = {
        ActionEnum.DISCARD: "🗑️",
        ActionEnum.NEEDS_PREV: "⬅️",
        ActionEnum.NEEDS_NEXT: "➡️",
        ActionEnum.NEEDS_CONTEXT: "🔗",
        ActionEnum.PROCEED: "✅"
    }

    print(f"""
🎯 DECISIÓN DEL GUARDIÁN:
   {action_icons.get(response.action, '❓')} Acción: {response.action.value.upper()}
   📝 Razón: {response.reason}
   📊 Complejidad: {response.complexity_level}
   {'💡 Pista: ' + response.missing_context_clue if response.missing_context_clue else ''}
""")

    decisions_log = add_decision_log(state, "evaluate", response.action.value, {
        "reason": response.reason,
        "complexity": response.complexity_level,
        "word_count": word_count
    })

    return {
        "evaluation": response,
        "expansion_count": state.get("expansion_count", 0),
        "retry_count": state.get("retry_count", 0),
        "decisions_log": decisions_log
    }


def expand_sequential_node(state: AgentState) -> dict:
    """
    NODO 2a: Expansión Secuencial.
    Busca el chunk anterior o siguiente por índice.
    """
    log_separator("NODO: EXPAND SEQUENTIAL (Sanador)", "═")

    action = state["evaluation"].action
    chunk = state["original_chunk"]
    service = state.get("retriever_service")
    new_content = state["current_content"]

    print(f"""
🔧 CONFIGURACIÓN:
   • Acción requerida: {action.value}
   • Chunk original: {chunk.chunk_id}
   • Expansión #{state['expansion_count'] + 1}
""")

    if service is None:
        print("   ⚠️ No hay servicio de retrieval disponible.")
        return {
            "current_content": new_content,
            "expansion_count": state["expansion_count"] + 1,
            "context_chunk_ids": state.get("context_chunk_ids", [chunk.chunk_id]),
            "context_chunk_indices": state.get("context_chunk_indices", []),
            "context_doc_ids": state.get("context_doc_ids", []),
            "context_strategy": state.get("context_strategy", "single"),
            "decisions_log": add_decision_log(state, "expand_sequential", "no_service", {})
        }

    current_idx = chunk.metadata.get("index")
    if current_idx is None:
        current_idx = chunk.metadata.get("chunk_index")

    if current_idx is None:
        print("   ⚠️ Metadata sin índice, no se puede expandir secuencialmente.")
        return {
            "current_content": new_content,
            "expansion_count": state["expansion_count"] + 1,
            "context_chunk_ids": state.get("context_chunk_ids", [chunk.chunk_id]),
            "context_chunk_indices": state.get("context_chunk_indices", []),
            "context_doc_ids": state.get("context_doc_ids", []),
            "context_strategy": state.get("context_strategy", "single"),
            "decisions_log": add_decision_log(state, "expand_sequential", "no_index", {})
        }

    try:
        current_idx = int(current_idx)
    except (TypeError, ValueError):
        print("   ⚠️ Índice inválido, no se puede expandir.")
        return {
            "current_content": new_content,
            "expansion_count": state["expansion_count"] + 1,
            "context_chunk_ids": state.get("context_chunk_ids", [chunk.chunk_id]),
            "context_chunk_indices": state.get("context_chunk_indices", []),
            "context_doc_ids": state.get("context_doc_ids", []),
            "context_strategy": state.get("context_strategy", "single"),
            "decisions_log": add_decision_log(state, "expand_sequential", "invalid_index", {})
        }

    doc_id = chunk.metadata.get("doc_id")
    if not doc_id:
        doc_id = os.path.splitext(os.path.basename(chunk.source_document))[0]

    context_chunk_ids = list(state.get("context_chunk_ids") or [chunk.chunk_id])
    context_chunk_indices = list(state.get("context_chunk_indices") or [])
    if not context_chunk_indices:
        base_idx = _get_chunk_index(chunk)
        if base_idx is not None:
            context_chunk_indices = [base_idx]
    context_doc_ids = list(state.get("context_doc_ids") or [doc_id])
    context_strategy = state.get("context_strategy", "single")

    found_chunk = None
    if action == ActionEnum.NEEDS_PREV:
        target_id = f"{doc_id}_chunk_{current_idx - 1}"
        print(f"   🔍 Buscando chunk anterior: {target_id}")
        found_chunk = service.get_chunk(target_id)
        if not found_chunk:
            found_chunk = _find_chunk_by_metadata(service, doc_id, current_idx - 1)
        if found_chunk:
            prev_content = found_chunk.get_content_for_generation()
            new_content = f"{prev_content}\n\n[...]\n\n{new_content}"
            print(f"   ✅ Encontrado y fusionado (+{len(prev_content.split())} palabras)")
            context_chunk_ids = [found_chunk.chunk_id] + context_chunk_ids
            prev_idx = _get_chunk_index(found_chunk)
            if prev_idx is not None:
                context_chunk_indices = [prev_idx] + context_chunk_indices
            prev_doc_id = found_chunk.metadata.get("doc_id") if found_chunk.metadata else None
            if not prev_doc_id:
                prev_doc_id = os.path.splitext(os.path.basename(found_chunk.source_document))[0]
            context_doc_ids = [prev_doc_id] + context_doc_ids
            if "sequential_prev" not in context_strategy:
                context_strategy = f"{context_strategy}+sequential_prev" if context_strategy != "single" else "sequential_prev"

    elif action == ActionEnum.NEEDS_NEXT:
        target_idx = current_idx + 1 + state["expansion_count"]
        target_id = f"{doc_id}_chunk_{target_idx}"
        print(f"   🔍 Buscando chunk siguiente: {target_id}")
        found_chunk = service.get_chunk(target_id)
        if not found_chunk:
            found_chunk = _find_chunk_by_metadata(service, doc_id, target_idx)
        if found_chunk:
            next_content = found_chunk.get_content_for_generation()
            new_content = f"{new_content}\n\n[...]\n\n{next_content}"
            print(f"   ✅ Encontrado y fusionado (+{len(next_content.split())} palabras)")
            context_chunk_ids = context_chunk_ids + [found_chunk.chunk_id]
            next_idx = _get_chunk_index(found_chunk)
            if next_idx is not None:
                context_chunk_indices = context_chunk_indices + [next_idx]
            next_doc_id = found_chunk.metadata.get("doc_id") if found_chunk.metadata else None
            if not next_doc_id:
                next_doc_id = os.path.splitext(os.path.basename(found_chunk.source_document))[0]
            context_doc_ids = context_doc_ids + [next_doc_id]
            if "sequential_next" not in context_strategy:
                context_strategy = f"{context_strategy}+sequential_next" if context_strategy != "single" else "sequential_next"

    if not found_chunk:
        print("   ❌ No se encontró el chunk solicitado")

    print(f"\n📊 Contenido actual: {len(new_content.split())} palabras")
    context_chunk_ids, context_chunk_indices, context_doc_ids = _dedupe_context(
        context_chunk_ids, context_chunk_indices, context_doc_ids
    )

    return {
        "current_content": new_content,
        "expansion_count": state["expansion_count"] + 1,
        "context_chunk_ids": context_chunk_ids,
        "context_chunk_indices": context_chunk_indices,
        "context_doc_ids": context_doc_ids,
        "context_strategy": context_strategy,
        "decisions_log": add_decision_log(state, "expand_sequential", "expanded" if found_chunk else "not_found", {
            "action": action.value,
            "found": found_chunk is not None
        })
    }


def expand_semantic_node(state: AgentState) -> dict:
    """
    NODO 2b: Expansión Semántica.
    Busca chunks relacionados semánticamente usando FAISS.
    """
    log_separator("NODO: EXPAND SEMANTIC (FAISS)", "═")

    chunk = state["original_chunk"]
    service = state.get("retriever_service")
    settings = get_settings()

    print(f"""
🔧 CONFIGURACIÓN:
   • Chunk original: {chunk.chunk_id}
   • Threshold similitud: {settings.embedding_similarity_threshold}
   • Max chunks relacionados: {settings.max_related_chunks}
""")

    related_contents = list(state.get("related_contents", []))
    base_doc_id = chunk.metadata.get("doc_id") if chunk.metadata else None
    if not base_doc_id:
        base_doc_id = os.path.splitext(os.path.basename(chunk.source_document))[0]

    context_chunk_ids = list(state.get("context_chunk_ids") or [chunk.chunk_id])
    context_chunk_indices = list(state.get("context_chunk_indices") or [])
    if not context_chunk_indices:
        base_idx = _get_chunk_index(chunk)
        if base_idx is not None:
            context_chunk_indices = [base_idx]
    context_doc_ids = list(state.get("context_doc_ids") or [base_doc_id])
    context_strategy = state.get("context_strategy", "single")

    if service is None or not getattr(service, '_is_initialized', False):
        print("   ⚠️ Servicio de retrieval no disponible o no inicializado")
        return {
            "related_contents": related_contents,
            "expansion_count": state["expansion_count"] + 1,
            "context_chunk_ids": context_chunk_ids,
            "context_chunk_indices": context_chunk_indices,
            "context_doc_ids": context_doc_ids,
            "context_strategy": state.get("context_strategy", "single"),
            "decisions_log": add_decision_log(state, "expand_semantic", "no_service", {})
        }

    try:
        print("   🔍 Buscando chunks semánticamente relacionados...")
        related = service.retrieve_related_chunks(
            chunk_id=chunk.chunk_id,
            k=settings.max_related_chunks,
            threshold=settings.embedding_similarity_threshold
        )

        filtered_related = []
        for rel in related:
            rel_doc = rel.get("source_document")
            rel_meta = rel.get("metadata") or {}
            rel_doc_id = rel_meta.get("doc_id")
            if rel_doc == chunk.source_document or (
                rel_doc_id and rel_doc_id == chunk.metadata.get("doc_id")
            ):
                filtered_related.append(rel)

        if len(filtered_related) != len(related):
            print(f"   ⚠️ Filtrados por documento: {len(related) - len(filtered_related)}")

        print(f"\n📚 CHUNKS RELACIONADOS ENCONTRADOS: {len(filtered_related)}")
        for i, rel in enumerate(filtered_related, 1):
            print(f"""
   [{i}] {rel['chunk_id']}
       • Similitud: {rel['similarity_score']:.3f}
       • Documento: {rel.get('source_document', 'N/A')}
       • Preview: {rel['content'][:100]}...
""")
            related_contents.append(rel['content'])
            context_chunk_ids.append(rel["chunk_id"])
            rel_meta = rel.get("metadata") or {}
            rel_idx = rel_meta.get("index", rel_meta.get("chunk_index"))
            if rel_idx is not None:
                try:
                    context_chunk_indices.append(int(rel_idx))
                except (TypeError, ValueError):
                    pass
            rel_doc_id = rel_meta.get("doc_id")
            if not rel_doc_id:
                rel_doc_id = os.path.splitext(os.path.basename(rel.get("source_document", "")))[0]
            if rel_doc_id:
                context_doc_ids.append(rel_doc_id)

        if filtered_related:
            if "semantic" not in context_strategy:
                context_strategy = f"{context_strategy}+semantic" if context_strategy != "single" else "semantic"

    except Exception as e:
        print(f"   ❌ Error buscando chunks: {e}")
        logger.error(f"Error en expand_semantic: {e}")

    context_chunk_ids, context_chunk_indices, context_doc_ids = _dedupe_context(
        context_chunk_ids, context_chunk_indices, context_doc_ids
    )
    return {
        "related_contents": related_contents,
        "expansion_count": state["expansion_count"] + 1,
        "context_chunk_ids": context_chunk_ids,
        "context_chunk_indices": context_chunk_indices,
        "context_doc_ids": context_doc_ids,
        "context_strategy": context_strategy,
        "decisions_log": add_decision_log(state, "expand_semantic", "found" if filtered_related else "empty", {
            "num_found": len(filtered_related)
        })
    }


def generate_node(state: AgentState) -> dict:
    """
    NODO 3: El Generador.
    Crea la pregunta usando el prompt avanzado.
    """
    log_separator("NODO: GENERATE (Generador)", "═")

    retry_num = state.get('retry_count', 0) + 1

    # Cargar prompt avanzado
    prompt_config = load_advanced_prompt()

    if prompt_config:
        print("   ✅ Prompt avanzado cargado")
    else:
        print("   ⚠️ Usando prompt simple (fallback)")
        difficulty = _select_difficulty(state)
        return _generate_simple(state, difficulty=difficulty)

    # Seleccionar dificultad aleatoria (persistente en reintentos)
    difficulty = _select_difficulty(state)
    difficulty_label = difficulty["label"]
    difficulty_criteria = _format_difficulty_criteria(difficulty["criteria"])
    difficulty_roll = difficulty["roll"]
    review_feedback = _format_review_feedback(state.get("review_feedback"))

    # Seleccionar estilo de pregunta aleatoriamente
    question_style = select_question_style()
    style_name = question_style.get("name", "correcta_directa")
    style_template = question_style.get("template", "Señale la opción CORRECTA")
    style_instruction = question_style.get("instruction", "")
    style_type = question_style.get("type", "positive")
    style_example = question_style.get("example", "")

    # Construir contexto con límite de tokens
    settings = get_settings()
    max_tokens = settings.max_context_tokens
    main_content = state['current_content']
    selected_related = list(state.get('related_contents', []))

    if max_tokens and max_tokens > 0:
        main_tokens = _estimate_tokens(main_content)
        if main_tokens >= max_tokens:
            if selected_related:
                print("   ⚠️ Contexto principal excede el límite, descartando relacionados")
            selected_related = []
        else:
            total_tokens = main_tokens
            limited_related = []
            for rel_content in selected_related:
                rel_tokens = _estimate_tokens(rel_content)
                if total_tokens + rel_tokens <= max_tokens:
                    limited_related.append(rel_content)
                    total_tokens += rel_tokens
                else:
                    break
            if len(limited_related) < len(selected_related):
                print(f"   ⚠️ Contexto limitado a {max_tokens} tokens (relacionados truncados)")
            selected_related = limited_related

    has_related = len(selected_related) > 0
    method = "multi_context" if has_related else "single_context"

    feedback_str = " (con feedback C)" if review_feedback else ""
    print(f"   🔧 Generando: {method}, dificultad={difficulty_label}, chunks={len(selected_related)+1}{feedback_str}")

    if has_related:
        combined_context = f"[CONTEXTO PRINCIPAL]\n{main_content}"
        for i, rel_content in enumerate(selected_related, 1):
            combined_context += f"\n\n[CONTEXTO RELACIONADO {i}]\n{rel_content}"
    else:
        combined_context = main_content

    # Extraer componentes del prompt
    system_role = prompt_config.get('system_role', '')
    prompt_template = prompt_config.get('generation_prompt_template', '')
    few_shot_examples = prompt_config.get('few_shot_examples', [])

    # Construir few-shot context
    few_shot_context = ""
    if few_shot_examples:
        few_shot_context = "## EJEMPLOS DE REFERENCIA (CON TIPS)\n\n"
        for example in few_shot_examples[:3]:
            context = example.get("context", "")
            question = example.get("question", "")
            options = example.get("options", [])
            correct = example.get("correct_answer", "")
            feedback = example.get("feedback", "")

            if context:
                few_shot_context += f"**Contexto:** {context[:200]}...\n"
            if question:
                few_shot_context += f"**Pregunta:** {question}\n"
            if options and len(options) == 4:
                few_shot_context += (
                    f"**Opciones:** A) {options[0]} | B) {options[1]} | "
                    f"C) {options[2]} | D) {options[3]}\n"
                )
            if correct:
                few_shot_context += f"**Respuesta correcta:** {correct}\n"
            if feedback:
                few_shot_context += f"**Tip:** {feedback}\n"
            few_shot_context += "\n"
        print(f"   📝 Few-shot examples incluidos: {min(3, len(few_shot_examples))}")

    # Renderizar template
    try:
        generation_prompt = render_prompt_template(
            prompt_template,
            {
                "chunk_content": combined_context,
                "source_document": state.get('source_document', 'documento'),
                "topic": state.get('topic', 'general'),
                "page_number": state.get('page_number', 1),
                "difficulty_label": difficulty_label,
                "difficulty_criteria": difficulty_criteria,
                "review_feedback": review_feedback,
                "tip_min_words": settings.tip_min_words,
                "tip_max_words": settings.tip_max_words,
                "num_options": settings.num_answer_options,
                "num_correct_options": settings.num_correct_options,
            }
        )
    except Exception as e:
        print(f"   ❌ Error renderizando template: {e}")
        return _generate_simple(state, difficulty=difficulty)

    # Usar la ley detectada por el Agent Z (ya viene en metadata)
    law_reference = state.get('law_reference', '')

    # Construir instrucciones del estilo de pregunta seleccionado
    law_instruction = ""
    if law_reference:
        law_instruction = f"""
**📜 LEY/NORMA DE REFERENCIA - CRITICO:**
La ley PRINCIPAL de este contenido es: **{law_reference}**

⚠️⚠️⚠️ REGLA ABSOLUTAMENTE OBLIGATORIA ⚠️⚠️⚠️
El ENUNCIADO de tu pregunta DEBE incluir LITERALMENTE: "{law_reference}"

IMPORTANTE: Si el texto menciona "Constitucion" o "articulo 14 de la Constitucion" como REFERENCIA,
eso NO significa que la ley principal sea la Constitucion. La ley principal es: {law_reference}

FORMATOS CORRECTOS (USA ESTOS):
- "Segun el Articulo X de {law_reference}, ..."
- "De acuerdo con {law_reference}, cual...?"
- "Articulo X de {law_reference}. Senale la opcion CORRECTA"
- "Conforme a {law_reference}, senale..."

FORMATOS INCORRECTOS (NUNCA USES):
- "Segun la Constitucion espanola..." (si la ley es {law_reference})
- "Senale la proposicion INCORRECTA" (sin mencionar {law_reference})
- Cualquier enunciado que NO mencione "{law_reference}"
"""

    style_instructions = f"""
**🎯 ESTILO DE PREGUNTA OBLIGATORIO (seleccionado aleatoriamente):**

DEBES usar EXACTAMENTE este formato de pregunta:
- **Template:** "{style_template}"
- **Tipo:** {style_type} (la opción correcta es {question_style.get('correct_option_is', 'verdadera')})
- **Instrucción:** {style_instruction}
- **Ejemplo:** "{style_example}"
{law_instruction}
**REGLAS DE REDACCIÓN:**
- NO uses frases como "respecto a las", "descritas en el", "según el texto del"
- Las opciones deben ser TELEGRÁFICAS (máximo 20 palabras, ideal 8-15)
- TODAS las opciones deben tener longitud SIMILAR (±3 palabras entre ellas)
- NO copies artículos completos - SINTETIZA el dato clave
- **OBLIGATORIO:** Si mencionas un artículo (ej: "Artículo 14"), SIEMPRE especifica la ley/norma (ej: "Artículo 14 de la Constitución", "Artículo 6 del CEDH")
"""

    full_prompt = f"{few_shot_context}\n\n{style_instructions}\n\n{generation_prompt}"

    # Agregar advertencia si hay ruido detectado
    if state.get('has_noise'):
        noise_reason = state.get('noise_reason', 'artefactos detectados')
        full_prompt += f"\n\n**⚠️ ADVERTENCIA:** Este chunk contiene elementos de ruido ({noise_reason}). IGNORA completamente estas partes ruidosas (encabezados, pies de página, tests de verificación, etc.) y genera preguntas SOLO del contenido legal/sustantivo relevante."

    if review_feedback:
        full_prompt += f"\n\n**FEEDBACK DEL AGENTE C (corrige esto estrictamente):**\n{review_feedback}"

    print("\n🤖 Invocando LLM para generación...")
    generator = get_llm(structured_output=QuestionData, for_generation=True)

    # ⏱️ Iniciar tracking de tiempo
    import time
    start_time = time.time()

    try:
        response = generator.invoke([
            SystemMessage(content=system_role),
            HumanMessage(content=full_prompt)
        ])

        # ⏱️ Calcular tiempo de generación
        generation_time = time.time() - start_time

        # Asignar método de generación
        response.generation_method = method

        # Normalizar opciones antes de barajar para evitar etiquetas mezcladas
        response = _normalize_answers(response)

        # 🎲 BARAJAR OPCIONES para evitar sesgo
        response = shuffle_question_answers(response)

        # Only extract article from context if LLM didn't provide one
        if not response.article or len(response.article.strip()) < 20:
            target_article_id = _extract_article_id(response.tip or "")
            if not target_article_id:
                target_article_id = _extract_article_id(response.question or "")
            article_text = _extract_full_article_from_context(combined_context, target_article_id)
            if article_text:
                response.article = _clean_article_text(article_text)

        context_chunk_ids = list(state.get("context_chunk_ids") or [state["original_chunk"].chunk_id])
        context_chunk_indices = list(state.get("context_chunk_indices") or [])
        if not context_chunk_indices:
            base_idx = _get_chunk_index(state["original_chunk"])
            if base_idx is not None:
                context_chunk_indices = [base_idx]
        context_doc_ids = list(state.get("context_doc_ids") or [])
        if not context_doc_ids:
            base_doc_id = state["original_chunk"].metadata.get("doc_id") if state["original_chunk"].metadata else None
            if not base_doc_id:
                base_doc_id = os.path.splitext(os.path.basename(state["original_chunk"].source_document))[0]
            context_doc_ids = [base_doc_id]

        if len(context_chunk_indices) < len(context_chunk_ids):
            context_chunk_indices = context_chunk_indices + [-1] * (len(context_chunk_ids) - len(context_chunk_indices))
        if len(context_doc_ids) < len(context_chunk_ids):
            context_doc_ids = context_doc_ids + [""] * (len(context_chunk_ids) - len(context_doc_ids))

        if len(context_chunk_indices) < len(context_chunk_ids):
            context_chunk_indices = context_chunk_indices + [-1] * (len(context_chunk_ids) - len(context_chunk_indices))
        if len(context_doc_ids) < len(context_chunk_ids):
            context_doc_ids = context_doc_ids + [""] * (len(context_chunk_ids) - len(context_doc_ids))

        context_chunk_ids, context_chunk_indices, context_doc_ids = _dedupe_context(
            context_chunk_ids, context_chunk_indices, context_doc_ids
        )

        response.source_chunk_ids = context_chunk_ids
        response.source_chunk_indices = context_chunk_indices
        response.source_doc_ids = context_doc_ids
        response.context_strategy = state.get("context_strategy", method)

        # Log simple - pregunta completa se mostrará solo si pasa validación
        print(f"   ✅ Pregunta generada ({generation_time:.2f}s) - validando...")

        return {
            "generated_question": response,
            "generation_time": generation_time,  # ✨ AÑADIR TIEMPO
            "law_reference": law_reference,  # ✨ Ley detectada por LLM (no el metadata original)
            "difficulty_label": difficulty_label,
            "difficulty_criteria": difficulty.get("criteria"),
            "difficulty_roll": difficulty_roll,
            "decisions_log": add_decision_log(state, "generate", "success", {
                "method": method,
                "question_preview": response.question[:100],
                "context_chunks": context_chunk_ids,
                "shuffled": True,
                "difficulty": difficulty_label,
                "difficulty_roll": difficulty_roll,
                "generation_time": generation_time,
                "detected_law": law_reference
            })
        }

    except Exception as e:
        print(f"   ❌ Error en generación: {e}")
        logger.error(f"Error en generate_node: {e}")

        error_question = QuestionData(
            question="ERROR",
            answer1="x", answer2="x", answer3="x", answer4="x",
            correct=1,
            tip="Error en generación",
            error=str(e)
        )
        return {
            "generated_question": error_question,
            "difficulty_label": difficulty_label,
            "difficulty_criteria": difficulty.get("criteria"),
            "difficulty_roll": difficulty_roll,
            "decisions_log": add_decision_log(state, "generate", "error", {"error": str(e)})
        }


def _generate_simple(
    state: AgentState,
    difficulty: Optional[Dict[str, Any]] = None
) -> dict:
    """Generación simple como fallback."""
    print("   🔄 Usando generación simple...")

    # ⏱️ Iniciar tracking de tiempo
    import time
    start_time = time.time()

    generator = get_llm(structured_output=QuestionData, for_generation=True)

    difficulty = difficulty or _select_difficulty(state)
    difficulty_label = difficulty["label"]
    difficulty_criteria = _format_difficulty_criteria(difficulty["criteria"])
    difficulty_roll = difficulty["roll"]
    review_feedback = _format_review_feedback(state.get("review_feedback"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt_text("agent_b_simple_system")),
        ("human", load_prompt_text("agent_b_simple_human"))
    ])

    try:
        result = prompt | generator
        question = result.invoke({
            "content": state["current_content"],
            "difficulty_label": difficulty_label,
            "difficulty_criteria": difficulty_criteria,
            "review_feedback": review_feedback
        })
        question.generation_method = "simple_fallback"

        # ⏱️ Calcular tiempo de generación
        generation_time = time.time() - start_time

        # 🎲 BARAJAR OPCIONES para evitar sesgo
        question = shuffle_question_answers(question)

        context_chunk_ids = list(state.get("context_chunk_ids") or [state["original_chunk"].chunk_id])
        context_chunk_indices = list(state.get("context_chunk_indices") or [])
        if not context_chunk_indices:
            base_idx = _get_chunk_index(state["original_chunk"])
            if base_idx is not None:
                context_chunk_indices = [base_idx]
        context_doc_ids = list(state.get("context_doc_ids") or [])
        if not context_doc_ids:
            base_doc_id = state["original_chunk"].metadata.get("doc_id") if state["original_chunk"].metadata else None
            if not base_doc_id:
                base_doc_id = os.path.splitext(os.path.basename(state["original_chunk"].source_document))[0]
            context_doc_ids = [base_doc_id]

        context_chunk_ids, context_chunk_indices, context_doc_ids = _dedupe_context(
            context_chunk_ids, context_chunk_indices, context_doc_ids
        )

        question.source_chunk_ids = context_chunk_ids
        question.source_chunk_indices = context_chunk_indices
        question.source_doc_ids = context_doc_ids
        question.context_strategy = state.get("context_strategy", "simple_fallback")

        log_question(question, "PREGUNTA GENERADA (Fallback - Barajada)")
        print(f"   ⏱️ Tiempo de generación: {generation_time:.2f}s")
        return {
            "generated_question": question,
            "generation_time": generation_time,  # ✨ AÑADIR TIEMPO
            "difficulty_label": difficulty_label,
            "difficulty_criteria": difficulty.get("criteria"),
            "difficulty_roll": difficulty_roll
        }
    except Exception as e:
        return {"generated_question": QuestionData(
            question="ERROR",
            answer1="x", answer2="x", answer3="x", answer4="x",
            correct=1, tip="Error", error=str(e)
        ),
            "difficulty_label": difficulty_label,
            "difficulty_criteria": difficulty.get("criteria"),
            "difficulty_roll": difficulty_roll
        }


def validate_node(state: AgentState) -> dict:
    """
    NODO 4: El Validador.
    Verifica formato y calidad de la pregunta.
    """
    log_separator("NODO: VALIDATE (Validador)", "═")

    q = state["generated_question"]
    errors = []

    error_detail = None
    has_llm_error = False
    if q.error is not None:
        error_text = str(q.error).strip()
        if error_text and error_text.lower() not in ("none", "null"):
            has_llm_error = True
            error_detail = f"Error: {error_text}"

    print("   🔍 Validando...")

    # Validaciones con logs
    settings = get_settings()
    tip_word_count = _count_words(q.tip)
    tip_min_words = settings.tip_min_words
    tip_max_words = settings.tip_max_words

    answers = [q.answer1, q.answer2, q.answer3, q.answer4]
    non_empty_answers = [a for a in answers if a and str(a).strip()]
    expected_options = max(1, settings.num_answer_options)
    expected_correct = max(1, settings.num_correct_options)

    # Validación de longitud de opciones (relajado para ser menos estricto)
    MAX_OPTION_WORDS = 50  # Aumentado de 35
    option_word_counts = [_count_words(a) for a in non_empty_answers]
    max_option_words = max(option_word_counts) if option_word_counts else 0
    min_option_words = min(option_word_counts) if option_word_counts else 0
    correct_answer_words = _count_words(answers[q.correct - 1]) if 1 <= q.correct <= len(answers) else 0
    avg_incorrect_words = sum(option_word_counts) - correct_answer_words
    avg_incorrect_words = avg_incorrect_words / (len(option_word_counts) - 1) if len(option_word_counts) > 1 else 0

    # Detectar desequilibrio de longitud (relajado: 2.5x en lugar de 1.8x)
    length_imbalance = correct_answer_words > avg_incorrect_words * 2.5 if avg_incorrect_words > 0 else False

    checks = [
        ("Error LLM", has_llm_error, error_detail),
        ("Enunciado muy corto", len(q.question) < 15, f"Solo {len(q.question)} chars"),
        ("Enunciado muy largo", len(q.question) > 500, f"{len(q.question)} chars"),
        ("Número de opciones", len(non_empty_answers) != expected_options, f"{len(non_empty_answers)} (esperado {expected_options})"),
        ("Opciones repetidas", len(set(non_empty_answers)) < len(non_empty_answers), "Hay opciones duplicadas"),
        ("Índice inválido", q.correct < 1 or q.correct > expected_options, f"correct={q.correct} (esperado 1-{expected_options})"),
        ("Número de correctas no soportado", expected_correct != 1, f"NUM_CORRECT_OPTIONS={expected_correct} (solo 1 soportada)"),
        # Validaciones de longitud de tip ELIMINADAS para ser menos estricto
        ("Opción demasiado larga", max_option_words > MAX_OPTION_WORDS, f"Opción con {max_option_words} palabras (máx {MAX_OPTION_WORDS}). SINTETIZA las opciones."),
        ("Desequilibrio de longitud", length_imbalance, f"Correcta={correct_answer_words} palabras vs promedio incorrectas={avg_incorrect_words:.0f}. Equilibra longitudes."),
    ]

    for name, failed, detail in checks:
        if failed:  # Solo mostrar errores
            print(f"   ❌ {name}: {detail}")
            errors.append(f"{name}: {detail}")

    # NOTA: Filtros de palabras/frases prohibidas eliminados para ser menos estricto
    question_lower = q.question.lower()

    # Verificar que si se menciona un artículo, se especifique la ley/norma
    print("\n   🔍 Verificando referencia legal completa...")
    expected_law = state.get('law_reference', '')
    article_pattern = re.search(r'\b(art[íi]culo|art\.?)\s*\d+', question_lower)

    # Lista de nombres de leyes/normas que deben aparecer si se menciona un artículo
    law_keywords = [
        'constitución', 'carta', 'convenio', 'declaración', 'ley orgánica',
        'ley', 'real decreto', 'reglamento', 'código', 'estatuto',
        'tratado', 'directiva', 'naciones unidas', 'derechos humanos',
        'cedh', 'dudh', 'tfue', 'tue', 'lopj', 'lecrim', 'lec', 'igualdad'
    ]

    if article_pattern:
        has_law_reference = any(law in question_lower for law in law_keywords)
        if not has_law_reference:
            law_hint = f" Debe ser: '{expected_law}'" if expected_law else ""
            print(f"   ❌ Artículo sin ley/norma: Se menciona '{article_pattern.group()}' pero no se especifica la ley.{law_hint}")
            errors.append(f"Artículo sin ley: Menciona el artículo pero NO la ley/norma.{law_hint} AÑADE la referencia legal.")
        else:
            print(f"   ✅ Referencia legal detectada: OK")

    # Validar que se mencione la ley esperada EN EL ENUNCIADO (obligatorio)
    if expected_law:
        expected_law_lower = expected_law.lower()
        # Extraer palabras clave de la ley esperada
        expected_keywords = [w for w in expected_law_lower.split() if len(w) > 3]
        has_expected_law_in_question = any(kw in question_lower for kw in expected_keywords[:3])
        if has_expected_law_in_question:
            print(f"   ✅ Ley '{expected_law}' en enunciado: OK")
        else:
            print(f"   ❌ Ley '{expected_law}' NO está en el enunciado")
            errors.append(f"Falta ley en enunciado: DEBES mencionar '{expected_law}' en la pregunta. Ejemplo: 'Según el Artículo X de {expected_law}, ...'")

    # Detectar preguntas negativas y validar coherencia
    negative_patterns = ['incorrecta', 'excepto', 'no es', 'no será', 'no puede', 'no corresponde', 'falsa']
    question_lower = q.question.lower()
    is_negative_question = any(pat in question_lower for pat in negative_patterns)

    if is_negative_question:
        print("\n   ⚠️ PREGUNTA NEGATIVA detectada (INCORRECTA/EXCEPTO/NO)")
        # Verificar que el tip mencione por qué la opción es falsa
        tip_lower = q.tip.lower()
        explains_false = any(word in tip_lower for word in ['incorrecta', 'falsa', 'no es cierto', 'contradice', 'error'])
        if not explains_false:
            print("   ⚠️ Advertencia: El tip debería explicar por qué la opción marcada es FALSA")
            # No es error bloqueante, pero se registra como advertencia
        else:
            print("   ✅ El tip explica correctamente la opción incorrecta")

    is_valid = len(errors) == 0
    feedback = ", ".join(errors) if errors else "OK"

    if is_valid:
        print(f"   ✅ Validación exitosa")
    else:
        print(f"   ❌ Validación fallida ({len(errors)} error{'es' if len(errors) > 1 else ''})")

    return {
        "validation_result": ValidationResult(
            is_valid=is_valid,
            feedback=feedback,
            errors=errors
        ),
        "decisions_log": add_decision_log(state, "validate", "valid" if is_valid else "invalid", {
            "errors": errors
        })
    }


def rewrite_node(state: AgentState) -> dict:
    """
    NODO 5: Preparación para reintentar.
    """
    log_separator("NODO: REWRITE (Reintento)", "═")

    feedback = state.get("validation_result")
    retry_count = state["retry_count"] + 1

    print(f"""
🔄 PREPARANDO REINTENTO:
   • Intento actual: #{retry_count}
   • Feedback previo: {feedback.feedback if feedback else 'N/A'}
   • Errores a corregir: {feedback.errors if feedback else []}
""")

    return {
        "retry_count": retry_count,
        "decisions_log": add_decision_log(state, "rewrite", "retry", {
            "retry_num": retry_count,
            "feedback": feedback.feedback if feedback else None
        })
    }


# ==========================================
# 5. EDGES (LÓGICA DE RUTEO)
# ==========================================

def route_evaluation(state: AgentState) -> str:
    """Decide siguiente nodo basado en evaluación."""
    action = state["evaluation"].action
    expansion_count = state.get("expansion_count", 0)
    settings = get_settings()
    enable_multi_chunk = state.get(
        "enable_multi_chunk",
        settings.enable_multi_chunk and settings.b_enable_multi_chunk
    )
    max_expansions = settings.b_max_expansions

    # Routing simplificado

    if action == ActionEnum.DISCARD:
        print("   → Destino: END (chunk descartado)")
        return END

    if action in [ActionEnum.NEEDS_PREV, ActionEnum.NEEDS_NEXT]:
        if not settings.b_enable_sequential_expansion:
            print("   → Destino: generate (expansión secuencial desactivada)")
            return "generate"
        if expansion_count < max_expansions:
            print("   → Destino: expand_sequential")
            return "expand_sequential"
        else:
            print("   → Destino: generate (límite de expansión)")
            return "generate"

    if action == ActionEnum.NEEDS_CONTEXT:
        if not settings.b_enable_semantic_expansion:
            print("   → Destino: generate (expansión semántica desactivada)")
            return "generate"
        if expansion_count < max_expansions and enable_multi_chunk:
            print("   → Destino: expand_semantic (FAISS)")
            return "expand_semantic"
        else:
            print("   → Destino: generate")
            return "generate"

    print("   → Destino: generate")
    return "generate"


def route_validation(state: AgentState) -> str:
    """Decide si reintenta o termina."""
    is_valid = state["validation_result"].is_valid
    retry_count = state["retry_count"]
    max_retries = 2

    # Detectar error de contenido insuficiente - NO reintentar, saltar chunk
    q = state.get("generated_question")
    if q and q.error:
        error_text = str(q.error).lower()
        if "contenido_insuficiente" in error_text or "insuficiente" in error_text:
            print("   → Destino: END (chunk sin contenido sustantivo - saltando)")
            return END

    if is_valid:
        print("   → Destino: END (pregunta válida)")
        return END

    if retry_count >= max_retries:
        print("   → Destino: END (máx reintentos)")
        return END

    print("   → Destino: rewrite")
    return "rewrite"


# ==========================================
# 6. CONSTRUCCIÓN DEL GRAFO
# ==========================================

def build_agent_b():
    """Construye y compila el grafo del Agente B."""
    workflow = StateGraph(AgentState)

    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("expand_sequential", expand_sequential_node)
    workflow.add_node("expand_semantic", expand_semantic_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_evaluation,
        {
            END: END,
            "expand_sequential": "expand_sequential",
            "expand_semantic": "expand_semantic",
            "generate": "generate"
        }
    )

    workflow.add_edge("expand_sequential", "evaluate")
    workflow.add_edge("expand_semantic", "generate")
    workflow.add_edge("generate", "validate")

    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {
            END: END,
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("rewrite", "generate")

    return workflow.compile()


# ==========================================
# 7. FUNCIONES DE CONVENIENCIA
# ==========================================

def create_agent_b_generator(enable_multi_chunk: bool = True):
    """Crea el Agente B compilado."""
    settings = get_settings()
    settings.enable_multi_chunk = enable_multi_chunk
    return build_agent_b()


def generate_questions_for_chunk(
    chunk: Chunk,
    num_questions: int = 1,
    topic: str = "general",
    retriever_service: Any = None,
    enable_multi_chunk: Optional[bool] = None,
    review_feedback: Optional[str] = None,
    show_summary: bool = True
) -> List[QuestionData]:
    """Helper para generar preguntas de un chunk."""

    if enable_multi_chunk is None:
        settings = get_settings()
        enable_multi_chunk = settings.enable_multi_chunk and settings.b_enable_multi_chunk

    log_separator(f"Generando {num_questions} pregunta(s) - Chunk {chunk.chunk_id}", "█")

    agent = build_agent_b()
    questions = []

    for i in range(num_questions):
        log_separator(f"PREGUNTA {i+1}/{num_questions}", "▓")

        base_doc_id = chunk.metadata.get("doc_id") if chunk.metadata else None
        if not base_doc_id:
            base_doc_id = os.path.splitext(os.path.basename(chunk.source_document))[0]

        base_idx = _get_chunk_index(chunk)

        # Extraer law_reference del metadata del chunk
        law_reference = None
        if chunk.metadata:
            law_reference = chunk.metadata.get("law_reference")

        inputs = {
            "original_chunk": chunk,
            "current_content": chunk.get_content_for_generation(),
            "related_contents": [],
            "retriever_service": retriever_service,
            "enable_multi_chunk": enable_multi_chunk,
            "expansion_count": 0,
            "retry_count": 0,
            "source_document": chunk.source_document,
            "topic": topic,
            "page_number": chunk.page or 1,
            "law_reference": law_reference,
            "context_chunk_ids": [chunk.chunk_id],
            "context_chunk_indices": [base_idx] if base_idx is not None else [],
            "context_doc_ids": [base_doc_id],
            "context_strategy": "single",
            "evaluation": None,
            "generated_question": None,
            "validation_result": None,
            "decisions_log": [],
            "difficulty_label": None,
            "difficulty_criteria": None,
            "difficulty_roll": None,
            "review_feedback": review_feedback
        }

        try:
            final_state = agent.invoke(inputs)

            if final_state.get("generated_question"):
                q = final_state["generated_question"]
                # ⏱️ Asignar tiempo de generación si está disponible
                if final_state.get("generation_time"):
                    q.generation_time = final_state["generation_time"]

                if not q.error:
                    questions.append(q)
                    print(f"\n✅ Pregunta {i+1} generada exitosamente")
                else:
                    print(f"\n❌ Pregunta {i+1} tiene errores: {q.error}")

            # Recorrido del grafo eliminado para reducir verbosidad

        except Exception as e:
            print(f"\n❌ Error generando pregunta {i+1}: {e}")
            logger.error(f"Error en generate_questions_for_chunk: {e}")

    if show_summary:
        log_separator("RESUMEN FINAL", "█")
        print(f"""
📊 ESTADÍSTICAS:
   • Preguntas solicitadas: {num_questions}
   • Preguntas generadas: {len(questions)}
   • Tasa de éxito: {len(questions)/num_questions*100:.1f}%
""")

    return questions


def generate_questions_for_chunks(
    chunks: List[Chunk],
    num_questions: int = 1,
    topic: str = "general",
    retriever_service: Any = None,
    enable_multi_chunk: Optional[bool] = None,
    questions_per_chunk: int = 1,
    mix_adjacent: bool = True,
    mix_window: int = 1,
    review_feedback: Optional[str] = None
) -> List[QuestionData]:
    """Genera preguntas avanzando por múltiples chunks y mezclando contexto."""
    if not chunks:
        return []

    if enable_multi_chunk is None:
        enable_multi_chunk = get_settings().enable_multi_chunk

    if questions_per_chunk < 1:
        questions_per_chunk = 1

    mix_adjacent = mix_adjacent and enable_multi_chunk and mix_window > 0

    log_separator("INICIANDO GENERACIÓN MULTI-CHUNK", "█")
    print(f"""
📋 PARÁMETROS:
   • Chunks disponibles: {len(chunks)}
   • Preguntas a generar: {num_questions}
   • Preguntas por chunk: {questions_per_chunk}
   • Tema: {topic}
   • Retriever disponible: {'Sí' if retriever_service else 'No'}
   • Multi-chunk: {'Sí' if enable_multi_chunk else 'No'}
   • Mix adyacente: {'Sí' if mix_adjacent else 'No'} (ventana {mix_window})
""")

    agent = build_agent_b()
    questions: List[QuestionData] = []
    total_chunks = len(chunks)
    chunk_cursor = 0
    question_index = 0

    while question_index < num_questions:
        base_idx = chunk_cursor % total_chunks
        base_chunk = chunks[base_idx]

        context_chunks = [base_chunk]
        if mix_adjacent:
            for offset in range(1, mix_window + 1):
                target_idx = base_idx + offset
                if target_idx < total_chunks:
                    context_chunks.append(chunks[target_idx])

        context_strategy = "adjacent_mix" if len(context_chunks) > 1 else "single"
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        context_chunk_indices = []
        context_doc_ids = []
        for idx, ctx_chunk in enumerate(context_chunks):
            chunk_idx = _get_chunk_index(ctx_chunk)
            if chunk_idx is None:
                chunk_idx = base_idx + idx
            context_chunk_indices.append(chunk_idx)

            doc_id = ctx_chunk.metadata.get("doc_id") if ctx_chunk.metadata else None
            if not doc_id:
                doc_id = os.path.splitext(os.path.basename(ctx_chunk.source_document))[0]
            context_doc_ids.append(doc_id)

        context_chunk_ids, context_chunk_indices, context_doc_ids = _dedupe_context(
            context_chunk_ids, context_chunk_indices, context_doc_ids
        )

        print(f"\n   {'='*80}")
        print(f"   📝 Chunk base {base_idx + 1}/{total_chunks} | Contexto: {context_strategy}")
        print(f"   🧩 Chunks usados: {', '.join(context_chunk_ids)}")
        print(f"   {'='*80}")

        for _ in range(questions_per_chunk):
            if question_index >= num_questions:
                break

            inputs = {
                "original_chunk": base_chunk,
                "current_content": base_chunk.get_content_for_generation(),
                "related_contents": [c.get_content_for_generation() for c in context_chunks[1:]],
                "retriever_service": retriever_service,
                "enable_multi_chunk": enable_multi_chunk,
                "expansion_count": 0,
                "retry_count": 0,
                "source_document": base_chunk.source_document,
                "topic": topic,
                "page_number": base_chunk.page or 1,
                "context_chunk_ids": context_chunk_ids,
                "context_chunk_indices": context_chunk_indices,
                "context_doc_ids": context_doc_ids,
                "context_strategy": context_strategy,
                "evaluation": None,
                "generated_question": None,
                "validation_result": None,
                "decisions_log": [],
                "difficulty_label": None,
                "difficulty_criteria": None,
                "difficulty_roll": None,
                "review_feedback": review_feedback
            }

            try:
                final_state = agent.invoke(inputs)
                question_index += 1

                if final_state.get("generated_question"):
                    q = final_state["generated_question"]
                    if final_state.get("generation_time"):
                        q.generation_time = final_state["generation_time"]
                    if not q.error:
                        questions.append(q)
                        print(f"\n✅ Pregunta {question_index} generada exitosamente")
                    else:
                        print(f"\n❌ Pregunta {question_index} tiene errores: {q.error}")

                # Recorrido del grafo eliminado para reducir verbosidad

            except Exception as e:
                question_index += 1
                print(f"\n❌ Error generando pregunta {question_index}: {e}")
                logger.error(f"Error en generate_questions_for_chunks: {e}")

        chunk_cursor += 1
        if chunk_cursor >= total_chunks and question_index < num_questions:
            print("\n🔄 Reiniciando ciclo de chunks para completar preguntas restantes")
            chunk_cursor = 0

    log_separator("RESUMEN FINAL", "█")
    print(f"""
📊 ESTADÍSTICAS:
   • Preguntas solicitadas: {num_questions}
   • Preguntas generadas: {len(questions)}
   • Tasa de éxito: {len(questions)/num_questions*100:.1f}%
""")

    return questions


# ==========================================
# 8. RELACIÓN CON AGENTE C
# ==========================================

"""
FLUJO COMPLETO DEL PIPELINE:

┌─────────────────┐
│   AGENTE A      │  Chunking de documentos
│   (Chunker)     │
└────────┬────────┘
         │ chunks
         ▼
┌─────────────────┐
│   AGENTE B      │  Genera preguntas (este archivo)
│   (Generator)   │
└────────┬────────┘
         │ questions
         ▼
┌─────────────────┐
│   AGENTE C      │  Evalúa calidad (Ragas)
│   (Evaluator)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
AUTO_PASS  AUTO_FAIL ──────► Retry (vuelve a Agent B)
    │         │
    │         ▼
    │    MANUAL_REVIEW
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│   AGENTE D      │  Persistencia + Deduplicación
│   (Persistence) │
└─────────────────┘

INTEGRACIÓN:
- Agent B genera QuestionData
- Agent C evalúa con:
  - evaluate_question_specialized() → criterios de oposiciones
  - evaluate_question_with_ragas() → faithfulness, relevancy
- Si auto_fail: feedback → Agent B regenera
- Si auto_pass: → Agent D persiste
"""


# ==========================================
# 9. MAIN (PRUEBA)
# ==========================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          AGENTE B - QUESTION GENERATOR (TEST MODE)               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Chunk de prueba
    test_chunk = Chunk(
        chunk_id="test_001",
        content="""Artículo 138 del Código Penal:

        1. El que matare a otro será castigado, como reo de homicidio,
        con la pena de prisión de diez a quince años.

        2. Los hechos serán castigados con la pena superior en grado
        en los siguientes casos:
        a) cuando concurra en su comisión alguna de las circunstancias
        del apartado 1 del artículo 140, o
        b) cuando los hechos sean además constitutivos de un delito
        de atentado del artículo 550.""",
        source_document="codigo_penal.pdf",
        page=45,
        token_count=120,
        metadata={"doc_id": "CP", "index": 10}
    )

    questions = generate_questions_for_chunk(
        chunk=test_chunk,
        num_questions=1,
        topic="Derecho Penal"
    )

    if questions:
        log_separator("PREGUNTAS FINALES", "█")
        for i, q in enumerate(questions, 1):
            log_question(q, f"PREGUNTA FINAL #{i}")
