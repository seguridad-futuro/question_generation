"""Agente C: Quality Evaluator Agent (ReAct)

Agente autónomo que evalúa la calidad de preguntas generadas usando Ragas.
Razona sobre si aprobar, rechazar o marcar para revisión manual.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Dict, List, Literal, Optional, Any
import logging
from pydantic import BaseModel, Field, field_validator
import json
import re
import unicodedata

from models.question import Question
from models.chunk import Chunk
from models.quality_metrics import QualityMetrics
from config.thresholds import DEFAULT_QUALITY_THRESHOLDS
from utils.prompt_loader import load_prompt_text, render_prompt


# ==========================================
# MODELOS PYDANTIC PARA VALIDACIÓN
# ==========================================

class RagasEvaluation(BaseModel):
    """Resultado de evaluación con Ragas."""
    faithfulness: float = Field(description="Score de faithfulness (0-1)")
    answer_relevancy: float = Field(description="Score de answer relevancy (0-1)")
    success: bool = Field(description="Si la evaluación fue exitosa")
    error: Optional[str] = Field(default=None, description="Mensaje de error si falló")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


class QualityClassification(BaseModel):
    """Clasificación de calidad de una pregunta."""
    classification: str = Field(description="Clasificación: auto_pass, auto_fail, manual_review")
    reason: str = Field(description="Razón de la clasificación")
    action: str = Field(description="Acción recomendada: approve, reject_and_retry, mark_for_manual_review")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


class ContextCheckResult(BaseModel):
    """Resultado de verificación de respuesta en contexto."""
    exact_match: bool = Field(description="Si hay coincidencia exacta")
    word_overlap: float = Field(description="Porcentaje de palabras en común (0-1)")
    in_context: bool = Field(description="Si la respuesta está en el contexto")
    confidence: float = Field(description="Nivel de confianza (0-1)")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


class DifficultyAnalysis(BaseModel):
    """Análisis de dificultad de una pregunta."""
    difficulty: str = Field(description="Nivel de dificultad: fácil, medio, difícil")
    reason: str = Field(description="Razón del nivel asignado")
    error: Optional[str] = Field(default=None, description="Mensaje de error si falló")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


class SpecializedEvaluation(BaseModel):
    """Evaluación especializada para oposiciones de Guardia Civil."""
    classification: str = Field(description="Clasificación: auto_pass, auto_fail, manual_review")
    action: str = Field(description="Acción recomendada")
    faithfulness_estimated: float = Field(description="Score de faithfulness estimado (0-1)")
    relevancy_estimated: float = Field(description="Score de relevancy estimado (0-1)")
    validation_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detalles de validación por criterio"
    )
    improvement_feedback: Optional[str] = Field(
        default=None,
        description="Feedback específico si requiere mejoras"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confianza del evaluador (0-1)"
    )
    error: Optional[str] = Field(default=None, description="Mensaje de error si falló")
    raw_response: Optional[str] = Field(default=None, description="Respuesta cruda del LLM")
    overall_decision: Optional[str] = Field(default=None, description="Decisión general del evaluador")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


class QuickEvaluation(BaseModel):
    """Evaluación rápida mínima para salida estructurada."""
    classification: str = Field(description="auto_pass, auto_fail o manual_review")
    comment: str = Field(description="Comentario breve (1-2 frases)")
    improvement_feedback: Optional[str] = Field(default=None, description="Comentario breve (1-2 frases)")
    faithfulness_estimated: Optional[float] = Field(default=None, description="Score faithfulness (0-1)")
    relevancy_estimated: Optional[float] = Field(default=None, description="Score relevancy (0-1)")
    confidence: Optional[float] = Field(default=None, description="Confianza (0-1)")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("comment vacío")
        return str(v).strip()


class QuickComment(BaseModel):
    """Comentario breve del evaluador."""
    comment: str = Field(description="Comentario breve (1-2 frases)")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("comment vacío")
        return str(v).strip()


class RetryDecision(BaseModel):
    """Decisión sobre si hacer retry."""
    should_retry: bool = Field(description="Si se debe reintentar")
    reason: str = Field(description="Razón de la decisión")
    action: str = Field(description="Acción recomendada")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Personaliza el JSON schema para eliminar additionalProperties."""
        json_schema = handler(core_schema)
        json_schema.pop('additionalProperties', None)
        return json_schema


# ==========================================
# HERRAMIENTAS DEL AGENTE C
# ==========================================

def _render_prompt_template(prompt_template: str, values: Dict[str, Any]) -> str:
    """Rellena placeholders {key} sin interpretar llaves literales del template."""
    rendered = prompt_template
    missing = []
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder not in rendered:
            missing.append(placeholder)
            continue
        rendered = rendered.replace(placeholder, str(value))
    if missing:
        raise KeyError(f"Placeholders faltantes en el prompt: {', '.join(missing)}")
    return rendered


def _normalize_for_match(text: str) -> str:
    """Normaliza texto para comparar (lower, sin acentos, sin signos)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tip_mentions_correct_answer(tip: str, correct_answer: str) -> bool:
    """Verifica si el tip menciona el texto de la respuesta correcta."""
    tip_norm = _normalize_for_match(tip)
    answer_norm = _normalize_for_match(correct_answer)
    if not tip_norm or not answer_norm:
        return False
    if answer_norm in tip_norm:
        return True
    answer_words = set(answer_norm.split())
    if len(answer_words) < 3:
        return False
    overlap = len(answer_words.intersection(tip_norm.split())) / max(1, len(answer_words))
    return overlap >= 0.6


@tool
def evaluate_question_with_ragas(
    question: str,
    correct_answer: str,
    context: str
) -> RagasEvaluation:
    """Evalúa una pregunta usando Ragas (faithfulness y answer relevancy).

    Args:
        question: Texto de la pregunta
        correct_answer: Respuesta correcta
        context: Contexto del chunk

    Returns:
        RagasEvaluation: Scores de Ragas y status de éxito
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        # Preparar datos para Ragas
        data = {
            "question": [question],
            "answer": [correct_answer],
            "contexts": [[context]]
        }

        dataset = Dataset.from_dict(data)

        # Evaluar
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )

        return RagasEvaluation(
            faithfulness=float(result["faithfulness"]),
            answer_relevancy=float(result["answer_relevancy"]),
            success=True
        )

    except Exception as e:
        # Si falla Ragas, retornar error
        return RagasEvaluation(
            faithfulness=0.0,
            answer_relevancy=0.0,
            success=False,
            error=str(e)
        )


@tool
def classify_quality_score(
    faithfulness: float,
    relevancy: float,
    threshold_auto_pass_f: float = 0.85,
    threshold_auto_pass_r: float = 0.85,
    threshold_auto_fail_f: float = 0.60,
    threshold_auto_fail_r: float = 0.60
) -> QualityClassification:
    """Clasifica una pregunta según sus scores de calidad.

    Args:
        faithfulness: Score de faithfulness (0-1)
        relevancy: Score de relevancy (0-1)
        threshold_auto_pass_f: Threshold para auto-pass faithfulness
        threshold_auto_pass_r: Threshold para auto-pass relevancy
        threshold_auto_fail_f: Threshold para auto-fail faithfulness
        threshold_auto_fail_r: Threshold para auto-fail relevancy

    Returns:
        QualityClassification: Clasificación y razonamiento
    """
    # Auto-pass: AMBOS scores >= threshold
    if faithfulness >= threshold_auto_pass_f and relevancy >= threshold_auto_pass_r:
        return QualityClassification(
            classification="auto_pass",
            reason=f"Alta calidad: faithfulness={faithfulness:.2f}, relevancy={relevancy:.2f}",
            action="approve"
        )

    # Auto-fail: AL MENOS UNO score < threshold
    if faithfulness < threshold_auto_fail_f or relevancy < threshold_auto_fail_r:
        return QualityClassification(
            classification="auto_fail",
            reason=f"Baja calidad: faithfulness={faithfulness:.2f}, relevancy={relevancy:.2f}",
            action="reject_and_retry"
        )

    # Zona gris: Manual review
    return QualityClassification(
        classification="manual_review",
        reason=f"Calidad intermedia: faithfulness={faithfulness:.2f}, relevancy={relevancy:.2f}",
        action="mark_for_manual_review"
    )


@tool
def generate_improvement_feedback(
    question: str,
    faithfulness: float,
    relevancy: float,
    context: str
) -> str:
    """Genera feedback específico para mejorar una pregunta de baja calidad.

    Args:
        question: Pregunta original
        faithfulness: Score de faithfulness
        relevancy: Score de relevancy
        context: Contexto del chunk

    Returns:
        Feedback detallado para regeneración
    """
    feedback_parts = []

    # Feedback sobre faithfulness
    if faithfulness < 0.60:
        feedback_parts.append(
            f"⚠️ FAITHFULNESS MUY BAJO ({faithfulness:.2f}): "
            "La respuesta correcta NO es fiel al contexto. "
            "Asegúrate de que la respuesta esté EXPLÍCITAMENTE en el texto. "
            "No inventes información."
        )
    elif faithfulness < 0.85:
        feedback_parts.append(
            f"⚠️ FAITHFULNESS MEJORABLE ({faithfulness:.2f}): "
            "La respuesta es parcialmente fiel al contexto. "
            "Usa citas textuales o información muy clara del contexto."
        )

    # Feedback sobre relevancy
    if relevancy < 0.60:
        feedback_parts.append(
            f"⚠️ RELEVANCY MUY BAJO ({relevancy:.2f}): "
            "La respuesta NO es relevante para la pregunta. "
            "Reformula la pregunta para que la respuesta correcta sea más directa."
        )
    elif relevancy < 0.85:
        feedback_parts.append(
            f"⚠️ RELEVANCY MEJORABLE ({relevancy:.2f}): "
            "La respuesta es parcialmente relevante. "
            "Haz la pregunta más específica y directa."
        )

    # Recomendaciones generales
    if feedback_parts:
        feedback_parts.append("\n💡 RECOMENDACIONES:")
        feedback_parts.append("- Usa información textual del contexto")
        feedback_parts.append("- Formula preguntas directas y específicas")
        feedback_parts.append("- Asegúrate de que la respuesta correcta sea obvia del contexto")

    return "\n".join(feedback_parts)


@tool
def check_answer_in_context(answer: str, context: str) -> ContextCheckResult:
    """Verifica si la respuesta correcta está explícitamente en el contexto.

    Args:
        answer: Respuesta correcta
        context: Contexto del chunk

    Returns:
        ContextCheckResult: Resultado de verificación
    """
    context_lower = context.lower()
    answer_lower = answer.lower()

    # Verificación exacta
    exact_match = answer_lower in context_lower

    # Verificación por palabras clave
    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())
    common_words = answer_words.intersection(context_words)
    word_overlap = len(common_words) / len(answer_words) if answer_words else 0

    return ContextCheckResult(
        exact_match=exact_match,
        word_overlap=word_overlap,
        in_context=exact_match or word_overlap > 0.7,
        confidence=word_overlap
    )


@tool
def analyze_question_difficulty(question: str, context: str, file_name: str = "") -> DifficultyAnalysis:
    """Analiza la dificultad de una pregunta.

    Args:
        question: Texto de la pregunta
        context: Contexto del chunk
        file_name: Nombre del archivo fuente (proporciona contexto adicional)

    Returns:
        DifficultyAnalysis: Análisis de dificultad
    """
    from utils.llm_factory import create_llm
    from config.settings import get_settings

    logger = logging.getLogger("AgentC")

    # Prompt MUY simple y directo
    prompt = f"""Pregunta de oposición: "{question}"

Clasifica su dificultad para un estudiante:
- fácil: dato básico/obvio
- medio: requiere estudio
- difícil: detalle específico, fácil confundirse

Responde SOLO con este JSON (sin explicaciones adicionales):
{{"difficulty": "fácil", "reason": "razón breve"}}"""

    try:
        settings = get_settings()
        difficulty_model = settings.agent_c_difficulty_model or settings.openai_model
        llm = create_llm(model=difficulty_model, temperature=0, max_tokens=5000)
        response = llm.invoke(prompt)

        # Extraer contenido
        content = ""
        if hasattr(response, 'content') and response.content:
            content = str(response.content).strip()
        elif isinstance(response, str):
            content = response.strip()

        logger.debug(f"Respuesta dificultad raw: {content[:300]}")

        if not content:
            return DifficultyAnalysis(
                difficulty="medio",
                reason="Sin respuesta del modelo"
            )

        # Limpiar markdown
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if "{" in part:
                    content = part.strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                    break

        # Extraer JSON
        start = content.find('{')
        end = content.rfind('}') + 1

        if start != -1 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)

            difficulty = data.get("difficulty", "medio").lower().strip()
            reason = data.get("reason", "")

            # Normalizar
            if "fácil" in difficulty or "facil" in difficulty:
                difficulty = "fácil"
            elif "difícil" in difficulty or "dificil" in difficulty:
                difficulty = "difícil"
            else:
                difficulty = "medio"

            return DifficultyAnalysis(
                difficulty=difficulty,
                reason=reason if reason else f"Clasificada como {difficulty}"
            )

        # Inferir de texto si no hay JSON
        content_lower = content.lower()
        if "fácil" in content_lower or "facil" in content_lower:
            return DifficultyAnalysis(difficulty="fácil", reason=content[:150])
        elif "difícil" in content_lower or "dificil" in content_lower:
            return DifficultyAnalysis(difficulty="difícil", reason=content[:150])
        else:
            return DifficultyAnalysis(difficulty="medio", reason=content[:150] if content else "Análisis inconcluso")

    except Exception as e:
        logger.warning(f"Error analizando dificultad: {e}")
        return DifficultyAnalysis(
            difficulty="medio",
            reason="Error en análisis"
        )


@tool
def validate_distractors_are_incorrect(
    question: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str,
    correct_option: int,
    context: str
) -> Dict[str, Any]:
    """Valida que las opciones incorrectas sean REALMENTE incorrectas según el contexto.

    Esta herramienta usa un LLM para verificar que cada distractor sea falso/incorrecto.

    Args:
        question: Texto de la pregunta
        option1-4: Las 4 opciones
        correct_option: Índice de la correcta (1-4)
        context: Contexto del documento fuente

    Returns:
        Dict con:
        - all_distractors_incorrect: bool
        - incorrect_distractors: List[int] (índices de distractores que también son correctos)
        - feedback: str
    """
    from utils.llm_factory import create_llm_for_evaluation
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm_for_evaluation()

    options = {1: option1, 2: option2, 3: option3, 4: option4}
    distractors = {i: opt for i, opt in options.items() if i != correct_option}

    incorrect_distractors = []
    feedback_parts = []

    for idx, distractor_text in distractors.items():
        letter = chr(64 + idx)  # 1=A, 2=B, 3=C, 4=D

        prompt = render_prompt(
            load_prompt_text("agent_c_validate_distractor_user"),
            {
                "question": question,
                "letter": letter,
                "distractor_text": distractor_text,
                "context_excerpt": context[:1000]
            }
        )

        try:
            response = llm.invoke([
                SystemMessage(content=load_prompt_text("agent_c_validate_distractor_system")),
                HumanMessage(content=prompt)
            ])

            content = response.content.upper()

            if "VEREDICTO: CORRECTA" in content or "VEREDICTO:CORRECTA" in content:
                incorrect_distractors.append(idx)
                feedback_parts.append(f"Opción {letter} marcada como incorrecta PERO ES CORRECTA según el contexto")
            elif "VEREDICTO: AMBIGUA" in content or "VEREDICTO:AMBIGUA" in content:
                incorrect_distractors.append(idx)
                feedback_parts.append(f"Opción {letter} es AMBIGUA, no claramente incorrecta")

        except Exception as e:
            feedback_parts.append(f"Error validando opción {letter}: {e}")

    all_correct = len(incorrect_distractors) == 0

    result = {
        "all_distractors_incorrect": all_correct,
        "incorrect_distractors": incorrect_distractors,
        "feedback": "\n".join(feedback_parts) if feedback_parts else "Todos los distractores son correctamente incorrectos"
    }

    return result


@tool
def validate_tip_consistency(
    question: str,
    correct_option: int,
    tip: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str
) -> Dict[str, Any]:
    """Valida que el tip sea consistente y no contradictorio.

    Detecta errores como:
    - "La respuesta correcta es X" pero luego dice "X es incorrecta"
    - Explicaciones que se contradicen
    - Referencias incorrectas al texto de la opción correcta

    Args:
        question: Texto de la pregunta
        correct_option: Índice de la correcta (1-4)
        tip: Texto del tip/explicación
        option1-4: Las 4 opciones

    Returns:
        Dict con:
        - is_consistent: bool
        - contradictions: List[str] (lista de contradicciones encontradas)
        - feedback: str
    """
    from utils.llm_factory import create_llm_for_evaluation
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm_for_evaluation()

    options = {1: option1, 2: option2, 3: option3, 4: option4}
    correct_answer = options.get(correct_option, "")

    prompt = render_prompt(
        load_prompt_text("agent_c_tip_consistency_user"),
        {
            "question": question,
            "option1": option1,
            "option2": option2,
            "option3": option3,
            "option4": option4,
            "correct_answer": correct_answer,
            "tip": tip
        }
    )

    try:
        response = llm.invoke([
            SystemMessage(content=load_prompt_text("agent_c_tip_consistency_system")),
            HumanMessage(content=prompt)
        ])

        content = response.content.upper()

        # Detectar si es consistente
        is_consistent = "CONSISTENTE: SÍ" in content or "CONSISTENTE:SÍ" in content

        # Extraer contradicciones
        contradictions = []
        if not is_consistent:
            # Buscar la sección de contradicciones
            if "CONTRADICCIONES:" in content:
                contradictions_section = content.split("CONTRADICCIONES:")[1]
                if "EXPLICACIÓN:" in contradictions_section:
                    contradictions_section = contradictions_section.split("EXPLICACIÓN:")[0]

                # Extraer líneas con "-" que no sean "Ninguna"
                lines = contradictions_section.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") and "NINGUNA" not in line:
                        contradictions.append(line[1:].strip())

        result = {
            "is_consistent": is_consistent,
            "contradictions": contradictions,
            "feedback": "\n".join(contradictions) if contradictions else "El tip es consistente y sin contradicciones",
            "raw_response": response.content
        }

        return result

    except Exception as e:
        return {
            "is_consistent": False,
            "contradictions": [f"Error al validar: {str(e)}"],
            "feedback": f"Error al validar consistencia: {str(e)}",
            "raw_response": ""
        }


@tool
def validate_tip_supports_answer(
    question: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str,
    correct_option: int,
    tip: str
) -> Dict[str, Any]:
    """Valida que con el tip se pueda deducir la respuesta correcta.

    Usa un LLM para elegir la opcion correcta basandose SOLO en el tip/feedback.

    Returns:
        Dict con:
        - selected_option: int (1-4) o 0 si ambiguo
        - confidence: float (0-1)
        - matches_correct: bool
        - reason: str
        - error: str (opcional)
    """
    from utils.llm_factory import create_llm_for_evaluation
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm_for_evaluation()

    prompt = render_prompt(
        load_prompt_text("agent_c_tip_supports_answer_user"),
        {
            "question": question,
            "option1": option1,
            "option2": option2,
            "option3": option3,
            "option4": option4,
            "tip": tip
        }
    )

    try:
        response = llm.invoke([
            SystemMessage(content=load_prompt_text("agent_c_tip_supports_answer_system")),
            HumanMessage(content=prompt)
        ])
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end] if start != -1 and end != -1 else content
        parsed = json.loads(json_str)

        selected_option = int(parsed.get("selected_option", 0))
        if selected_option not in (0, 1, 2, 3, 4):
            selected_option = 0
        confidence = float(parsed.get("confidence", 0.0))
        reason = parsed.get("reason", "")

        matches_correct = selected_option == int(correct_option)

        return {
            "selected_option": selected_option,
            "confidence": confidence,
            "matches_correct": matches_correct,
            "reason": reason
        }
    except Exception as e:
        return {
            "selected_option": 0,
            "confidence": 0.0,
            "matches_correct": False,
            "reason": "Error al validar tip",
            "error": str(e)
        }


@tool
def evaluate_question_specialized(
    question: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str,
    correct_option: int,
    tip: str,
    article: str,
    context: str,
    source_document: str = "",
    topic: str = "",
    distractor_techniques: str = "",
    num_options: int = 4,
    num_correct_options: int = 1
) -> SpecializedEvaluation:
    """Evalúa una pregunta usando criterios especializados de oposiciones de Guardia Civil.

    Esta herramienta usa un prompt avanzado que valida:
    - Referencias legales explícitas
    - Precisión técnica absoluta
    - Distractores plausibles con técnicas válidas
    - Feedback estructurado apropiadamente
    - Formato oficial de oposiciones

    Args:
        question: Texto de la pregunta
        option1-4: Las 4 opciones de respuesta
        correct_option: Número de la opción correcta (1-4)
        tip: Feedback/explicación
        article: Artículo o ley citada
        context: Contexto del documento fuente
        source_document: Nombre del documento
        topic: Tema de la pregunta
        distractor_techniques: Técnicas de distracción aplicadas

    Returns:
        SpecializedEvaluation: Evaluación detallada según criterios especializados
    """
    from utils.llm_factory import create_llm_for_evaluation

    try:
        system_role = load_prompt_text("agent_c_specialized_system")
        evaluation_template = load_prompt_text("agent_c_specialized_user")
    except FileNotFoundError as e:
        return SpecializedEvaluation(
            classification="auto_fail",
            action="reject_and_retry",
            faithfulness_estimated=0.0,
            relevancy_estimated=0.0,
            error=f"No se encontró el archivo de prompt: {e}"
        )

    # Rellenar el template con los datos de la pregunta (sin romper llaves JSON)
    try:
        evaluation_prompt = _render_prompt_template(
            evaluation_template,
            {
                "question": question,
                "option1": option1,
                "option2": option2,
                "option3": option3,
                "option4": option4,
                "correct_option": correct_option,
                "tip": tip,
                "article": article or "No especificado",
                "context": context,
                "source_document": source_document or "Desconocido",
                "topic": topic or "Desconocido",
                "distractor_techniques": distractor_techniques or "No especificado",
                "num_options": num_options,
                "num_correct_options": num_correct_options,
            },
        )
    except KeyError as e:
        return SpecializedEvaluation(
            classification="auto_fail",
            action="reject_and_retry",
            faithfulness_estimated=0.0,
            relevancy_estimated=0.0,
            error=f"Template inválido: {str(e)}"
        )

    # Crear LLM para evaluación
    llm = create_llm_for_evaluation()

    # Invocar LLM con system role y prompt
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_role),
        HumanMessage(content=evaluation_prompt)
    ]

    try:
        response = llm.invoke(messages)
        content = response.content

        # Extraer JSON de la respuesta
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        evaluation_result = json.loads(content)
        evaluation_result.setdefault("raw_response", response.content)
        evaluation_result.setdefault("validation_details", None)
        evaluation_result.setdefault("improvement_feedback", "")
        evaluation_result.setdefault("confidence", None)

        # Asegurar que tenga los campos necesarios
        if "classification" not in evaluation_result:
            # Inferir clasificación de otros campos
            if evaluation_result.get("overall_decision") == "APROBADA":
                evaluation_result["classification"] = "auto_pass"
            elif evaluation_result.get("overall_decision") == "RECHAZADA":
                evaluation_result["classification"] = "auto_fail"
            else:
                evaluation_result["classification"] = "manual_review"

        if "action" not in evaluation_result:
            # Mapear decisión a acción
            decision_map = {
                "auto_pass": "approve",
                "manual_review": "mark_for_manual_review",
                "auto_fail": "reject_and_retry"
            }
            evaluation_result["action"] = decision_map.get(
                evaluation_result["classification"],
                "mark_for_manual_review"
            )

        # Agregar scores estimados si no están
        if "faithfulness_estimated" not in evaluation_result:
            # Estimar basado en clasificación
            if evaluation_result["classification"] == "auto_pass":
                evaluation_result["faithfulness_estimated"] = 0.90
                evaluation_result["relevancy_estimated"] = 0.90
            elif evaluation_result["classification"] == "auto_fail":
                evaluation_result["faithfulness_estimated"] = 0.50
                evaluation_result["relevancy_estimated"] = 0.50
            else:
                evaluation_result["faithfulness_estimated"] = 0.75
                evaluation_result["relevancy_estimated"] = 0.75

        return SpecializedEvaluation(**evaluation_result)

    except json.JSONDecodeError as e:
        return SpecializedEvaluation(
            classification="manual_review",
            action="mark_for_manual_review",
            faithfulness_estimated=0.70,
            relevancy_estimated=0.70,
            error=f"Error al parsear JSON de evaluación: {str(e)}",
            raw_response=response.content[:500]
        )
    except Exception as e:
        return SpecializedEvaluation(
            classification="manual_review",
            action="mark_for_manual_review",
            faithfulness_estimated=0.70,
            relevancy_estimated=0.70,
            error=f"Error en evaluación especializada: {str(e)}"
        )


@tool
def evaluate_question_quick(
    question: str,
    option1: str,
    option2: str,
    option3: str,
    option4: str,
    correct_option: int,
    tip: str,
    article: str,
    context: str,
    metadata: Optional[Dict[str, Any]] = None
) -> SpecializedEvaluation:
    """Evaluación rápida 100% LLM: clasificación + comentario siempre desde LLM."""
    from utils.llm_factory import create_llm_for_quick_evaluation
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        system_role = load_prompt_text("agent_c_quick_system")
        evaluation_template = load_prompt_text("agent_c_quick_user")
    except FileNotFoundError as e:
        return SpecializedEvaluation(
            classification="manual_review",
            action="mark_for_manual_review",
            faithfulness_estimated=0.70,
            relevancy_estimated=0.70,
            error=f"No se encontró el archivo de prompt rapido: {e}"
        )

    metadata = metadata or {}
    try:
        evaluation_prompt = _render_prompt_template(
            evaluation_template,
            {
                "question": question,
                "option1": option1,
                "option2": option2,
                "option3": option3,
                "option4": option4,
                "correct_option": correct_option,
                "correct_option_text": option1 if int(correct_option) == 1 else (
                    option2 if int(correct_option) == 2 else (
                        option3 if int(correct_option) == 3 else option4
                    )
                ),
                "tip": tip,
                "article": article or "No especificado",
                "context": context,
                "question_length": metadata.get("question_length", 0),
                "tip_words": metadata.get("tip_words", 0),
                "option_word_counts": metadata.get("option_word_counts", []),
                "num_options": metadata.get("num_options", 4),
            },
        )
    except KeyError as e:
        return SpecializedEvaluation(
            classification="manual_review",
            action="mark_for_manual_review",
            faithfulness_estimated=0.70,
            relevancy_estimated=0.70,
            error=f"Template rapido invalido: {str(e)}"
        )

    messages = [
        SystemMessage(content=system_role),
        HumanMessage(content=evaluation_prompt)
    ]

    def _invoke_quick(max_tokens: int = 2000):
        # max_tokens alto porque gpt-5 usa tokens para reasoning interno
        # Usar mismo patrón que Agente B: sin method específico
        llm = create_llm_for_quick_evaluation(max_tokens=max_tokens).with_structured_output(QuickEvaluation)
        return llm.invoke(messages)

    def _plain_eval_comment():
        """Fallback 100% LLM sin structured output: devuelve (classification, comment)."""
        eval_prompt = (
            "Devuelve SOLO dos líneas:\n"
            "classification=auto_pass|auto_fail|manual_review\n"
            "comment=Comentario breve (1-2 frases) con observación concreta.\n"
            "No uses frases genéricas."
        )
        plain_messages = [
            SystemMessage(content=system_role),
            HumanMessage(content=evaluation_prompt + "\n\n" + eval_prompt)
        ]
        llm = create_llm_for_quick_evaluation(max_tokens=200)
        response = llm.invoke(plain_messages)
        text = (response.content or "").strip()
        classification = ""
        comment = ""
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("classification="):
                classification = line.split("=", 1)[1].strip()
            if line.lower().startswith("comment="):
                comment = line.split("=", 1)[1].strip()
        return classification, comment

    def _plain_comment():
        """Comentario LLM sin structured output."""
        comment_prompt = (
            "Devuelve SOLO un comentario breve (1-2 frases) "
            "con al menos 1 observación concreta (tip, referencia legal, ambigüedad, opciones). "
            "No uses frases genéricas."
        )
        comment_messages = [
            SystemMessage(content=system_role),
            HumanMessage(content=evaluation_prompt + "\n\n" + comment_prompt)
        ]
        llm = create_llm_for_quick_evaluation(max_tokens=160)
        response = llm.invoke(comment_messages)
        return (response.content or "").strip()

    quick = None
    last_error = None
    for attempt in range(2):
        try:
            quick = _invoke_quick()
            break
        except Exception as e:
            last_error = e
            logging.getLogger("AgentC").warning(f"Intento {attempt+1} structured output falló: {type(e).__name__}: {e}")
            quick = None

    if quick is None and last_error:
        logging.getLogger("AgentC").error(f"Structured output falló completamente: {last_error}")

    classification = getattr(quick, "classification", None) or ""
    quick_comment = getattr(quick, "comment", None) or ""
    logging.getLogger("AgentC").info(
        f"Structured output: class={classification}, comment='{quick_comment[:80]}...'" if quick_comment else f"Structured output: class={classification}, comment=VACIO"
    )

    decision_map = {
        "auto_pass": "approve",
        "manual_review": "mark_for_manual_review",
        "auto_fail": "reject_and_retry"
    }
    action = decision_map.get(classification or "manual_review", "mark_for_manual_review")

    if quick and quick.faithfulness_estimated is not None:
        faithfulness = quick.faithfulness_estimated
        relevancy = quick.relevancy_estimated if quick.relevancy_estimated is not None else quick.faithfulness_estimated
    else:
        if classification == "auto_pass":
            faithfulness = 0.85
            relevancy = 0.85
        elif classification == "auto_fail":
            faithfulness = 0.50
            relevancy = 0.50
        else:
            faithfulness = 0.70
            relevancy = 0.70

    feedback = ""
    if quick:
        feedback = (quick.comment or quick.improvement_feedback or "").strip()

    if not classification or not feedback:
        try:
            plain_classification, plain_comment = _plain_eval_comment()
            logging.getLogger("AgentC").debug(f"Fallback plain_eval_comment: class={plain_classification}, comment={plain_comment[:50] if plain_comment else 'VACIO'}...")
            if not classification:
                classification = plain_classification or "manual_review"
            if not feedback:
                feedback = plain_comment
        except Exception as e:
            logging.getLogger("AgentC").warning(f"Fallback _plain_eval_comment falló: {e}")

    if not feedback:
        # Último intento: comentario sin structured output
        try:
            feedback = _plain_comment()
            logging.getLogger("AgentC").debug(f"Fallback plain_comment: {feedback[:50] if feedback else 'VACIO'}...")
        except Exception as e:
            logging.getLogger("AgentC").warning(f"Fallback _plain_comment falló: {e}")

    # Fallback final: pedir al LLM que RAZONE sobre la pregunta
    generic_phrases = ["", "Revisión manual solicitada por evaluación global.", "Revisión manual solicitada", "Análisis inconcluso - requiere revisión manual."]
    if not feedback or feedback.strip() in generic_phrases:
        # Llamada directa al LLM para razonamiento real
        reasoning_prompt = f"""Analiza esta pregunta de oposición y da tu evaluación en 2-3 frases.

PREGUNTA: {question}
OPCIONES:
  1) {option1}
  2) {option2}
  3) {option3}
  4) {option4}
MARCADA COMO CORRECTA: Opción {correct_option}
TIP/EXPLICACIÓN: {tip}
ARTÍCULO CITADO: {article if article else '(ninguno)'}

CONTEXTO FUENTE (extracto):
{context[:1500]}

CLASIFICACIÓN DECIDIDA: {classification}

INSTRUCCIONES:
- Si es "auto_pass": explica brevemente por qué la pregunta es válida
- Si es "auto_fail": explica qué error específico tiene (ej: respuesta incorrecta, opciones ambiguas, etc.)
- Si es "manual_review": explica EXACTAMENTE qué duda tienes (ej: "el tip dice X pero el contexto dice Y", "opciones 2 y 3 parecen ambas correctas", etc.)

Responde SOLO con tu análisis, sin prefijos ni formato especial."""

        try:
            logging.getLogger("AgentC").info("Iniciando llamada LLM para razonamiento...")
            llm = create_llm_for_quick_evaluation(max_tokens=200)
            response = llm.invoke(reasoning_prompt)

            # Debug: ver estructura de la respuesta
            logging.getLogger("AgentC").info(f"Tipo respuesta: {type(response)}")
            logging.getLogger("AgentC").info(f"Atributos respuesta: {dir(response)}")

            # Intentar extraer contenido de diferentes formas
            llm_feedback = ""
            if hasattr(response, 'content') and response.content:
                llm_feedback = str(response.content).strip()
            elif hasattr(response, 'text') and response.text:
                llm_feedback = str(response.text).strip()
            elif hasattr(response, 'message') and response.message:
                llm_feedback = str(response.message.content if hasattr(response.message, 'content') else response.message).strip()
            elif isinstance(response, str):
                llm_feedback = response.strip()
            elif isinstance(response, dict):
                llm_feedback = str(response.get('content', response.get('text', response.get('output', '')))).strip()

            logging.getLogger("AgentC").info(f"Respuesta LLM extraída ({len(llm_feedback)} chars): {llm_feedback[:150] if llm_feedback else 'VACIO'}...")

            if llm_feedback and len(llm_feedback) > 20:
                feedback = llm_feedback
            else:
                logging.getLogger("AgentC").warning(f"Respuesta LLM muy corta o vacía: '{llm_feedback}'")
        except Exception as e:
            import traceback
            logging.getLogger("AgentC").error(f"Fallback razonamiento LLM falló: {type(e).__name__}: {e}")
            logging.getLogger("AgentC").error(traceback.format_exc())

    # Si todo falló, mensaje mínimo
    if not feedback or feedback.strip() in generic_phrases:
        correct_text = option1 if correct_option == 1 else (option2 if correct_option == 2 else (option3 if correct_option == 3 else option4))
        feedback = f"Clasificación: {classification}. Correcta marcada: opción {correct_option}. Requiere revisión manual."

    if not classification:
        classification = "manual_review"
    return SpecializedEvaluation(
        classification=classification,
        action=action,
        faithfulness_estimated=faithfulness,
        relevancy_estimated=relevancy,
        improvement_feedback=feedback,
        confidence=getattr(quick, "confidence", None)
    )


@tool
def decide_retry_strategy(
    retry_count: int,
    max_retries: int,
    faithfulness: float,
    relevancy: float
) -> RetryDecision:
    """Decide si vale la pena hacer retry y con qué estrategia.

    Args:
        retry_count: Número de intentos actuales
        max_retries: Máximo de retries permitidos
        faithfulness: Score actual
        relevancy: Score actual

    Returns:
        RetryDecision: Decisión de retry con estrategia
    """
    # Si alcanzamos el máximo, no retry
    if retry_count >= max_retries:
        return RetryDecision(
            should_retry=False,
            reason=f"Máximo de retries alcanzado ({max_retries})",
            action="mark_for_manual_review"
        )

    # Si ambos scores son muy bajos (<0.4), probablemente el contexto no es bueno
    if faithfulness < 0.4 and relevancy < 0.4:
        return RetryDecision(
            should_retry=False,
            reason="Scores demasiado bajos, posible problema con el chunk",
            action="skip_chunk"
        )

    # Si solo uno es bajo, vale la pena retry
    if faithfulness < 0.60 or relevancy < 0.60:
        return RetryDecision(
            should_retry=True,
            reason=f"Score bajo pero recuperable (intento {retry_count + 1}/{max_retries})",
            action="regenerate_with_feedback"
        )

    # Scores en zona gris (0.60-0.85)
    if retry_count < max_retries - 1:  # Dejar un intento de margen
        return RetryDecision(
            should_retry=True,
            reason="Scores en zona gris, intentar mejorar",
            action="regenerate_with_feedback"
        )

    return RetryDecision(
        should_retry=False,
        reason="No se pudo mejorar suficiente",
        action="mark_for_manual_review"
    )


# ==========================================
# CREACIÓN DEL AGENTE C
# ==========================================

class LoggedReActAgent:
    """Wrapper para imprimir razonamientos completos del agente ReAct."""

    def __init__(self, agent, name: str = "Agente C"):
        self._agent = agent
        self._name = name

    def invoke(self, *args, **kwargs):
        result = self._agent.invoke(*args, **kwargs)

        messages = None
        if isinstance(result, dict):
            messages = result.get("messages")

        if messages:
            print(f"\n{self._name} (ReAct) razonamiento completo:")
            for msg in messages:
                role = getattr(msg, "type", None)
                if role != "ai":
                    continue

                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        name = call.get("name", "tool")
                        args = call.get("args", {})
                        print(f"   [tool_call] {name} {args}")

                content = getattr(msg, "content", None)
                if content:
                    for line in str(content).splitlines():
                        print(f"   [ai] {line}")

        return result

    def __getattr__(self, name: str):
        return getattr(self._agent, name)


def create_agent_c_evaluator():
    """Crea el Agente C - Quality Evaluator con capacidades de razonamiento.

    El agente puede:
    - Evaluar preguntas con Ragas
    - Clasificar según thresholds
    - Generar feedback para mejoras
    - Decidir estrategia de retry
    - Verificar que respuestas estén en contexto

    Returns:
        Compiled LangGraph agent
    """
    from utils.llm_factory import create_llm_for_agents

    thresholds = DEFAULT_QUALITY_THRESHOLDS

    # Herramientas disponibles
    tools = [
        evaluate_question_specialized,  # ⭐ PRINCIPAL: Validación especializada para oposiciones
        evaluate_question_with_ragas,
        classify_quality_score,
        generate_improvement_feedback,
        check_answer_in_context,
        analyze_question_difficulty,
        decide_retry_strategy,
        validate_distractors_are_incorrect,
        validate_tip_consistency,
        validate_tip_supports_answer
    ]

    # System prompt para el agente
    system_message = render_prompt(
        load_prompt_text("agent_c_system"),
        {
            "auto_pass_f": thresholds.auto_pass_faithfulness,
            "auto_pass_r": thresholds.auto_pass_relevancy,
            "auto_fail_f": thresholds.auto_fail_faithfulness,
            "auto_fail_r": thresholds.auto_fail_relevancy
        }
    )

    # LLM para el agente (razonamiento) - usa factory
    llm = create_llm_for_agents()

    # Crear agente con LangGraph create_react_agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message
    )

    return LoggedReActAgent(agent)


# ==========================================
# FUNCIÓN HELPER PARA USAR EL AGENTE
# ==========================================

def evaluate_question(
    question: Question,
    chunk: Chunk,
    retry_count: int = 0,
    max_retries: int = 3,
    agent = None,
    context_override: Optional[str] = None
) -> Dict:
    """Usa el Agente C para evaluar una pregunta.

    Args:
        question: Question object a evaluar
        chunk: Chunk de contexto
        retry_count: Número de intentos actuales
        max_retries: Máximo de retries
        agent_executor: Agente pre-creado (opcional)

    Returns:
        Dict con resultado de evaluación: {
            "classification": "auto_pass" | "auto_fail" | "manual_review",
            "metrics": QualityMetrics,
            "should_retry": bool,
            "feedback": str (si auto_fail)
        }
    """
    from config.settings import get_settings

    settings = get_settings()
    file_name = chunk.metadata.get("file_name", "desconocido")
    context_text = context_override or chunk.content

    options = [question.answer1, question.answer2, question.answer3, question.answer4 or ""]
    non_empty_options = [opt for opt in options if opt and str(opt).strip()]
    expected_options = max(1, settings.num_answer_options)
    expected_correct = max(1, settings.num_correct_options)

    if len(non_empty_options) != expected_options:
        metrics = QualityMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context=context_text,
            feedback=f"Número de opciones inválido: {len(non_empty_options)} (esperado {expected_options})"
        )
        return {
            "classification": "auto_fail",
            "metrics": metrics,
            "should_retry": True,
            "feedback": metrics.feedback,
            "agent_reasoning": metrics.feedback
        }

    if expected_correct != 1:
        metrics = QualityMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context=context_text,
            feedback=f"NUM_CORRECT_OPTIONS={expected_correct} no soportado (solo 1 correcta)"
        )
        return {
            "classification": "auto_fail",
            "metrics": metrics,
            "should_retry": True,
            "feedback": metrics.feedback,
            "agent_reasoning": metrics.feedback
        }

    metadata = {
        "question_length": len(question.question or ""),
        "tip_words": len((question.tip or "").split()),
        "option_word_counts": [len(str(opt).split()) for opt in options],
        "num_options": expected_options
    }

    if getattr(settings, "agent_c_fast_mode", False):
        max_chars = getattr(settings, "agent_c_context_max_chars", 0)
        context_for_eval = context_text
        if max_chars and len(context_text) > max_chars:
            context_for_eval = context_text[:max_chars]
        evaluation = evaluate_question_quick.invoke({
            "question": question.question,
            "option1": question.answer1,
            "option2": question.answer2,
            "option3": question.answer3,
            "option4": question.answer4 or "",
            "correct_option": question.solution,
            "tip": question.tip or "",
            "article": getattr(question, "article", "") or "",
            "context": context_for_eval,
            "metadata": metadata
        })
    else:
        evaluation = evaluate_question_specialized.invoke({
            "question": question.question,
            "option1": question.answer1,
            "option2": question.answer2,
            "option3": question.answer3,
            "option4": question.answer4 or "",
            "correct_option": question.solution,
            "tip": question.tip or "",
            "article": getattr(question, "article", "") or "",
            "context": context_text,
            "source_document": file_name,
            "topic": str(question.topic),
            "distractor_techniques": "",
            "num_options": expected_options,
            "num_correct_options": expected_correct
        })

    correct_answer = question.get_correct_answer()
    tip_text = (question.tip or "").strip()

    metrics = QualityMetrics(
        faithfulness=evaluation.faithfulness_estimated,
        answer_relevancy=evaluation.relevancy_estimated,
        context=context_text,
        feedback=evaluation.improvement_feedback
    )

    should_retry = evaluation.classification == "auto_fail" and retry_count < max_retries

    # Modo rápido: saltar checks extra para reducir tiempo total por pregunta
    if getattr(settings, "agent_c_fast_mode", False):
        reasoning_payload = evaluation.improvement_feedback or evaluation.validation_details or evaluation.raw_response or ""
        if isinstance(reasoning_payload, dict):
            reasoning_payload = json.dumps(reasoning_payload, ensure_ascii=False)

        # Analizar dificultad incluso en modo rápido
        difficulty_result = None
        try:
            difficulty_result = analyze_question_difficulty.invoke({
                "question": question.question,
                "context": context_text,
                "file_name": file_name
            })
        except Exception as exc:
            logger = logging.getLogger("AgentC")
            logger.warning(f"Error analizando dificultad: {exc}")

        return {
            "classification": evaluation.classification,
            "metrics": metrics,
            "should_retry": should_retry,
            "feedback": evaluation.improvement_feedback or evaluation.raw_response or "",
            "agent_reasoning": reasoning_payload,
            "difficulty": difficulty_result.difficulty if difficulty_result else None,
            "difficulty_reason": difficulty_result.reason if difficulty_result else None
        }

    extra_checks: Dict[str, Any] = {}
    extra_issues: List[str] = []

    if evaluation.classification != "auto_fail":
        # Tip requerido y consistente
        if not tip_text:
            extra_issues.append("Tip vacio o ausente")
        else:
            tip_mentions = _tip_mentions_correct_answer(tip_text, correct_answer)
            extra_checks["tip_mentions_correct_answer"] = tip_mentions
            if not tip_mentions:
                extra_issues.append("El tip no menciona el texto de la respuesta correcta")

            try:
                tip_consistency = validate_tip_consistency.invoke({
                    "question": question.question,
                    "correct_option": question.solution,
                    "tip": tip_text,
                    "option1": question.answer1,
                    "option2": question.answer2,
                    "option3": question.answer3,
                    "option4": question.answer4 or ""
                })
                extra_checks["tip_consistency"] = tip_consistency
                if not tip_consistency.get("is_consistent", False):
                    extra_issues.append("Tip con contradicciones internas")
            except Exception as exc:
                extra_issues.append(f"Error validando consistencia del tip: {exc}")

            try:
                tip_support = validate_tip_supports_answer.invoke({
                    "question": question.question,
                    "option1": question.answer1,
                    "option2": question.answer2,
                    "option3": question.answer3,
                    "option4": question.answer4 or "",
                    "correct_option": question.solution,
                    "tip": tip_text
                })
                extra_checks["tip_supports_answer"] = tip_support
                selected_option = tip_support.get("selected_option", 0)
                if selected_option == 0:
                    extra_issues.append("El tip no permite deducir una unica respuesta")
                elif selected_option != int(question.solution):
                    extra_issues.append(
                        f"El tip apunta a otra opcion (elige {selected_option})"
                    )
            except Exception as exc:
                extra_issues.append(f"Error validando que el tip soporte la respuesta: {exc}")

        # Respuesta correcta debe estar en el contexto
        try:
            context_check = check_answer_in_context.invoke({
                "answer": correct_answer,
                "context": context_text
            })
            if hasattr(context_check, "model_dump"):
                context_check = context_check.model_dump()
            extra_checks["answer_in_context"] = context_check
            if not context_check.get("in_context", False):
                extra_issues.append("La respuesta correcta no esta explicitamente en el contexto")
        except Exception as exc:
            extra_issues.append(f"Error validando respuesta en contexto: {exc}")

        # Validación adicional: detectar más correctas de las esperadas
        if expected_correct == 1:
            try:
                distractor_check = validate_distractors_are_incorrect.invoke({
                    "question": question.question,
                    "context": context_text,
                    "option1": question.answer1,
                    "option2": question.answer2,
                    "option3": question.answer3,
                    "option4": question.answer4 or "",
                    "correct_option": question.solution
                })
                extra_checks["distractor_check"] = distractor_check
                incorrect_distractors = distractor_check.get("incorrect_distractors", [])
                if incorrect_distractors:
                    extra = ", ".join(str(i) for i in incorrect_distractors)
                    extra_issues.append(
                        f"Se detectaron mas opciones correctas de las esperadas: {extra}"
                    )
            except Exception as exc:
                logger = logging.getLogger("AgentC")
                logger.warning(f"Error validando distractores: {exc}")

    # Clasificar issues por severidad
    critical_issues = []
    minor_issues = []

    for issue in extra_issues:
        # Issues criticos que requieren rechazo
        if any(kw in issue.lower() for kw in [
            "mas opciones correctas",
            "tip apunta a otra opcion",
            "contradicciones internas"
        ]):
            critical_issues.append(issue)
        else:
            # Issues menores que son advertencias (no rechazan)
            minor_issues.append(issue)

    # Analizar dificultad de la pregunta (antes de cualquier return)
    difficulty_result = None
    try:
        difficulty_result = analyze_question_difficulty.invoke({
            "question": question.question,
            "context": context_text,
            "file_name": file_name
        })
    except Exception as exc:
        logger = logging.getLogger("AgentC")
        logger.warning(f"Error analizando dificultad: {exc}")

    difficulty_value = difficulty_result.difficulty if difficulty_result else None
    difficulty_reason_value = difficulty_result.reason if difficulty_result else None

    # Solo auto_fail si hay issues CRITICOS
    if critical_issues:
        feedback = "CRITICO: " + "; ".join(critical_issues)
        if minor_issues:
            feedback += " | ADVERTENCIAS: " + "; ".join(minor_issues)
        metrics.feedback = feedback
        reasoning_payload = {
            "specialized": evaluation.validation_details or evaluation.raw_response or "",
            "extra_checks": extra_checks,
            "critical_issues": critical_issues,
            "minor_issues": minor_issues
        }
        return {
            "classification": "auto_fail",
            "metrics": metrics,
            "should_retry": retry_count < max_retries,
            "feedback": feedback,
            "agent_reasoning": json.dumps(reasoning_payload, ensure_ascii=False),
            "difficulty": difficulty_value,
            "difficulty_reason": difficulty_reason_value
        }

    # Si solo hay issues menores, aprobar con advertencias (no rechazar)
    if minor_issues:
        feedback = "ADVERTENCIAS (no bloqueantes): " + "; ".join(minor_issues)
        metrics.feedback = feedback
        # Aprobar pero con nota de advertencias
        return {
            "classification": "auto_pass",  # Aprobar a pesar de advertencias menores
            "metrics": metrics,
            "should_retry": False,
            "feedback": feedback,
            "agent_reasoning": json.dumps({
                "decision": "auto_pass_with_warnings",
                "minor_issues": minor_issues,
                "extra_checks": extra_checks
            }, ensure_ascii=False),
            "difficulty": difficulty_value,
            "difficulty_reason": difficulty_reason_value
        }

    reasoning_payload = evaluation.validation_details or evaluation.raw_response or ""
    if isinstance(reasoning_payload, dict):
        reasoning_payload = json.dumps(reasoning_payload, ensure_ascii=False)

    return {
        "classification": evaluation.classification,
        "metrics": metrics,
        "should_retry": should_retry,
        "feedback": evaluation.improvement_feedback or evaluation.error or "",
        "agent_reasoning": reasoning_payload,
        "difficulty": difficulty_value,
        "difficulty_reason": difficulty_reason_value
    }


# ==========================================
# EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    from models.chunk import Chunk
    from models.question import Question

    # Chunk de ejemplo
    chunk = Chunk(
        chunk_id="chunk_001",
        content="""Artículo 138 del Código Penal:

El que matare a otro será castigado, como reo de homicidio, con la pena
de prisión de diez a quince años.""",
        source_document="codigo_penal.pdf",
        page=45,
        token_count=50
    )

    from config.settings import get_settings
    settings = get_settings()
    model_name = settings.openai_model

    # Pregunta de ejemplo (alta calidad)
    good_question = Question(
        academy=1,
        topic=101,
        question="¿Cuál es la pena por homicidio según el artículo 138 del Código Penal?",
        answer1="Prisión de 10 a 15 años",
        answer2="Prisión de 5 a 10 años",
        answer3="Prisión de 15 a 20 años",
        answer4="Prisión de 20 a 25 años",
        solution=1,
        tip="El artículo 138 CP establece pena de 10 a 15 años",
        llm_model=model_name,
        source_chunk_id=chunk.chunk_id
    )

    # Pregunta de ejemplo (baja calidad - respuesta no en contexto)
    bad_question = Question(
        academy=1,
        topic=101,
        question="¿Cuál es la capital de España?",
        answer1="Madrid",
        answer2="Barcelona",
        answer3="Valencia",
        answer4="Sevilla",
        solution=1,
        tip="Madrid es la capital",
        llm_model=model_name,
        source_chunk_id=chunk.chunk_id
    )

    # Crear agente
    agent = create_agent_c_evaluator()

    print("\n" + "="*80)
    print("AGENTE C - QUALITY EVALUATOR")
    print("="*80 + "\n")

    # Evaluar pregunta buena
    print("📝 EVALUANDO PREGUNTA DE ALTA CALIDAD:")
    print("-" * 80)
    result_good = agent.invoke({
        "input": f"""Evalúa esta pregunta:

PREGUNTA: {good_question.question}
RESPUESTA CORRECTA: {good_question.get_correct_answer()}
CONTEXTO: {chunk.content}

Decide si aprobar, rechazar o marcar para revisión."""
    })
    print("\nRESULTADO:")
    print(result_good["output"])

    print("\n" + "="*80)

    # Evaluar pregunta mala
    print("\n📝 EVALUANDO PREGUNTA DE BAJA CALIDAD:")
    print("-" * 80)
    result_bad = agent.invoke({
        "input": f"""Evalúa esta pregunta:

PREGUNTA: {bad_question.question}
RESPUESTA CORRECTA: {bad_question.get_correct_answer()}
CONTEXTO: {chunk.content}

Decide si aprobar, rechazar o marcar para revisión."""
    })
    print("\nRESULTADO:")
    print(result_bad["output"])
