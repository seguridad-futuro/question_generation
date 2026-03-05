"""Agente E: PDF Generator - Genera PDFs de exámenes por tema

Arquitectura moderna usando LangGraph:
- StateGraph para flujo de nodos
- Agrupación de preguntas por tema
- Generación de PDFs con formato profesional
- Múltiples formatos: examen, con soluciones, solo respuestas
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# LangGraph Imports
from langgraph.graph import StateGraph, END

# ReportLab para PDFs
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, KeepTogether, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Matplotlib para histograma
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import io

# Imports del proyecto
from models.question import Question


# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================

def setup_logger(name: str = "AgentE", level: int = logging.INFO) -> logging.Logger:
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


logger = setup_logger("AgentE")


def log_separator(title: str = "", char: str = "=", width: int = 70):
    """Imprime separador visual."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def _strip_option_prefix(text: Optional[str]) -> str:
    """Elimina etiquetas tipo 'A)', 'B.' o '✓' al inicio de una opción."""
    if not text:
        return ""
    cleaned = text.strip()
    # Eliminar símbolos de viñeta al inicio (guion al final del grupo de caracteres)
    cleaned = re.sub(r"^[✓✔•*\-]+\s*", "", cleaned)
    # Eliminar etiquetas tipo A), B., 1), etc.
    label_re = re.compile(r"^\s*(?:\(?[A-Da-d]|[1-4]\)?)[.)\-:]+\s*")
    while True:
        updated = label_re.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned


def _normalize_tip_for_pdf(tip: Optional[str], options_by_letter: dict) -> str:
    """Elimina referencias a letras/números en el tip, usando texto literal."""
    if not tip:
        return ""

    options = {
        "A": _strip_option_prefix(options_by_letter.get("A", "")),
        "B": _strip_option_prefix(options_by_letter.get("B", "")),
        "C": _strip_option_prefix(options_by_letter.get("C", "")),
        "D": _strip_option_prefix(options_by_letter.get("D", "")),
        "1": _strip_option_prefix(options_by_letter.get("A", "")),
        "2": _strip_option_prefix(options_by_letter.get("B", "")),
        "3": _strip_option_prefix(options_by_letter.get("C", "")),
        "4": _strip_option_prefix(options_by_letter.get("D", "")),
    }

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

    def replace_ref(match):
        prefix = match.group(1)
        key = match.group(2).upper()
        text = options.get(key)
        if text:
            return f"{prefix}{text}"
        return match.group(0)

    updated = tip
    for pattern in patterns:
        updated = re.sub(pattern, replace_ref, updated, flags=re.IGNORECASE)

    return updated


def _truncate_text(text: Optional[str], max_len: int = 600) -> str:
    """Trunca texto largo para mostrar en PDF sin desbordes."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


# ==========================================
# 1. MODELOS DE DATOS (PYDANTIC)
# ==========================================

class PDFFormatEnum(str, Enum):
    """Formatos de PDF disponibles."""
    EXAM = "exam"  # Solo preguntas, sin respuestas
    WITH_SOLUTIONS = "with_solutions"  # Preguntas + respuestas al final
    ONLY_ANSWERS = "only_answers"  # Solo respuestas
    STUDY_GUIDE = "study_guide"  # Preguntas + tips expandidos
    STUDY_GUIDE_WITH_CHUNKS = "study_guide_with_chunks"  # Guía de estudio + chunks fuente


class PDFMetadata(BaseModel):
    """Metadata del PDF generado."""
    topic: int
    total_questions: int
    format: PDFFormatEnum
    generated_at: str
    file_name: str
    file_path: str


class QuestionGroup(BaseModel):
    """Grupo de preguntas por tema."""
    topic: int
    topic_name: str
    questions: List[Question]
    total_questions: int


# ==========================================
# 2. ESTADO DEL AGENTE (STATE)
# ==========================================

class AgentEState(TypedDict):
    """Estado compartido entre nodos del grafo."""
    # Input
    questions: List[Question]
    output_dir: Path
    pdf_format: PDFFormatEnum
    topic_filter: Optional[int]
    single_pdf: bool

    # Procesamiento
    questions_by_topic: Dict[int, List[Question]]
    pdf_metadata: List[PDFMetadata]

    # Tracking
    decisions_log: List[Dict[str, Any]]
    node_count: int


# ==========================================
# 3. FUNCIONES DE UTILIDAD
# ==========================================

def add_decision_log(state: AgentEState, node: str, decision: str, details: Dict = None) -> List[Dict]:
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


# ==========================================
# 4. NODOS DEL GRAFO
# ==========================================

def group_by_topic_node(state: AgentEState) -> dict:
    """
    NODO 1: Agrupa preguntas por tema.
    """
    log_separator("NODO: GROUP BY TOPIC", "═")

    questions = state["questions"]
    topic_filter = state.get("topic_filter")

    print(f"""
📊 AGRUPANDO PREGUNTAS:
   • Total preguntas: {len(questions)}
   • Filtro de tema: {topic_filter or 'Ninguno (todos)'}
""")

    questions_by_topic = {}

    for question in questions:
        topic = question.topic

        # Filtrar por tema si está especificado
        if topic_filter is not None and topic != topic_filter:
            continue

        if topic not in questions_by_topic:
            questions_by_topic[topic] = []

        questions_by_topic[topic].append(question)

    print(f"""
✅ AGRUPACIÓN COMPLETADA:
   • Temas únicos: {len(questions_by_topic)}
   • Preguntas por tema:
""")

    for topic, qs in sorted(questions_by_topic.items()):
        print(f"     Tema {topic}: {len(qs)} preguntas")

    return {
        "questions_by_topic": questions_by_topic,
        "decisions_log": add_decision_log(state, "group_by_topic", "completed", {
            "unique_topics": len(questions_by_topic),
            "total_questions": sum(len(qs) for qs in questions_by_topic.values())
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def generate_pdfs_node(state: AgentEState) -> dict:
    """
    NODO 2: Genera PDFs para cada tema o un único PDF combinado.
    """
    log_separator("NODO: GENERATE PDFS", "═")

    questions_by_topic = state["questions_by_topic"]
    output_dir = state["output_dir"]
    pdf_format = state["pdf_format"]
    single_pdf = state.get("single_pdf", False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"""
📄 GENERANDO PDFS:
   • Modo: {'PDF único (combinado)' if single_pdf else 'PDFs por tema'}
   • Temas: {len(questions_by_topic)}
   • Formato: {pdf_format.value}
   • Directorio: {output_dir}
""")

    pdf_metadata_list = []

    if single_pdf:
        # Generar un único PDF combinando todos los temas
        print(f"\n{'─'*70}")

        # Combinar todas las preguntas de todos los temas
        all_questions = []
        total_questions = 0
        for topic in sorted(questions_by_topic.keys()):
            all_questions.extend(questions_by_topic[topic])
            total_questions += len(questions_by_topic[topic])

        print(f"Generando PDF único con {total_questions} preguntas de {len(questions_by_topic)} temas...")

        try:
            # Generar PDF único
            pdf_path = _generate_combined_pdf(
                questions=all_questions,
                output_dir=output_dir,
                pdf_format=pdf_format,
                num_topics=len(questions_by_topic)
            )

            # Crear metadata
            metadata = PDFMetadata(
                topic=0,  # 0 indica PDF combinado
                total_questions=total_questions,
                format=pdf_format,
                generated_at=datetime.now().isoformat(),
                file_name=pdf_path.name,
                file_path=str(pdf_path)
            )

            pdf_metadata_list.append(metadata)

            print(f"✅ PDF combinado generado: {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error generando PDF combinado: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Generar un PDF por cada tema (comportamiento anterior)
        for topic, questions in sorted(questions_by_topic.items()):
            print(f"\n{'─'*70}")
            print(f"Generando PDF para Tema {topic} ({len(questions)} preguntas)...")

            try:
                # Generar PDF
                pdf_path = _generate_topic_pdf(
                    topic=topic,
                    questions=questions,
                    output_dir=output_dir,
                    pdf_format=pdf_format
                )

                # Crear metadata
                metadata = PDFMetadata(
                    topic=topic,
                    total_questions=len(questions),
                    format=pdf_format,
                    generated_at=datetime.now().isoformat(),
                    file_name=pdf_path.name,
                    file_path=str(pdf_path)
                )

                pdf_metadata_list.append(metadata)

                print(f"✅ PDF generado: {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error generando PDF para tema {topic}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ PDFs GENERADOS: {len(pdf_metadata_list)}")
    print(f"{'='*70}")

    return {
        "pdf_metadata": pdf_metadata_list,
        "decisions_log": add_decision_log(state, "generate_pdfs", "completed", {
            "pdfs_generated": len(pdf_metadata_list),
            "single_pdf_mode": single_pdf
        }),
        "node_count": state.get("node_count", 0) + 1
    }


def _generate_topic_pdf(
    topic: int,
    questions: List[Question],
    output_dir: Path,
    pdf_format: PDFFormatEnum
) -> Path:
    """Genera un PDF para un tema específico."""

    # Nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Tema_{topic}_{pdf_format.value}_{timestamp}.pdf"
    pdf_path = output_dir / file_name

    # Crear documento PDF
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch,
    )

    # Estilos
    styles = getSampleStyleSheet()

    # Estilo para título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Estilo para subtítulos
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor("#424242"),
        spaceAfter=10,
        spaceBefore=10
    )

    # Estilo para preguntas
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    # Estilo para opciones
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=13,
        leftIndent=20,
        spaceAfter=3
    )

    # Estilo para tips
    tip_style = ParagraphStyle(
        'TipStyle',
        parent=styles['BodyText'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1565c0"),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        spaceBefore=5,
        backColor=colors.HexColor("#e3f2fd"),
        borderPadding=5
    )

    # Elementos del PDF
    elements = []

    # Título
    elements.append(Paragraph(
        f"<b>Examen - Tema {topic}</b>",
        title_style
    ))

    # Información
    elements.append(Paragraph(
        f"<i>Total de preguntas: {len(questions)} | Formato: {pdf_format.value}</i>",
        subtitle_style
    ))

    elements.append(Paragraph(
        f"<i>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>",
        subtitle_style
    ))

    elements.append(Spacer(1, 0.3*inch))

    # Agregar preguntas según el formato
    if pdf_format == PDFFormatEnum.EXAM:
        elements.extend(_format_exam_only(questions, question_style, option_style))

    elif pdf_format == PDFFormatEnum.WITH_SOLUTIONS:
        # Preguntas
        elements.append(Paragraph("<b>PREGUNTAS</b>", subtitle_style))
        elements.extend(_format_exam_only(questions, question_style, option_style))

        # Page break
        elements.append(PageBreak())

        # Soluciones
        elements.append(Paragraph("<b>SOLUCIONES</b>", subtitle_style))
        elements.extend(_format_solutions(questions, question_style, tip_style))

    elif pdf_format == PDFFormatEnum.ONLY_ANSWERS:
        elements.extend(_format_solutions(questions, question_style, tip_style))

    elif pdf_format == PDFFormatEnum.STUDY_GUIDE:
        elements.extend(_format_study_guide(questions, question_style, option_style, tip_style))

    elif pdf_format == PDFFormatEnum.STUDY_GUIDE_WITH_CHUNKS:
        # Crear estilos adicionales para chunks
        chunk_style = ParagraphStyle(
            'ChunkStyle',
            parent=styles['BodyText'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#455a64"),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=5,
            backColor=colors.HexColor("#fff9c4"),
            borderPadding=8,
            borderColor=colors.HexColor("#fbc02d"),
            borderWidth=1
        )
        elements.extend(_format_study_guide_with_chunks(questions, question_style, option_style, tip_style, chunk_style))

        # Añadir página de estadísticas con histograma
        elements.append(PageBreak())
        elements.extend(_create_statistics_page(questions, title_style, subtitle_style))

    # Construir PDF
    doc.build(elements)

    return pdf_path


def _generate_combined_pdf(
    questions: List[Question],
    output_dir: Path,
    pdf_format: PDFFormatEnum,
    num_topics: int
) -> Path:
    """Genera un PDF único combinando todas las preguntas de todos los temas."""

    # Nombre del archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Combined_{pdf_format.value}_{num_topics}topics_{timestamp}.pdf"
    pdf_path = output_dir / file_name

    # Crear documento PDF
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch,
    )

    # Estilos
    styles = getSampleStyleSheet()

    # Estilo para título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Estilo para subtítulos
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor("#424242"),
        spaceAfter=10,
        spaceBefore=10
    )

    # Estilo para preguntas
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    # Estilo para opciones
    option_style = ParagraphStyle(
        'OptionStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=13,
        leftIndent=20,
        spaceAfter=3
    )

    # Estilo para tips
    tip_style = ParagraphStyle(
        'TipStyle',
        parent=styles['BodyText'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1565c0"),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        spaceBefore=5,
        backColor=colors.HexColor("#e3f2fd"),
        borderPadding=5
    )

    # Elementos del PDF
    elements = []

    # Título
    elements.append(Paragraph(
        f"<b>Examen - {num_topics} Temas Combinados</b>",
        title_style
    ))

    # Información
    elements.append(Paragraph(
        f"<i>Total de preguntas: {len(questions)} | Formato: {pdf_format.value}</i>",
        subtitle_style
    ))

    elements.append(Paragraph(
        f"<i>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>",
        subtitle_style
    ))

    elements.append(Spacer(1, 0.3*inch))

    # Agregar preguntas según el formato
    if pdf_format == PDFFormatEnum.EXAM:
        elements.extend(_format_exam_only(questions, question_style, option_style))

    elif pdf_format == PDFFormatEnum.WITH_SOLUTIONS:
        # Preguntas
        elements.append(Paragraph("<b>PREGUNTAS</b>", subtitle_style))
        elements.extend(_format_exam_only(questions, question_style, option_style))

        # Page break
        elements.append(PageBreak())

        # Soluciones
        elements.append(Paragraph("<b>SOLUCIONES</b>", subtitle_style))
        elements.extend(_format_solutions(questions, question_style, tip_style))

    elif pdf_format == PDFFormatEnum.ONLY_ANSWERS:
        elements.extend(_format_solutions(questions, question_style, tip_style))

    elif pdf_format == PDFFormatEnum.STUDY_GUIDE:
        elements.extend(_format_study_guide(questions, question_style, option_style, tip_style))

    elif pdf_format == PDFFormatEnum.STUDY_GUIDE_WITH_CHUNKS:
        # Crear estilos adicionales para chunks
        chunk_style = ParagraphStyle(
            'ChunkStyle',
            parent=styles['BodyText'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#455a64"),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=5,
            backColor=colors.HexColor("#fff9c4"),
            borderPadding=8,
            borderColor=colors.HexColor("#fbc02d"),
            borderWidth=1
        )
        elements.extend(_format_study_guide_with_chunks(questions, question_style, option_style, tip_style, chunk_style))

        # Añadir página de estadísticas con histograma
        elements.append(PageBreak())
        elements.extend(_create_statistics_page(questions, title_style, subtitle_style))

    # Construir PDF
    doc.build(elements)

    return pdf_path


def _format_exam_only(questions, question_style, option_style):
    """Formato solo preguntas (sin respuestas)."""
    elements = []

    for i, q in enumerate(questions, 1):
        # Pregunta
        elements.append(Paragraph(
            f"<b>{i}. {q.question}</b>",
            question_style
        ))

        # Opciones
        options = [
            f"A) {_strip_option_prefix(q.answer1)}",
            f"B) {_strip_option_prefix(q.answer2)}",
            f"C) {_strip_option_prefix(q.answer3)}",
            f"D) {_strip_option_prefix(q.answer4)}"
        ]

        for option in options:
            elements.append(Paragraph(option, option_style))

        elements.append(Spacer(1, 0.15*inch))

    return elements


def _format_solutions(questions, question_style, tip_style):
    """Formato solo soluciones."""
    elements = []

    for i, q in enumerate(questions, 1):
        correct_letter = ['A', 'B', 'C', 'D'][q.solution - 1]
        correct_answer = _strip_option_prefix(q.get_correct_answer())

        elements.append(Paragraph(
            f"<b>{i}.</b> Respuesta correcta: <b>{correct_letter}</b> - {correct_answer}",
            question_style
        ))

        if q.tip:
            tip_text = _normalize_tip_for_pdf(q.tip, {
                "A": q.answer1,
                "B": q.answer2,
                "C": q.answer3,
                "D": q.answer4
            })
            elements.append(Paragraph(
                f"<i>💡 {tip_text}</i>",
                tip_style
            ))

        if q.article:
            elements.append(Paragraph(
                f"<i>📖 Artículo: {q.article}</i>",
                tip_style
            ))

        elements.append(Spacer(1, 0.1*inch))

    return elements


def _format_study_guide(questions, question_style, option_style, tip_style):
    """Formato guía de estudio (preguntas + respuestas inline)."""
    elements = []

    for i, q in enumerate(questions, 1):
        # Pregunta
        elements.append(Paragraph(
            f"<b>{i}. {q.question}</b>",
            question_style
        ))

        # Opciones con marcador de correcta
        correct_letter = ['A', 'B', 'C', 'D'][q.solution - 1]
        options = [
            (f"A) {_strip_option_prefix(q.answer1)}", 'A' == correct_letter),
            (f"B) {_strip_option_prefix(q.answer2)}", 'B' == correct_letter),
            (f"C) {_strip_option_prefix(q.answer3)}", 'C' == correct_letter),
            (f"D) {_strip_option_prefix(q.answer4)}", 'D' == correct_letter)
        ]

        for option_text, is_correct in options:
            if is_correct:
                elements.append(Paragraph(
                    f"<b>✓ {option_text}</b> <i>(Correcta)</i>",
                    option_style
                ))
            else:
                elements.append(Paragraph(option_text, option_style))

        # Tip
        if q.tip:
            tip_text = _normalize_tip_for_pdf(q.tip, {
                "A": q.answer1,
                "B": q.answer2,
                "C": q.answer3,
                "D": q.answer4
            })
            elements.append(Paragraph(
                f"<i>💡 {tip_text}</i>",
                tip_style
            ))

        # Artículo
        if q.article:
            elements.append(Paragraph(
                f"<i>📖 {q.article}</i>",
                tip_style
            ))

        # Dificultad evaluada por Agente C
        difficulty = getattr(q, "difficulty", None)
        difficulty_reason = getattr(q, "difficulty_reason", None)
        if difficulty:
            difficulty_emoji = {
                "fácil": "🟢", "facil": "🟢",
                "medio": "🟡",
                "difícil": "🔴", "dificil": "🔴"
            }.get(difficulty.lower(), "⚪")
            difficulty_text = f"<b>{difficulty_emoji} Dificultad: {difficulty.capitalize()}</b>"
            if difficulty_reason:
                difficulty_text += f" - <i>{difficulty_reason}</i>"
            elements.append(Paragraph(difficulty_text, tip_style))

        elements.append(Spacer(1, 0.2*inch))

    return elements


def _create_statistics_page(questions, title_style, subtitle_style):
    """Crea página de estadísticas con histograma de tiempos."""
    elements = []

    # Título
    elements.append(Paragraph("<b>Estadísticas de Generación</b>", title_style))
    elements.append(Spacer(1, 0.3*inch))

    # Recopilar tiempos
    generation_times = [q.generation_time for q in questions if q.generation_time is not None]

    if not generation_times:
        elements.append(Paragraph(
            "<i>No hay datos de tiempos de generación disponibles.</i>",
            subtitle_style
        ))
        return elements

    # Estadísticas básicas
    avg_time = sum(generation_times) / len(generation_times)
    min_time = min(generation_times)
    max_time = max(generation_times)
    total_time = sum(generation_times)

    stats_text = f"""
    <b>Resumen de Tiempos:</b><br/>
    • Total de preguntas: {len(questions)}<br/>
    • Tiempo total: {total_time:.2f}s<br/>
    • Tiempo promedio: {avg_time:.2f}s<br/>
    • Tiempo mínimo: {min_time:.2f}s<br/>
    • Tiempo máximo: {max_time:.2f}s<br/>
    """
    elements.append(Paragraph(stats_text, subtitle_style))
    elements.append(Spacer(1, 0.3*inch))

    # Generar histograma
    try:
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 5))

        # Crear histograma
        ax.hist(generation_times, bins=min(10, len(generation_times)),
                color='#1565c0', alpha=0.7, edgecolor='black')

        # Configurar
        ax.set_xlabel('Tiempo de generación (segundos)', fontsize=12)
        ax.set_ylabel('Número de preguntas', fontsize=12)
        ax.set_title('Distribución de Tiempos de Generación', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Añadir línea de promedio
        ax.axvline(avg_time, color='red', linestyle='--', linewidth=2,
                   label=f'Promedio: {avg_time:.2f}s')
        ax.legend()

        # Guardar en buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)

        # Añadir imagen al PDF
        img = Image(img_buffer, width=6*inch, height=3.75*inch)
        elements.append(img)

    except Exception as e:
        logger.error(f"Error generando histograma: {e}")
        elements.append(Paragraph(
            f"<i>Error al generar histograma: {e}</i>",
            subtitle_style
        ))

    return elements


def _format_study_guide_with_chunks(questions, question_style, option_style, tip_style, chunk_style):
    """Formato guía de estudio con chunks fuente."""
    elements = []
    review_style = ParagraphStyle(
        'ManualReviewStyle',
        parent=tip_style,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#b71c1c"),
        backColor=colors.HexColor("#ffebee"),
        borderColor=colors.HexColor("#d32f2f"),
        borderWidth=1,
        borderPadding=6,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=6,
        spaceAfter=10
    )
    eval_style = ParagraphStyle(
        'EvalCommentStyle',
        parent=tip_style,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#0d47a1"),
        backColor=colors.HexColor("#e3f2fd"),
        borderColor=colors.HexColor("#1565c0"),
        borderWidth=1,
        borderPadding=6,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=6,
        spaceAfter=10
    )

    for i, q in enumerate(questions, 1):
        # Pregunta con tiempo de generación
        time_parts = []
        if q.generation_time:
            time_parts.append(f"Gen {q.generation_time:.2f}s")
        review_time = getattr(q, "review_time", None)
        if review_time:
            time_parts.append(f"Rev {review_time:.2f}s")
        time_text = f" <i>(⏱️ {' | '.join(time_parts)})</i>" if time_parts else ""
        elements.append(Paragraph(
            f"<b>{i}. {q.question}</b>{time_text}",
            question_style
        ))

        # Opciones con checkmark en la correcta
        correct_letter = ['A', 'B', 'C', 'D'][q.solution - 1]
        options = [
            (f"A) {_strip_option_prefix(q.answer1)}", 'A' == correct_letter),
            (f"B) {_strip_option_prefix(q.answer2)}", 'B' == correct_letter),
            (f"C) {_strip_option_prefix(q.answer3)}", 'C' == correct_letter),
            (f"D) {_strip_option_prefix(q.answer4)}", 'D' == correct_letter)
        ]

        for option_text, is_correct in options:
            if is_correct:
                elements.append(Paragraph(
                    f"{option_text} <b>✓</b>",
                    option_style
                ))
            else:
                elements.append(Paragraph(option_text, option_style))

        # Tip
        if q.tip:
            tip_text = _normalize_tip_for_pdf(q.tip, {
                "A": q.answer1,
                "B": q.answer2,
                "C": q.answer3,
                "D": q.answer4
            })
            elements.append(Paragraph(
                f"<i>💡 Tip: {tip_text}</i>",
                tip_style
            ))

        # Artículo
        if q.article:
            elements.append(Paragraph(
                f"<i>📖 Artículo: {q.article}</i>",
                tip_style
            ))

        # Dificultad evaluada por Agente C
        difficulty = getattr(q, "difficulty", None)
        difficulty_reason = getattr(q, "difficulty_reason", None)
        if difficulty:
            # Emoji según dificultad
            difficulty_emoji = {
                "fácil": "🟢",
                "facil": "🟢",
                "medio": "🟡",
                "difícil": "🔴",
                "dificil": "🔴"
            }.get(difficulty.lower(), "⚪")
            difficulty_text = f"<b>{difficulty_emoji} Dificultad: {difficulty.capitalize()}</b>"
            if difficulty_reason:
                difficulty_text += f"<br/><i>Razón: {difficulty_reason}</i>"
            elements.append(Paragraph(difficulty_text, tip_style))

        # Indicador de múltiples chunks/artículos usados
        num_chunks = getattr(q, "num_chunks_used", None) or 1
        context_strategy = getattr(q, "context_strategy", None)
        if num_chunks > 1 or context_strategy == "multi_context":
            multi_text = f"<b>🔗 Pregunta multi-artículo:</b> Combina información de {num_chunks} fuentes/artículos relacionados"
            elements.append(Paragraph(multi_text, tip_style))

        # Manual review destacado
        if getattr(q, "needs_manual_review", False):
            reason = getattr(q, "manual_review_reason", "") or "Revisión manual solicitada."
            details = getattr(q, "manual_review_details", "") or ""
            details = _truncate_text(details, 600)
            review_text = f"<b>⚠️ REVISIÓN MANUAL</b>: {reason}"
            if details and details != reason:
                review_text += f"<br/><i>Detalle:</i> {details}"
            elements.append(Paragraph(review_text, review_style))
        else:
            comment = getattr(q, "review_comment", "") or ""
            details = getattr(q, "review_details", "") or ""
            details = _truncate_text(details, 600)
            if comment:
                eval_text = f"<b>🧠 Evaluación C</b>: {comment}"
                if details and details != comment:
                    eval_text += f"<br/><i>Detalle:</i> {details}"
                elements.append(Paragraph(eval_text, eval_style))

        # Chunk fuente (si está disponible)
        if hasattr(q, 'source_chunk') and q.source_chunk:
            # Mostrar chunk completo sin truncar
            chunk_text = q.source_chunk

            elements.append(Paragraph(
                f"<i>📄 Chunk fuente: {chunk_text}</i>",
                chunk_style
            ))

        elements.append(Spacer(1, 0.4*inch))

    return elements


# ==========================================
# 5. CONSTRUCCIÓN DEL GRAFO
# ==========================================

def build_agent_e():
    """Construye y compila el grafo del Agente E."""
    workflow = StateGraph(AgentEState)

    # Añadir nodos
    workflow.add_node("group_by_topic", group_by_topic_node)
    workflow.add_node("generate_pdfs", generate_pdfs_node)

    # Definir flujo lineal
    workflow.set_entry_point("group_by_topic")
    workflow.add_edge("group_by_topic", "generate_pdfs")
    workflow.add_edge("generate_pdfs", END)

    return workflow.compile()


# ==========================================
# 6. CLASE PRINCIPAL DEL AGENTE
# ==========================================

class PDFGeneratorAgent:
    """Agente E: Genera PDFs de exámenes por tema."""

    def __init__(self):
        """Inicializa el agente."""
        self.graph = build_agent_e()
        logger.info("Initialized PDFGeneratorAgent")

    def generate_pdfs(
        self,
        questions: List[Question],
        output_dir: Path = None,
        pdf_format: PDFFormatEnum = PDFFormatEnum.WITH_SOLUTIONS,
        topic_filter: Optional[int] = None,
        single_pdf: bool = False
    ) -> List[PDFMetadata]:
        """Genera PDFs agrupados por tema o un único PDF combinado.

        Args:
            questions: Lista de preguntas
            output_dir: Directorio de salida
            pdf_format: Formato del PDF
            topic_filter: Filtrar solo un tema (opcional)
            single_pdf: Generar un único PDF en lugar de separados por tema

        Returns:
            Lista de metadata de PDFs generados
        """
        if output_dir is None:
            output_dir = Path("output/pdfs")

        log_separator(f"AGENTE E: PDF GENERATOR", "█")

        # Estado inicial
        initial_state = {
            "questions": questions,
            "output_dir": output_dir,
            "pdf_format": pdf_format,
            "topic_filter": topic_filter,
            "single_pdf": single_pdf,
            "questions_by_topic": {},
            "pdf_metadata": [],
            "decisions_log": [],
            "node_count": 0
        }

        # Ejecutar grafo
        final_state = self.graph.invoke(initial_state)

        # Log de recorrido
        print("\n📊 RECORRIDO DEL GRAFO:")
        for log in final_state.get("decisions_log", []):
            print(f"   {log['timestamp'][-8:]} | {log['node']:25} | {log['decision']}")

        return final_state.get("pdf_metadata", [])


# ==========================================
# 7. FUNCIONES DE CONVENIENCIA
# ==========================================

def generate_pdfs_from_questions(
    questions: List[Question],
    output_dir: str = "output/pdfs",
    pdf_format: str = "with_solutions",
    topic_filter: Optional[int] = None,
    single_pdf: bool = False
) -> List[PDFMetadata]:
    """Función standalone para generar PDFs.

    Args:
        questions: Lista de preguntas
        output_dir: Directorio de salida
        pdf_format: Formato: exam, with_solutions, only_answers, study_guide
        topic_filter: Filtrar solo un tema
        single_pdf: Generar un único PDF en lugar de separados por tema

    Returns:
        Lista de metadata de PDFs generados
    """
    agent = PDFGeneratorAgent()

    format_enum = PDFFormatEnum(pdf_format)

    return agent.generate_pdfs(
        questions=questions,
        output_dir=Path(output_dir),
        pdf_format=format_enum,
        topic_filter=topic_filter,
        single_pdf=single_pdf
    )


__all__ = [
    'PDFGeneratorAgent',
    'PDFFormatEnum',
    'generate_pdfs_from_questions',
]
