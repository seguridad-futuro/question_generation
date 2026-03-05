#!/usr/bin/env python3
"""
ORQUESTADOR PRINCIPAL - Pipeline de Generación de Preguntas

Este script coordina todo el pipeline:
1. Lee PDFs desde input_docs/
2. Los divide en chunks coherentes (Agent Z)
3. Genera preguntas (Agent B)
4. Evalúa calidad (Agent C)
5. Persiste en SQLite (Agent D)
6. Marca documentos procesados en questions_done.csv

Uso:
    python main.py                          # Procesa todos los PDFs pendientes
    python main.py --doc archivo.pdf        # Procesa un archivo específico
    python main.py --questions 10           # 10 preguntas por documento
    python main.py --topic 101 --academy 1  # Especifica topic y academy
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ==========================================
# CONSTANTES (configuración no en .env)
# ==========================================

# Archivo de tracking de documentos procesados
DONE_FILE = "questions_done.csv"

# Preguntas por documento (default)
DEFAULT_QUESTIONS_PER_DOC = 5

# Preguntas por chunk (default)
DEFAULT_QUESTIONS_PER_CHUNK = 1

# Extensiones de archivo soportadas
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

# Máximo de reintentos por pregunta fallida
MAX_RETRIES_PER_QUESTION = 2

# Activar/desactivar validación del Agente C (default: True)
ENABLE_AGENT_C_VALIDATION = True

# Logging level
LOG_LEVEL = logging.INFO

# ==========================================
# IMPORTS DEL PROYECTO
# ==========================================

from config.settings import get_settings
from agents.agent_z_rewriter import RewriterAgent
from agents.agent_b_generator import generate_questions_for_chunks, generate_questions_for_chunk, QuestionData
from agents.agent_c_evaluator import evaluate_question
from agents.agent_d_persistence import AgentD, persist_validated_questions
from agents.agent_e_pdf_generator import PDFGeneratorAgent, PDFFormatEnum
from models.chunk import Chunk
from models.question import Question

# ==========================================
# LOGGING SETUP
# ==========================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Orchestrator")


def log_banner(text: str, char: str = "█"):
    """Imprime banner visual."""
    width = 70
    print(f"\n{char * width}")
    padding = (width - len(text) - 2) // 2
    print(f"{char}{' ' * padding}{text}{' ' * (width - padding - len(text) - 2)}{char}")
    print(f"{char * width}\n")


# ==========================================
# TRACKING DE DOCUMENTOS PROCESADOS
# ==========================================

def get_done_file_path() -> Path:
    """Retorna path al archivo de tracking."""
    settings = get_settings()
    return settings.project_root / DONE_FILE


def load_done_documents() -> List[str]:
    """Carga lista de documentos ya procesados."""
    done_file = get_done_file_path()
    if not done_file.exists():
        return []

    done_docs = []
    with open(done_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_docs.append(row['filename'])
    return done_docs


def mark_document_done(filename: str, questions_generated: int, questions_persisted: int):
    """Marca un documento como procesado."""
    done_file = get_done_file_path()
    file_exists = done_file.exists()

    with open(done_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'processed_at', 'questions_generated', 'questions_persisted']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'filename': filename,
            'processed_at': datetime.now().isoformat(),
            'questions_generated': questions_generated,
            'questions_persisted': questions_persisted
        })

    logger.info(f"✅ Marcado como procesado: {filename}")


def get_pending_documents() -> List[Path]:
    """Obtiene documentos pendientes de procesar."""
    settings = get_settings()
    input_dir = settings.input_docs_dir

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Creado directorio: {input_dir}")
        return []

    # Obtener documentos ya procesados
    done_docs = load_done_documents()

    # Buscar documentos pendientes
    pending = []
    for ext in SUPPORTED_EXTENSIONS:
        for file in input_dir.glob(f"*{ext}"):
            if file.name not in done_docs:
                pending.append(file)

    return sorted(pending)


# ==========================================
# CONVERSIÓN DE DATOS
# ==========================================

def question_data_to_dict(
    q: QuestionData,
    chunk: Chunk,
    context_text: Optional[str] = None,
    faithfulness: Optional[float] = None,
    relevancy: Optional[float] = None,
    needs_manual_review: bool = False
) -> Dict[str, Any]:
    """Convierte QuestionData a dict para persistencia."""
    return {
        "question": q.question,
        "answer1": q.answer1,
        "answer2": q.answer2,
        "answer3": q.answer3,
        "answer4": q.answer4,
        "correct": q.correct,
        "tip": q.tip,
        "article": q.article,
        "source_chunk_id": chunk.chunk_id,
        "source_chunk_ids": q.source_chunk_ids,
        "source_chunk_indices": q.source_chunk_indices,
        "source_doc_ids": q.source_doc_ids,
        "context_strategy": q.context_strategy,
        "source_document": chunk.source_document,
        "source_chunk": context_text or chunk.content,  # ✨ AÑADIR CHUNK PARA PDF
        "generation_method": q.generation_method,
        "generation_time": q.generation_time,  # ⏱️ AÑADIR TIEMPO
        "faithfulness_score": faithfulness,
        "relevancy_score": relevancy,
        "needs_manual_review": needs_manual_review
    }


def _question_data_to_model(
    q: QuestionData,
    chunk: Chunk,
    academy: int,
    topic: int,
    context_text: str
) -> Question:
    """Convierte QuestionData en Question para evaluación del Agente C."""
    return Question(
        academy=academy,
        topic=topic,
        question=q.question,
        answer1=q.answer1,
        answer2=q.answer2,
        answer3=q.answer3,
        answer4=q.answer4,
        solution=q.correct,
        tip=q.tip or "",
        article=q.article or "",
        source_chunk_id=chunk.chunk_id,
        source_document=chunk.source_document,
        source_chunk=context_text,
        generation_time=q.generation_time or 0.0
    )


def _resolve_base_chunk(
    q: QuestionData,
    chunks_by_id: Dict[str, Chunk],
    fallback_chunk: Optional[Chunk] = None
) -> Optional[Chunk]:
    """Selecciona el chunk base para la pregunta."""
    if q.source_chunk_ids:
        base_chunk = chunks_by_id.get(q.source_chunk_ids[0])
        if base_chunk:
            return base_chunk
    return fallback_chunk


def _build_context_text(
    q: QuestionData,
    chunks_by_id: Dict[str, Chunk],
    fallback_chunk: Optional[Chunk] = None
) -> str:
    """Construye el contexto combinado respetando el orden de chunks."""
    context_parts = []
    for chunk_id in q.source_chunk_ids or []:
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            context_parts.append(chunk.content)
    if not context_parts and fallback_chunk:
        context_parts = [fallback_chunk.content]
    return "\n\n---\n\n".join(context_parts)


def _print_agent_c_reasoning(evaluation) -> None:
    """Imprime razonamiento resumido del Agente C."""
    if evaluation is None:
        return

    if isinstance(evaluation, dict):
        classification = evaluation.get("classification", "N/A")
        print(f"         🧠 Agente C: {classification}")
        metrics = evaluation.get("metrics")
        if metrics:
            try:
                print(
                    f"         🔎 Scores: faithfulness={metrics.faithfulness:.2f}, "
                    f"relevancy={metrics.answer_relevancy:.2f}"
                )
            except Exception:
                pass
        feedback = evaluation.get("feedback") or ""
        if feedback:
            feedback = feedback.strip()
            if len(feedback) > 200:
                feedback = feedback[:200] + "..."
            print(f"         📝 Feedback: {feedback}")
        return

    decision = evaluation.overall_decision or evaluation.classification
    print(f"         🧠 Agente C: {evaluation.classification} ({decision})")

    if evaluation.confidence is not None:
        print(f"         🔎 Confianza: {evaluation.confidence:.2f}")

    details = evaluation.validation_details or {}
    if isinstance(details, dict) and details:
        for key, detail in details.items():
            if isinstance(detail, dict):
                score = detail.get("score", "")
                reasoning = detail.get("reasoning", "")
            else:
                score = ""
                reasoning = str(detail)
            reasoning = reasoning.strip()
            if len(reasoning) > 160:
                reasoning = reasoning[:160] + "..."
            label = f"{key}: {score}".strip()
            print(f"         - {label} {reasoning}")

    if evaluation.improvement_feedback:
        feedback = evaluation.improvement_feedback.strip()
        if len(feedback) > 200:
            feedback = feedback[:200] + "..."
        print(f"         📝 Feedback: {feedback}")

    if evaluation.error:
        print(f"         ⚠️ Error: {evaluation.error}")


# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

def process_document(
    doc_path: Path,
    topic: int,
    academy: int,
    questions_per_doc: int = DEFAULT_QUESTIONS_PER_DOC,
    questions_per_chunk: int = DEFAULT_QUESTIONS_PER_CHUNK,
    enable_agent_c: bool = True
) -> Dict[str, Any]:
    """Procesa un documento completo.

    Args:
        doc_path: Path al documento
        topic: ID del topic
        academy: ID de la academia
        questions_per_doc: Número total de preguntas a generar
        questions_per_chunk: Preguntas por chunk

    Returns:
        Dict con estadísticas del procesamiento
    """
    log_banner(f"PROCESANDO: {doc_path.name}")

    stats = {
        "document": doc_path.name,
        "chunks": 0,
        "questions_generated": 0,
        "questions_persisted": 0,
        "errors": []
    }

    # ─────────────────────────────────────────
    # PASO 1: CHUNKING (Agent Z)
    # ─────────────────────────────────────────
    print("\n📄 PASO 1: Generando chunks coherentes...")

    try:
        agent_z = RewriterAgent()
        chunks = agent_z.create_coherent_chunks(doc_path, topic=topic)
        # Filtrar solo chunks de alta calidad
        high_quality_chunks = [c for c in chunks if c.metadata.get("coherent")]
        stats["chunks"] = len(high_quality_chunks)
        print(f"   ✅ {len(high_quality_chunks)} chunks coherentes generados (de {len(chunks)} totales)")
    except Exception as e:
        logger.error(f"Error en chunking: {e}")
        stats["errors"].append(f"Chunking: {e}")
        return stats

    chunks = high_quality_chunks

    # ─────────────────────────────────────────
    # PASO 2 & 3: GENERACIÓN (Agent B) + VALIDACIÓN (Agent C)
    # ─────────────────────────────────────────
    print("\n🤖 PASO 2-3: Generando y validando preguntas...")

    all_questions = []
    questions_needed = questions_per_doc
    max_generation_rounds = max(1, MAX_RETRIES_PER_QUESTION + 1)
    generation_round = 0
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    while questions_needed > 0 and generation_round < max_generation_rounds:
        generation_round += 1
        if generation_round > 1:
            print(f"\n   🔄 Ronda adicional de generación ({generation_round}/{max_generation_rounds})")

        candidates = generate_questions_for_chunks(
            chunks=chunks,
            num_questions=questions_needed,
            topic=f"Topic {topic}",
            questions_per_chunk=questions_per_chunk
        )

        if not candidates:
            print("   ❌ No se generaron candidatos en esta ronda.")
            break

        for q_index, q in enumerate(candidates, 1):
            if questions_needed <= 0:
                break

            if not q or q.error:
                print(f"   ❌ Candidato {q_index}: error en generación")
                continue

            base_chunk = _resolve_base_chunk(q, chunks_by_id, chunks[0] if chunks else None)
            if base_chunk is None:
                print(f"   ❌ Candidato {q_index}: no se pudo resolver chunk base")
                continue

            context_text = _build_context_text(q, chunks_by_id, base_chunk)
            context_ids = q.source_chunk_ids or [base_chunk.chunk_id]
            context_indices = q.source_chunk_indices or []
            context_doc_ids = q.source_doc_ids or []

            print(f"\n   {'='*80}")
            print(f"   📝 Pregunta candidata {q_index} | Chunks usados: {', '.join(context_ids)}")
            if context_indices:
                print(f"   📍 Índices: {', '.join(str(i) for i in context_indices)}")
            if context_doc_ids:
                print(f"   📚 Doc IDs: {', '.join(context_doc_ids)}")
            print(f"   {'='*80}")
            print(f"      Pregunta: {q.question}")
            print(f"      A) {q.answer1}")
            print(f"      B) {q.answer2}")
            print(f"      C) {q.answer3}")
            print(f"      D) {q.answer4}")
            print(f"      Correcta: {chr(64 + q.correct)}")
            print(f"      Artículo: {q.article}")
            if q.tip:
                print(f"      Tip: {q.tip[:150]}...")

            evaluation = None
            needs_manual_review = False
            final_q = q
            final_context_text = context_text

            if enable_agent_c:
                print(f"         🔍 Evaluación del Agente C...")
                attempt = 0
                while True:
                    temp_question = _question_data_to_model(
                        final_q,
                        base_chunk,
                        academy=academy,
                        topic=topic,
                        context_text=final_context_text
                    )
                    evaluation = evaluate_question(
                        temp_question,
                        base_chunk,
                        retry_count=attempt,
                        max_retries=MAX_RETRIES_PER_QUESTION,
                        context_override=final_context_text
                    )

                    _print_agent_c_reasoning(evaluation)

                    classification = evaluation.get("classification")
                    if classification == "auto_pass":
                        print(f"         ✅ Agente C: Pregunta válida")
                        break
                    if classification == "manual_review":
                        needs_manual_review = True
                        print(f"         ⚠️ Marcada para revisión manual")
                        break

                    print(f"         ❌ RECHAZADA por Agente C")
                    feedback = (evaluation.get("feedback") or "").strip()
                    if not feedback:
                        feedback = (
                            "Corrige referencias legales, precisión técnica, tip y distractores; "
                            "la respuesta correcta debe estar explícitamente en el contexto."
                        )

                    if attempt >= MAX_RETRIES_PER_QUESTION:
                        final_q = None
                        break

                    print(f"         📝 Feedback para Agente B: {feedback[:200]}")
                    regenerated = generate_questions_for_chunk(
                        chunk=base_chunk,
                        num_questions=1,
                        topic=f"Topic {topic}",
                        review_feedback=feedback
                    )
                    if not regenerated or not regenerated[0] or regenerated[0].error:
                        final_q = None
                        break

                    attempt += 1
                    final_q = regenerated[0]
                    final_context_text = _build_context_text(final_q, chunks_by_id, base_chunk)
                    print(f"         🔁 Regenerada #{attempt}: {final_q.question[:120]}...")

                if final_q is None:
                    continue
            else:
                print(f"         ⏭️  Agente C desactivado: aprobando sin validación")

            q = final_q
            context_text = final_context_text
            context_ids = q.source_chunk_ids or [base_chunk.chunk_id]
            context_indices = q.source_chunk_indices or []
            context_doc_ids = q.source_doc_ids or []

            q_dict = question_data_to_dict(
                q,
                base_chunk,
                context_text=context_text,
                faithfulness=evaluation["metrics"].faithfulness if evaluation else None,
                relevancy=evaluation["metrics"].answer_relevancy if evaluation else None,
                needs_manual_review=needs_manual_review
            )
            all_questions.append(q_dict)
            questions_needed -= 1

        if questions_needed > 0:
            print(f"\n   ⚠️ Faltan {questions_needed} preguntas por completar")

    stats["questions_generated"] = len(all_questions)
    print(f"\n   📊 Total preguntas VALIDADAS: {len(all_questions)}")

    if not all_questions:
        logger.warning("No se generaron preguntas válidas")
        return stats

    validated_questions = all_questions

    # ─────────────────────────────────────────
    # PASO 4: PERSISTENCIA (Agent D)
    # ─────────────────────────────────────────
    print("\n💾 PASO 4: Persistiendo en SQLite...")

    batch_name = f"{doc_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = {"inserted": 0, "duplicates": 0}
    try:
        result = persist_validated_questions(
            validated_questions=validated_questions,
            topic=topic,
            academy=academy,
            batch_name=batch_name
        )
        stats["questions_persisted"] = result["inserted"]
        print(f"   ✅ {result['inserted']} preguntas persistidas")
        print(f"   ⏭️  {result['duplicates']} duplicadas (omitidas)")
    except Exception as e:
        logger.error(f"Error en persistencia: {e}")
        stats["errors"].append(f"Persistence: {e}")

    # ─────────────────────────────────────────
    # PASO 5: GENERAR PDF (Agent E)
    # ─────────────────────────────────────────
    if result["inserted"] > 0:
        print("\n📄 PASO 5: Generando PDF del examen...")

        try:
            # Convertir dicts a objetos Question
            questions_for_pdf = []
            for q_dict in validated_questions:
                q = Question(
                    academy=academy,
                    topic=topic,
                    question=q_dict["question"],
                    answer1=q_dict["answer1"],
                    answer2=q_dict["answer2"],
                    answer3=q_dict["answer3"],
                    answer4=q_dict["answer4"],
                    solution=q_dict["correct"],
                    tip=q_dict.get("tip", ""),
                    article=q_dict.get("article", ""),
                    source_chunk=q_dict.get("source_chunk", ""),  # ✨ INCLUIR CHUNK
                    generation_time=q_dict.get("generation_time")  # ⏱️ INCLUIR TIEMPO
                )
                questions_for_pdf.append(q)

            # Generar PDF
            agent_e = PDFGeneratorAgent()
            output_dir = Path(f"output/pdfs/tema_{topic}")

            metadata = agent_e.generate_pdfs(
                questions=questions_for_pdf,
                output_dir=output_dir,
                pdf_format=PDFFormatEnum.STUDY_GUIDE_WITH_CHUNKS,
                topic_filter=topic
            )

            if metadata:
                stats["pdf_generated"] = metadata[0].file_path
                print(f"   ✅ PDF generado: {metadata[0].file_name}")
                print(f"   📁 Ubicación: {metadata[0].file_path}")
        except Exception as e:
            logger.error(f"Error generando PDF: {e}")
            stats["errors"].append(f"PDF Generation: {e}")

    return stats


def run_pipeline(
    documents: List[Path] = None,
    topic: int = 1,
    academy: int = 1,
    questions_per_doc: int = DEFAULT_QUESTIONS_PER_DOC,
    enable_agent_c: bool = True
):
    """Ejecuta el pipeline completo.

    Args:
        documents: Lista de documentos a procesar (None = todos los pendientes)
        topic: ID del topic
        academy: ID de la academia
        questions_per_doc: Número de preguntas por documento
        enable_agent_c: Si True, valida con Agente C. Si False, Z -> B -> D -> E
        academy: ID de la academia
        questions_per_doc: Preguntas por documento
    """
    log_banner("PIPELINE DE GENERACIÓN DE PREGUNTAS", "═")

    settings = get_settings()

    print(f"""
📋 CONFIGURACIÓN:
   • Modelo LLM: {settings.openai_model}
   • Topic: {topic}
   • Academy: {academy}
   • Preguntas/doc: {questions_per_doc}
   • Agente C: {'✅ ACTIVADO' if enable_agent_c else '⏭️ DESACTIVADO (Z -> B -> D -> E)'}
   • Input dir: {settings.input_docs_dir}
   • Database: {settings.database_path}
""")

    # Obtener documentos a procesar
    if documents is None:
        documents = get_pending_documents()

    if not documents:
        print("⚠️  No hay documentos pendientes de procesar.")
        print(f"   Añade PDFs a: {settings.input_docs_dir}")
        print(f"   O elimina entradas de: {DONE_FILE}")
        return

    print(f"📁 DOCUMENTOS A PROCESAR: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.name}")

    # Procesar cada documento
    all_stats = []

    for doc_path in documents:
        try:
            stats = process_document(
                doc_path=doc_path,
                topic=topic,
                academy=academy,
                questions_per_doc=questions_per_doc,
                enable_agent_c=enable_agent_c
            )
            all_stats.append(stats)

            # Marcar como procesado
            if stats["questions_persisted"] > 0 or stats["questions_generated"] > 0:
                mark_document_done(
                    filename=doc_path.name,
                    questions_generated=stats["questions_generated"],
                    questions_persisted=stats["questions_persisted"]
                )

        except Exception as e:
            logger.error(f"Error procesando {doc_path.name}: {e}")
            all_stats.append({
                "document": doc_path.name,
                "error": str(e)
            })

    # ─────────────────────────────────────────
    # RESUMEN FINAL
    # ─────────────────────────────────────────
    log_banner("RESUMEN FINAL", "═")

    total_generated = sum(s.get("questions_generated", 0) for s in all_stats)
    total_persisted = sum(s.get("questions_persisted", 0) for s in all_stats)
    total_chunks = sum(s.get("chunks", 0) for s in all_stats)

    print(f"""
📊 ESTADÍSTICAS TOTALES:
   • Documentos procesados: {len(all_stats)}
   • Chunks generados: {total_chunks}
   • Preguntas generadas: {total_generated}
   • Preguntas persistidas: {total_persisted}
""")

    print("📄 DETALLE POR DOCUMENTO:")
    for stats in all_stats:
        status = "✅" if not stats.get("errors") else "⚠️"
        print(f"   {status} {stats['document']}: "
              f"{stats.get('questions_persisted', 0)}/{stats.get('questions_generated', 0)} preguntas")
        if stats.get("errors"):
            for err in stats["errors"]:
                print(f"      ❌ {err}")

    # Mostrar stats de la BD
    agent_d = AgentD()
    db_stats = agent_d.get_stats()
    print(f"""
💾 ESTADO DE LA BASE DE DATOS:
   • Total preguntas: {db_stats['database']['total']}
   • Únicas: {db_stats['database']['unique']}
   • Duplicadas: {db_stats['database']['duplicates']}
   • Para revisión manual: {db_stats['database']['manual_review']}
""")


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Generación de Preguntas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                              # Procesa todos los PDFs pendientes
  python main.py --doc tema1.pdf              # Procesa un archivo específico
  python main.py --questions 20               # 20 preguntas por documento
  python main.py --topic 101 --academy 1      # Especifica topic y academy
  python main.py --list                       # Lista documentos pendientes
        """
    )

    parser.add_argument(
        "--doc", "-d",
        type=str,
        help="Documento a procesar (match parcial, ej: 'Tema-18' o path completo)"
    )

    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=DEFAULT_QUESTIONS_PER_DOC,
        help=f"Número de preguntas por documento (default: {DEFAULT_QUESTIONS_PER_DOC})"
    )

    parser.add_argument(
        "--topic", "-t",
        type=int,
        default=1,
        help="ID del topic (default: 1)"
    )

    parser.add_argument(
        "--academy", "-a",
        type=int,
        default=1,
        help="ID de la academia (default: 1)"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Lista documentos pendientes y sale"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina el archivo de tracking (permite reprocesar todo)"
    )

    parser.add_argument(
        "--skip-agent-c",
        action="store_true",
        help="Omite la validación del Agente C (Z -> B -> D -> E directamente)"
    )

    args = parser.parse_args()

    # Opción: Reset tracking
    if args.reset:
        done_file = get_done_file_path()
        if done_file.exists():
            done_file.unlink()
            print(f"✅ Eliminado: {done_file}")
        else:
            print("ℹ️  No existe archivo de tracking")
        return

    # Opción: Listar pendientes
    if args.list:
        pending = get_pending_documents()
        done = load_done_documents()

        print("\n📋 DOCUMENTOS PENDIENTES:")
        if pending:
            for doc in pending:
                print(f"   • {doc.name}")
        else:
            print("   (ninguno)")

        print(f"\n✅ DOCUMENTOS PROCESADOS ({len(done)}):")
        for doc in done[:10]:
            print(f"   • {doc}")
        if len(done) > 10:
            print(f"   ... y {len(done) - 10} más")
        return

    # Opción: Documento específico
    documents = None
    if args.doc:
        settings = get_settings()
        doc_path = Path(args.doc)

        # Si es path absoluto o existe directamente, usarlo
        if doc_path.is_absolute() and doc_path.exists():
            documents = [doc_path]
        elif (settings.input_docs_dir / args.doc).exists():
            documents = [settings.input_docs_dir / args.doc]
        else:
            # Match parcial: buscar archivos que contengan el texto
            doc_filter = args.doc.lower()
            matching_files = []
            for ext in SUPPORTED_EXTENSIONS:
                for f in settings.input_docs_dir.glob(f"*{ext}"):
                    if doc_filter in f.name.lower():
                        matching_files.append(f)

            if matching_files:
                documents = sorted(set(matching_files))
                print(f"📂 Archivos que coinciden con '{args.doc}':")
                for f in documents:
                    print(f"   • {f.name}")
            else:
                print(f"❌ No se encontraron archivos que coincidan con '{args.doc}'")
                print(f"   Archivos disponibles en {settings.input_docs_dir}:")
                for ext in SUPPORTED_EXTENSIONS:
                    for f in sorted(settings.input_docs_dir.glob(f"*{ext}")):
                        print(f"   • {f.name}")
                sys.exit(1)

    # Ejecutar pipeline
    run_pipeline(
        documents=documents,
        topic=args.topic,
        academy=args.academy,
        questions_per_doc=args.questions,
        enable_agent_c=not args.skip_agent_c  # Si skip=True, entonces enable=False
    )


if __name__ == "__main__":
    main()
