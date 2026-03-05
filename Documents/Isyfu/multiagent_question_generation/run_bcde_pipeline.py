#!/usr/bin/env python3
"""Run pipeline B->C->D->E using cached chunks (Agent Z skipped)."""

import argparse
import json
import threading
import time
from queue import Queue, Empty
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings
from models.chunk import Chunk
from models.question import Question
from agents.agent_b_generator import generate_questions_for_chunk, QuestionData
from agents.agent_c_evaluator import evaluate_question
from agents.agent_d_persistence import persist_validated_questions
from agents.agent_e_pdf_generator import PDFGeneratorAgent, PDFFormatEnum
from services.chunk_retriever import get_chunk_retriever_service


MAX_RETRIES_PER_QUESTION = 2

def _load_chunks_from_cache_file(path: Path) -> Tuple[List[Chunk], Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for chunk_dict in data.get("chunks", []):
        chunk = Chunk(
            chunk_id=chunk_dict["chunk_id"],
            content=chunk_dict["content"],
            source_document=chunk_dict["source_document"],
            page=chunk_dict.get("page"),
            token_count=chunk_dict.get("token_count"),
            metadata=chunk_dict.get("metadata", {})
        )
        chunks.append(chunk)
    return chunks, data


def _load_chunk_usage_cache(path: Path) -> Dict:
    if not path.exists():
        return {"updated_at": None, "chunks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_chunk_usage_cache(path: Path, cache: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["updated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _mark_chunk_used(cache: Dict, chunk: Chunk) -> None:
    entry = cache["chunks"].get(chunk.chunk_id, {})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_used"] = datetime.now().isoformat()
    entry.setdefault("doc_id", chunk.metadata.get("doc_id"))
    entry.setdefault("chunk_index", chunk.metadata.get("chunk_index"))
    cache["chunks"][chunk.chunk_id] = entry


def _build_context_text(q: QuestionData, chunks_by_id: Dict[str, Chunk], fallback: Chunk) -> str:
    context_parts = []
    for chunk_id in q.source_chunk_ids or []:
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            context_parts.append(chunk.content)
    if not context_parts:
        context_parts = [fallback.content]
    return "\n\n---\n\n".join(context_parts)


def _question_data_to_dict(
    q: QuestionData,
    base_chunk: Chunk,
    context_text: str,
    evaluation=None,
    needs_manual_review: bool = False,
    review_time: Optional[float] = None,
    manual_review_reason: Optional[str] = None,
    manual_review_details: Optional[str] = None,
    generation_time_override: Optional[float] = None,
    review_comment: Optional[str] = None,
    review_details: Optional[str] = None,
    difficulty: Optional[str] = None,
    difficulty_reason: Optional[str] = None
) -> Dict[str, object]:
    faithfulness = None
    relevancy = None
    if evaluation:
        if isinstance(evaluation, dict) and evaluation.get("metrics"):
            metrics = evaluation["metrics"]
            faithfulness = getattr(metrics, "faithfulness", None)
            relevancy = getattr(metrics, "answer_relevancy", None)
        else:
            faithfulness = getattr(evaluation, "faithfulness_estimated", None)
            relevancy = getattr(evaluation, "relevancy_estimated", None)
    return {
        "question": q.question,
        "answer1": q.answer1,
        "answer2": q.answer2,
        "answer3": q.answer3,
        "answer4": q.answer4,
        "correct": q.correct,
        "tip": q.tip,
        "article": q.article,
        "source_chunk_id": base_chunk.chunk_id,
        "source_chunk_ids": q.source_chunk_ids,
        "source_chunk_indices": q.source_chunk_indices,
        "source_doc_ids": q.source_doc_ids,
        "context_strategy": q.context_strategy,
        "source_document": base_chunk.source_document,
        "source_chunk": context_text,
        "generation_method": q.generation_method,
        "generation_time": generation_time_override if generation_time_override is not None else q.generation_time,
        "faithfulness_score": faithfulness,
        "relevancy_score": relevancy,
        "needs_manual_review": needs_manual_review,
        "review_time": review_time,
        "manual_review_reason": manual_review_reason,
        "manual_review_details": manual_review_details,
        "review_comment": review_comment,
        "review_details": review_details,
        "difficulty": difficulty,
        "difficulty_reason": difficulty_reason
    }


def _partition_chunks(
    pending: List[Tuple[Chunk, int]],
    workers: int
) -> List[List[Tuple[Chunk, int]]]:
    if workers <= 1 or len(pending) <= 1:
        return [pending]
    workers = min(workers, len(pending))
    chunk_size = (len(pending) + workers - 1) // workers
    partitions = []
    for i in range(workers):
        start = i * chunk_size
        if start >= len(pending):
            break
        partitions.append(pending[start:start + chunk_size])
    return partitions


def _process_chunk_group(
    worker_id: int,
    group: List[Tuple[Chunk, int]],
    retriever,
    chunks_by_id: Dict[str, Chunk],
    topic: int,
    academy: int,
    skip_agent_c: bool,
    result_queue: Optional[Queue] = None
) -> Tuple[List[Dict[str, object]], List[str]]:
    results: List[Dict[str, object]] = []
    used_chunk_ids: List[str] = []

    for chunk, remaining in group:
        print(f"\n   🧩 [W{worker_id}] Chunk {chunk.chunk_id} | pendientes: {remaining}")
        chunk_results: List[Dict[str, object]] = []
        chunk_used_ids: List[str] = []
        for q_idx in range(remaining):
            try:
                b_start = time.perf_counter()
                questions = generate_questions_for_chunk(
                    chunk=chunk,
                    num_questions=1,
                    topic=f"Topic {topic}",
                    retriever_service=retriever,
                    show_summary=False
                )
                b_elapsed = time.perf_counter() - b_start
            except Exception as exc:
                print(f"      ❌ [W{worker_id}] Error generando: {exc}")
                continue

        for q in questions:
            if not q or q.error:
                print(f"      ❌ [W{worker_id}] Error generando pregunta")
                continue

            context_text = _build_context_text(q, chunks_by_id, chunk)
            evaluation = None
            needs_manual_review = False
            final_q = q
            final_context_text = context_text
            c_elapsed = 0.0
            b_retry_elapsed = 0.0
            manual_review_reason = None
            manual_review_details = None
            review_comment = None
            review_details = None

            if not skip_agent_c:
                attempt = 0
                while True:
                    attempt_label = "Reintento" if attempt > 0 else "Pregunta"
                    print(f"\n🧾 [W{worker_id}] {attempt_label} (chunk {chunk.chunk_id}, #{q_idx + 1}/{remaining}):")
                    print(f"❓ {final_q.question}")
                    print(f"   A) {final_q.answer1}")
                    print(f"   B) {final_q.answer2}")
                    print(f"   C) {final_q.answer3}")
                    if final_q.answer4:
                        print(f"   D) {final_q.answer4}")
                    print(
                        f"   🧪 [W{worker_id}] Agente C evaluando "
                        f"(intento {attempt + 1}/{MAX_RETRIES_PER_QUESTION + 1})..."
                    )
                    temp_question = Question(
                        academy=academy,
                        topic=topic,
                        question=final_q.question,
                        answer1=final_q.answer1,
                        answer2=final_q.answer2,
                        answer3=final_q.answer3,
                        answer4=final_q.answer4,
                        solution=final_q.correct,
                        tip=final_q.tip or "",
                        article=final_q.article or "",
                        source_chunk=final_context_text,
                        source_chunk_id=chunk.chunk_id,
                        source_document=chunk.source_document,
                        generation_time=final_q.generation_time or 0.0
                    )
                    try:
                        c_start = time.perf_counter()
                        evaluation = evaluate_question(
                            temp_question,
                            chunk,
                            retry_count=attempt,
                            max_retries=MAX_RETRIES_PER_QUESTION,
                            context_override=final_context_text
                        )
                        c_elapsed += time.perf_counter() - c_start
                    except Exception as exc:
                        print(f"      ❌ [W{worker_id}] Error en Agente C: {exc}")
                        evaluation = None
                        break

                    classification = evaluation.get("classification") if evaluation else None
                    if classification == "auto_pass":
                        correct_letter = ['A', 'B', 'C', 'D'][final_q.correct - 1]
                        difficulty_eval = evaluation.get("difficulty") if evaluation else None
                        difficulty_reason_eval = evaluation.get("difficulty_reason") if evaluation else None
                        diff_str = f" | 📊 Dificultad: {difficulty_eval}" if difficulty_eval else ""
                        print(f"   ✅ [W{worker_id}] Agente C: auto_pass (correcta {correct_letter}){diff_str}")
                        if difficulty_reason_eval:
                            print(f"   📊 [W{worker_id}] Razón dificultad: {difficulty_reason_eval}")
                        feedback_text = (evaluation.get("feedback") or "").strip() if evaluation else ""
                        if feedback_text:
                            review_comment = feedback_text
                            print(f"   📝 [W{worker_id}] Motivo C: {feedback_text}")
                        break
                    if classification == "manual_review":
                        needs_manual_review = True
                        difficulty_eval = evaluation.get("difficulty") if evaluation else None
                        difficulty_reason_eval = evaluation.get("difficulty_reason") if evaluation else None
                        diff_str = f" | 📊 Dificultad: {difficulty_eval}" if difficulty_eval else ""
                        print(f"   ⚠️ [W{worker_id}] Agente C: manual_review{diff_str}")
                        if difficulty_reason_eval:
                            print(f"   📊 [W{worker_id}] Razón dificultad: {difficulty_reason_eval}")
                        reasoning = evaluation.get("agent_reasoning") if evaluation else ""
                        feedback_text = (evaluation.get("feedback") or "").strip() if evaluation else ""
                        if feedback_text:
                            manual_review_reason = feedback_text
                            review_comment = feedback_text
                            print(f"   📝 [W{worker_id}] Motivo C: {manual_review_reason}")
                        elif reasoning:
                            manual_review_reason = reasoning
                            review_comment = reasoning
                            print(f"   📝 [W{worker_id}] Motivo C: {manual_review_reason}")
                        if reasoning:
                            try:
                                if isinstance(reasoning, dict):
                                    reason_text = json.dumps(reasoning, ensure_ascii=False, indent=2)
                                else:
                                    parsed = json.loads(reasoning)
                                    reason_text = json.dumps(parsed, ensure_ascii=False, indent=2)
                            except Exception:
                                reason_text = str(reasoning)
                            if len(reason_text) > 2000:
                                reason_text = reason_text[:2000] + "..."
                            print(f"   🧠 [W{worker_id}] Razonamiento Agente C:\n{reason_text}")
                            manual_review_details = reason_text
                            review_details = reason_text
                        if not manual_review_reason:
                            # Generar razón basada en métricas si están disponibles
                            metrics = evaluation.get("metrics") if evaluation else None
                            if metrics:
                                f_score = getattr(metrics, "faithfulness", None)
                                r_score = getattr(metrics, "answer_relevancy", None)
                                if f_score is not None or r_score is not None:
                                    scores_info = []
                                    if f_score is not None:
                                        scores_info.append(f"faithfulness={f_score:.2f}")
                                    if r_score is not None:
                                        scores_info.append(f"relevancy={r_score:.2f}")
                                    manual_review_reason = f"Scores intermedios ({', '.join(scores_info)}): requiere verificación manual del tip, artículo o posible ambigüedad."
                                else:
                                    manual_review_reason = "Evaluación incierta: verificar coherencia entre pregunta, tip y contexto."
                            else:
                                manual_review_reason = "Evaluación incierta: verificar coherencia entre pregunta, tip y contexto."
                            review_comment = manual_review_reason
                            print(f"   📝 [W{worker_id}] Motivo C: {manual_review_reason}")
                        break

                    print(f"   ❌ [W{worker_id}] Agente C: auto_fail")
                    reasoning = evaluation.get("agent_reasoning") if evaluation else ""
                    feedback_text = (evaluation.get("feedback") or "").strip() if evaluation else ""
                    if feedback_text:
                        print(f"   📝 [W{worker_id}] Motivo C: {feedback_text}")
                        review_comment = feedback_text
                    elif reasoning:
                        print(f"   📝 [W{worker_id}] Motivo C: {reasoning}")
                        review_comment = reasoning
                    if reasoning:
                        try:
                            if isinstance(reasoning, dict):
                                reason_text = json.dumps(reasoning, ensure_ascii=False, indent=2)
                            else:
                                parsed = json.loads(reasoning)
                                reason_text = json.dumps(parsed, ensure_ascii=False, indent=2)
                        except Exception:
                            reason_text = str(reasoning)
                            if len(reason_text) > 2000:
                                reason_text = reason_text[:2000] + "..."
                            print(f"   🧠 [W{worker_id}] Razonamiento Agente C:\n{reason_text}")
                            review_details = reason_text

                    feedback = (evaluation.get("feedback") or "").strip() if evaluation else ""
                    if not feedback:
                        feedback = (
                            "Corrige referencias legales, precisión técnica, tip y distractores; "
                                "la respuesta correcta debe estar explícitamente en el contexto."
                            )

                        if attempt >= MAX_RETRIES_PER_QUESTION:
                            final_q = None
                            break

                        print(f"   🔁 [W{worker_id}] Reintentando generación con feedback del Agente C...")
                        regen_start = time.perf_counter()
                        regenerated = generate_questions_for_chunk(
                            chunk=chunk,
                            num_questions=1,
                            topic=f"Topic {topic}",
                            retriever_service=retriever,
                            review_feedback=feedback,
                            show_summary=False
                        )
                        b_retry_elapsed += time.perf_counter() - regen_start
                        if not regenerated or not regenerated[0] or regenerated[0].error:
                            final_q = None
                            break

                        attempt += 1
                        final_q = regenerated[0]
                        final_context_text = _build_context_text(final_q, chunks_by_id, chunk)

                    if final_q is None:
                        continue
                    q = final_q
                    context_text = final_context_text

                total_elapsed = b_elapsed + b_retry_elapsed + c_elapsed
                retry_suffix = f" + reintentos: {b_retry_elapsed:.2f}s" if b_retry_elapsed > 0 else ""
                print(
                    f"   ⏱️ [W{worker_id}] Tiempo B: {b_elapsed:.2f}s"
                    f"{retry_suffix} | Tiempo C: {c_elapsed:.2f}s | Total: {total_elapsed:.2f}s"
                )

                total_b_time = b_elapsed + b_retry_elapsed
                # Extraer dificultad de la evaluación del Agente C
                difficulty_from_eval = evaluation.get("difficulty") if evaluation else None
                difficulty_reason_from_eval = evaluation.get("difficulty_reason") if evaluation else None

                q_dict = _question_data_to_dict(
                    q,
                    chunk,
                    context_text,
                    evaluation=evaluation,
                    needs_manual_review=needs_manual_review,
                    review_time=c_elapsed or None,
                    manual_review_reason=manual_review_reason,
                    manual_review_details=manual_review_details,
                    generation_time_override=total_b_time if total_b_time else None,
                    review_comment=review_comment,
                    review_details=review_details,
                    difficulty=difficulty_from_eval,
                    difficulty_reason=difficulty_reason_from_eval
                )
                results.append(q_dict)
                chunk_results.append(q_dict)

                for chunk_id in q.source_chunk_ids or [chunk.chunk_id]:
                    used_chunk_ids.append(chunk_id)
                    chunk_used_ids.append(chunk_id)

        # Enviar preguntas del chunk actual a la cola INMEDIATAMENTE
        if result_queue is not None and (chunk_results or chunk_used_ids):
            result_queue.put({
                "worker_id": worker_id,
                "questions": chunk_results,
                "used_chunk_ids": chunk_used_ids
            })

    return results, used_chunk_ids


def _to_question_model(q_dict: Dict, academy: int, topic: int) -> Question:
    # Calcular número de chunks usados
    chunk_ids = q_dict.get("source_chunk_ids") or []
    num_chunks = len(chunk_ids) if chunk_ids else 1

    return Question(
        academy=academy,
        topic=topic,
        question=q_dict["question"],
        answer1=q_dict["answer1"],
        answer2=q_dict["answer2"],
        answer3=q_dict["answer3"],
        answer4=q_dict.get("answer4"),
        solution=q_dict["correct"],
        tip=q_dict.get("tip", ""),
        article=q_dict.get("article", ""),
        source_chunk=q_dict.get("source_chunk", ""),
        generation_time=q_dict.get("generation_time"),
        review_time=q_dict.get("review_time"),
        manual_review_reason=q_dict.get("manual_review_reason"),
        manual_review_details=q_dict.get("manual_review_details"),
        needs_manual_review=q_dict.get("needs_manual_review", False),
        review_comment=q_dict.get("review_comment"),
        review_details=q_dict.get("review_details"),
        difficulty=q_dict.get("difficulty"),
        difficulty_reason=q_dict.get("difficulty_reason"),
        # Multi-chunk info
        source_chunk_ids=chunk_ids if chunk_ids else None,
        context_strategy=q_dict.get("context_strategy"),
        num_chunks_used=num_chunks
    )


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Pipeline B->C->D->E using cached chunks (Agent Z skipped)"
    )
    parser.add_argument("--topic", type=int, default=1, help="Topic ID")
    parser.add_argument("--academy", type=int, default=1, help="Academy ID")
    parser.add_argument("--doc", type=str, help="Filter by document name (partial match, e.g. 'Tema-18')")
    parser.add_argument("--questions-per-chunk", type=int, default=settings.bcde_questions_per_chunk)
    parser.add_argument("--skip-agent-c", action="store_true", help="Skip Agent C validation")
    parser.add_argument("--force", action="store_true", help="Ignore chunk cache and process all")
    parser.add_argument("--reset-cache", action="store_true", help="Reset chunk usage cache")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Limit number of chunks to process")
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.b_parallel_agents,
        help="Number of parallel agents"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cached chunk files and exit"
    )
    parser.add_argument(
        "--single-pdf",
        action="store_true",
        help="Generate all questions in a single PDF instead of separate PDFs per topic"
    )

    args = parser.parse_args()

    rewritten_dir = settings.input_docs_dir / "rewritten"
    if not rewritten_dir.exists():
        print(f"No rewritten cache directory found: {rewritten_dir}")
        return 1

    # Listar archivos disponibles si se usa --list
    if args.list:
        all_files = sorted(rewritten_dir.glob("*.json"))
        all_files = [f for f in all_files if not f.name.endswith("_chunks_metadata.json")]
        print(f"\n📂 Archivos JSON disponibles en {rewritten_dir}:\n")
        for f in all_files:
            print(f"   • {f.stem}")
        print(f"\n   Total: {len(all_files)} archivos")
        print(f"\n💡 Uso: python run_bcde_pipeline.py --doc Tema-18")
        return 0

    if settings.bcde_chunk_cache_path:
        cache_path = Path(settings.bcde_chunk_cache_path)
    else:
        cache_path = settings.output_dir / "chunk_question_cache.json"

    if args.reset_cache and cache_path.exists():
        cache_path.unlink()

    usage_cache = _load_chunk_usage_cache(cache_path)

    cache_files = sorted(rewritten_dir.glob("*.json"))
    cache_files = [
        f for f in cache_files
        if not f.name.endswith("_chunks_metadata.json") and not f.name.endswith(".coord.json")
    ]
    if args.doc:
        # Match parcial: busca archivos que contengan el texto en su nombre
        doc_filter = args.doc.lower()
        cache_files = [f for f in cache_files if doc_filter in f.stem.lower()]
        if cache_files:
            print(f"📂 Archivos que coinciden con '{args.doc}':")
            for f in cache_files:
                print(f"   • {f.name}")
        else:
            print(f"❌ No se encontraron archivos que coincidan con '{args.doc}'")
            print(f"   Archivos disponibles en {rewritten_dir}:")
            for f in sorted(rewritten_dir.glob("*.json")):
                print(f"   • {f.stem}")

    if not cache_files:
        print("No cached chunk files found.")
        return 0

    all_valid_questions: List[Dict[str, object]] = []
    persist_stats = {"inserted": 0, "duplicates": 0}
    batch_name = f"bcde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for cache_file in cache_files:
        if cache_file.name.endswith("_chunks_metadata.json") or cache_file.name.endswith(".coord.json"):
            continue

        chunks, cache_data = _load_chunks_from_cache_file(cache_file)
        if not chunks:
            continue

        print(f"\n📄 Procesando {cache_file.name} | chunks: {len(chunks)}")

        chunks_by_id = {c.chunk_id: c for c in chunks}
        pending_chunks = []

        for chunk in chunks:
            current = usage_cache.get("chunks", {}).get(chunk.chunk_id, {}).get("count", 0)
            remaining = args.questions_per_chunk - int(current)
            if args.force:
                remaining = args.questions_per_chunk
            if remaining > 0:
                pending_chunks.append((chunk, remaining))

        if args.limit_chunks > 0:
            pending_chunks = pending_chunks[: args.limit_chunks]

        if not pending_chunks:
            print("   ✅ Sin chunks pendientes")
            continue

        retriever = get_chunk_retriever_service(
            force_new=True,
            enable_embeddings=settings.b_enable_semantic_expansion
        )
        retriever.initialize(chunks)

        result_queue: Queue = Queue()
        stop_event = threading.Event()
        writer_error: List[Exception] = []  # Para capturar errores del writer

        def _writer_loop() -> None:
            """Writer loop con manejo robusto de errores y guardado garantizado."""
            items_processed = 0
            while not stop_event.is_set() or not result_queue.empty():
                try:
                    payload = result_queue.get(timeout=1.0)  # Timeout aumentado de 0.2 a 1.0
                except Empty:
                    continue
                if not payload:
                    result_queue.task_done()
                    continue

                try:
                    questions = payload.get("questions", [])
                    used_ids = payload.get("used_chunk_ids", [])

                    if questions:
                        result = persist_validated_questions(
                            validated_questions=questions,
                            topic=args.topic,
                            academy=args.academy,
                            batch_name=batch_name
                        )
                        persist_stats["inserted"] += result.get("inserted", 0)
                        persist_stats["duplicates"] += result.get("duplicates", 0)
                        all_valid_questions.extend(questions)

                    for chunk_id in used_ids:
                        base_chunk = chunks_by_id.get(chunk_id)
                        if base_chunk:
                            _mark_chunk_used(usage_cache, base_chunk)

                    items_processed += 1
                    # Guardar caché cada 5 items para no perder progreso
                    if items_processed % 5 == 0:
                        _save_chunk_usage_cache(cache_path, usage_cache)

                except Exception as exc:
                    print(f"   ⚠️ Error en writer_loop procesando payload: {exc}")
                    writer_error.append(exc)
                finally:
                    result_queue.task_done()

            # Guardado final al terminar el loop
            try:
                _save_chunk_usage_cache(cache_path, usage_cache)
                print(f"   💾 Writer finalizó: {items_processed} items procesados")
            except Exception as exc:
                print(f"   ⚠️ Error guardando caché final en writer: {exc}")
                writer_error.append(exc)

        writer_thread = threading.Thread(target=_writer_loop, daemon=False)
        writer_thread.start()

        # Procesamiento SECUENCIAL (sin paralelización)
        print(f"   🔄 Procesando secuencialmente {len(pending_chunks)} chunks")
        _process_chunk_group(
            1,
            pending_chunks,
            retriever,
            chunks_by_id,
            args.topic,
            args.academy,
            args.skip_agent_c,
            result_queue
        )

        # Esperar a que la queue se vacíe antes de señalar stop
        print(f"   ⏳ Esperando a que el writer procese todos los items...")
        try:
            result_queue.join()  # Esperar a que todos los items sean procesados
        except Exception as exc:
            print(f"   ⚠️ Error esperando queue.join(): {exc}")

        stop_event.set()

        # Join con timeout para evitar cuelgue infinito
        writer_thread.join(timeout=30.0)
        if writer_thread.is_alive():
            print(f"   ⚠️ Writer thread no terminó en 30s, continuando...")
        else:
            print(f"   ✅ Writer thread finalizado correctamente")

        # Verificar si hubo errores en el writer
        if writer_error:
            print(f"   ⚠️ Se encontraron {len(writer_error)} errores en el writer")

    if not all_valid_questions:
        print("\n⚠️ No se generaron preguntas válidas.")
        _save_chunk_usage_cache(cache_path, usage_cache)
        return 0

    print(f"\n💾 Persistidas: {persist_stats['inserted']} | Duplicadas: {persist_stats['duplicates']}")

    questions_for_pdf = [
        _to_question_model(q_dict, args.academy, args.topic)
        for q_dict in all_valid_questions
    ]

    # Generar PDF
    agent_e = PDFGeneratorAgent()
    if args.single_pdf:
        output_dir = Path(f"output/pdfs/combined")
        topic_filter = None
    else:
        output_dir = Path(f"output/pdfs/tema_{args.topic}")
        topic_filter = args.topic

    metadata = agent_e.generate_pdfs(
        questions=questions_for_pdf,
        output_dir=output_dir,
        pdf_format=PDFFormatEnum.STUDY_GUIDE_WITH_CHUNKS,
        topic_filter=topic_filter,
        single_pdf=args.single_pdf
    )
    if metadata:
        print(f"📄 PDF generado: {metadata[0].file_name}")

    # Guardar JSON con todos los atributos
    json_output_dir = Path(f"output/json/tema_{args.topic}")
    json_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file_path = json_output_dir / f"Tema_{args.topic}_questions_{timestamp}.json"

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "batch_name": batch_name,
                "topic": args.topic,
                "academy": args.academy,
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(all_valid_questions),
                "inserted": persist_stats['inserted'],
                "duplicates": persist_stats['duplicates']
            },
            "questions": all_valid_questions
        }, f, indent=2, ensure_ascii=False)

    print(f"📋 JSON generado: {json_file_path.name}")

    _save_chunk_usage_cache(cache_path, usage_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
