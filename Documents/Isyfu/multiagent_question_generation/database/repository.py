"""SQLite Repository for Questions persistence (Schema V2)."""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from models.question import Question
from config.settings import get_settings


class QuestionRepository:
    """Repository para persistir preguntas en SQLite (Schema V2)."""

    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Inicializa la BD con schema V2."""
        schema_path = Path(__file__).parent / "init_schema_v2.sql"
        if schema_path.exists():
            with self._get_connection() as conn:
                conn.executescript(schema_path.read_text())
                conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager para conexión SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert(self, question: Question, batch_name: str = None) -> int:
        """Inserta pregunta + opciones (schema V2). Retorna ID."""
        with self._get_connection() as conn:
            # 1. Insertar en questions
            cursor = conn.execute("""
                INSERT INTO questions (
                    question, tip, article, topic, academy_id,
                    llm_model, by_llm, generation_method,
                    faithfulness_score, relevancy_score,
                    source_chunk_id, source_document,
                    is_duplicate, duplicate_of, retry_count, needs_manual_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                question.question,
                question.tip,
                getattr(question, 'article', None),
                question.topic,
                question.academy,
                question.llm_model,
                question.by_llm,
                getattr(question, 'generation_method', 'single_context'),
                question.faithfulness_score,
                question.relevancy_score,
                question.source_chunk_id,
                question.source_document,
                question.is_duplicate,
                question.duplicate_of,
                question.retry_count,
                question.needs_manual_review
            ))
            question_id = cursor.lastrowid

            # 2. Insertar opciones en question_options
            answers = [
                (question.answer1, question.solution == 1, 1),
                (question.answer2, question.solution == 2, 2),
                (question.answer3, question.solution == 3, 3),
            ]
            if question.answer4:
                answers.append((question.answer4, question.solution == 4, 4))

            for answer, is_correct, order in answers:
                conn.execute("""
                    INSERT INTO question_options (question_id, answer, is_correct, option_order)
                    VALUES (?, ?, ?, ?)
                """, (question_id, answer, is_correct, order))

            conn.commit()
            return question_id

    def insert_batch(self, questions: List[Question], batch_name: str = None) -> Dict[str, Any]:
        """Inserta múltiples preguntas. Retorna estadísticas."""
        inserted_ids = []
        duplicates = 0

        for q in questions:
            try:
                if not self.exists(q.question):
                    id = self.insert(q, batch_name)
                    inserted_ids.append(id)
                else:
                    duplicates += 1
            except sqlite3.IntegrityError:
                duplicates += 1

        # Registrar batch
        if batch_name:
            self._register_batch(batch_name, questions, inserted_ids, duplicates)

        return {
            "inserted": len(inserted_ids),
            "duplicates": duplicates,
            "total": len(questions),
            "ids": inserted_ids
        }

    def _register_batch(self, batch_name: str, questions: List, ids: List, duplicates: int):
        """Registra metadata del batch."""
        with self._get_connection() as conn:
            avg_faith = sum(q.faithfulness_score or 0 for q in questions) / len(questions) if questions else 0
            avg_rel = sum(q.relevancy_score or 0 for q in questions) / len(questions) if questions else 0

            conn.execute("""
                INSERT OR REPLACE INTO batches (
                    batch_name, topic, academy_id, total_questions,
                    unique_questions, duplicates, avg_faithfulness, avg_relevancy, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed')
            """, (
                batch_name,
                questions[0].topic if questions else 0,
                questions[0].academy if questions else 1,
                len(questions),
                len(ids),
                duplicates,
                avg_faith,
                avg_rel
            ))
            conn.commit()

    def exists(self, question_text: str) -> bool:
        """Verifica si pregunta existe (texto exacto)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM questions WHERE question = ? LIMIT 1",
                (question_text,)
            )
            return cursor.fetchone() is not None

    def get_by_id(self, id: int) -> Optional[Dict]:
        """Obtiene pregunta con opciones por ID (vista questions_with_options)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM questions_with_options WHERE id = ?", (id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_batch(self, batch_name: str) -> List[Dict]:
        """Obtiene preguntas de un batch."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT q.* FROM questions q
                JOIN batches b ON q.source_document LIKE '%' || b.batch_name || '%'
                WHERE b.batch_name = ?
            """, (batch_name,))
            return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Total de preguntas."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas de la BD."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            duplicates = conn.execute("SELECT COUNT(*) FROM questions WHERE is_duplicate = 1").fetchone()[0]
            manual = conn.execute("SELECT COUNT(*) FROM questions WHERE needs_manual_review = 1").fetchone()[0]
            return {"total": total, "unique": total - duplicates, "duplicates": duplicates, "manual_review": manual}
