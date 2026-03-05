"""Agente D: Persistence Agent - Guarda preguntas en SQLite.

Este agente recibe preguntas validadas y las persiste en la base de datos,
manejando deduplicación y estadísticas.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.question import Question
from database.repository import QuestionRepository
from config.settings import get_settings

# ==========================================
# LOGGING
# ==========================================

logger = logging.getLogger("AgentD")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '\n%(asctime)s | %(name)s | %(levelname)s\n%(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


def log_separator(title: str = "", char: str = "=", width: int = 70):
    """Imprime separador visual."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


# ==========================================
# AGENT D: PERSISTENCE
# ==========================================

class AgentD:
    """Agente de persistencia a SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        """Inicializa el agente con repositorio."""
        self.repo = QuestionRepository(db_path)
        self.stats = {
            "total_received": 0,
            "inserted": 0,
            "duplicates": 0,
            "errors": 0,
            "verified": 0,
            "verification_errors": 0
        }

    def _verify_inserted(self, question_id: int, question_data: Dict[str, Any]) -> bool:
        """Verifica que la pregunta insertada existe y coincide con los datos."""
        row = self.repo.get_by_id(question_id)
        if not row:
            logger.error(f"❌ Verificación SQLite: ID={question_id} no encontrado")
            return False

        mismatches = []
        fields_to_check = ["question", "answer1", "answer2", "answer3"]
        for field in fields_to_check:
            expected = question_data.get(field, "")
            actual = row.get(field, "")
            if expected != actual:
                mismatches.append(field)

        expected_answer4 = question_data.get("answer4")
        if expected_answer4 is not None:
            actual_answer4 = row.get("answer4")
            if expected_answer4 != actual_answer4:
                mismatches.append("answer4")

        expected_correct = question_data.get("correct")
        if expected_correct is not None:
            actual_correct = row.get("solution")
            if expected_correct != actual_correct:
                mismatches.append("correct")

        expected_chunk_id = question_data.get("source_chunk_id")
        if expected_chunk_id and row.get("source_chunk_id") != expected_chunk_id:
            mismatches.append("source_chunk_id")

        expected_doc = question_data.get("source_document")
        if expected_doc and row.get("source_document") != expected_doc:
            mismatches.append("source_document")

        if mismatches:
            logger.error(
                f"❌ Verificación SQLite: ID={question_id} campos distintos: {', '.join(mismatches)}"
            )
            return False

        return True

    def persist_question(
        self,
        question_data: Dict[str, Any],
        chunk_id: str,
        source_document: str,
        topic: int,
        academy: int,
        faithfulness: float = None,
        relevancy: float = None,
        batch_name: str = None
    ) -> Optional[int]:
        """Persiste una pregunta individual.

        Args:
            question_data: Dict con question, answer1-4, correct, tip, article
            chunk_id: ID del chunk fuente
            source_document: Documento fuente
            topic: ID del topic
            academy: ID de la academia
            faithfulness: Score de faithfulness (0-1)
            relevancy: Score de relevancy (0-1)
            batch_name: Nombre del batch

        Returns:
            ID de la pregunta insertada o None si duplicada/error
        """
        self.stats["total_received"] += 1

        try:
            # Convertir QuestionData a Question model
            question = Question(
                academy=academy,
                topic=topic,
                question=question_data.get("question", ""),
                answer1=question_data.get("answer1", ""),
                answer2=question_data.get("answer2", ""),
                answer3=question_data.get("answer3", ""),
                answer4=question_data.get("answer4"),
                solution=question_data.get("correct", 1),
                tip=question_data.get("tip", ""),
                llm_model=get_settings().generation_model or get_settings().openai_model,
                source_chunk_id=chunk_id,
                source_document=source_document,
                faithfulness_score=faithfulness,
                relevancy_score=relevancy,
                by_llm=True,
                needs_manual_review=question_data.get("needs_manual_review", False)
            )

            # Añadir article si existe
            if "article" in question_data:
                question.article = question_data["article"]

            # Verificar duplicado
            if self.repo.exists(question.question):
                self.stats["duplicates"] += 1
                logger.info(f"⏭️ Pregunta duplicada: {question.question[:50]}...")
                return None

            # Insertar
            question_id = self.repo.insert(question, batch_name)
            self.stats["inserted"] += 1
            logger.info(f"✅ Pregunta insertada ID={question_id}")

            if self._verify_inserted(question_id, question_data):
                self.stats["verified"] += 1
            else:
                self.stats["verification_errors"] += 1
            return question_id

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"❌ Error persistiendo: {e}")
            return None

    def persist_batch(
        self,
        questions: List[Dict[str, Any]],
        topic: int,
        academy: int,
        batch_name: str = None
    ) -> Dict[str, Any]:
        """Persiste un batch de preguntas.

        Args:
            questions: Lista de dicts con datos de preguntas
            topic: ID del topic
            academy: ID de la academia
            batch_name: Nombre del batch (auto-generado si None)

        Returns:
            Dict con estadísticas del batch
        """
        log_separator("AGENT D: PERSISTENCE", "█")

        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"""
📦 PERSISTIENDO BATCH:
   • Nombre: {batch_name}
   • Preguntas: {len(questions)}
   • Topic: {topic}
   • Academy: {academy}
""")

        inserted_ids = []
        verified_start = self.stats["verified"]
        verification_errors_start = self.stats["verification_errors"]
        for i, q_data in enumerate(questions, 1):
            print(f"   [{i}/{len(questions)}] Procesando...", end=" ")

            q_id = self.persist_question(
                question_data=q_data,
                chunk_id=q_data.get("source_chunk_id", f"chunk_{i}"),
                source_document=q_data.get("source_document", "unknown"),
                topic=topic,
                academy=academy,
                faithfulness=q_data.get("faithfulness_score"),
                relevancy=q_data.get("relevancy_score"),
                batch_name=batch_name
            )

            if q_id:
                inserted_ids.append(q_id)
                print(f"✅ ID={q_id}")
            else:
                print("⏭️ Skipped")

        # Estadísticas finales
        result = {
            "batch_name": batch_name,
            "total": len(questions),
            "inserted": len(inserted_ids),
            "duplicates": len(questions) - len(inserted_ids),
            "ids": inserted_ids,
            "verified": self.stats["verified"] - verified_start,
            "verification_errors": self.stats["verification_errors"] - verification_errors_start
        }

        log_separator("RESUMEN PERSISTENCIA", "─")
        print(f"""
📊 ESTADÍSTICAS:
   • Total recibidas: {result['total']}
   • Insertadas: {result['inserted']}
   • Duplicadas: {result['duplicates']}
   • Verificadas: {result['verified']}
   • Errores verificación: {result['verification_errors']}
   • IDs: {result['ids'][:5]}{'...' if len(result['ids']) > 5 else ''}
""")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del agente y la BD."""
        db_stats = self.repo.get_stats()
        return {
            "session": self.stats,
            "database": db_stats
        }


# ==========================================
# FUNCIÓN HELPER PARA INTEGRACIÓN
# ==========================================

def persist_validated_questions(
    validated_questions: List[Dict],
    topic: int,
    academy: int,
    batch_name: str = None
) -> Dict[str, Any]:
    """Helper para persistir preguntas validadas desde el pipeline.

    Args:
        validated_questions: Lista de preguntas que pasaron Agent C
        topic: ID del topic
        academy: ID de la academia
        batch_name: Nombre del batch

    Returns:
        Estadísticas de persistencia
    """
    agent = AgentD()
    return agent.persist_batch(validated_questions, topic, academy, batch_name)


# ==========================================
# MAIN (TEST)
# ==========================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              AGENTE D - PERSISTENCE (TEST MODE)                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Pregunta de prueba
    test_questions = [
        {
            "question": "¿Cuál es la pena por homicidio según el artículo 138 del Código Penal?",
            "answer1": "Prisión de 10 a 15 años",
            "answer2": "Prisión de 5 a 10 años",
            "answer3": "Prisión de 15 a 20 años",
            "answer4": "Prisión de 20 a 25 años",
            "correct": 1,
            "tip": "El artículo 138 CP establece pena de 10 a 15 años",
            "article": "Artículo 138.1: El que matare a otro será castigado, como reo de homicidio, con la pena de prisión de diez a quince años.",
            "source_chunk_id": "test_chunk_001",
            "source_document": "codigo_penal.pdf",
            "faithfulness_score": 0.95,
            "relevancy_score": 0.90
        },
        {
            "question": "¿Qué circunstancia convierte un homicidio en asesinato según el artículo 139?",
            "answer1": "La alevosía",
            "answer2": "La nocturnidad",
            "answer3": "El uso de armas",
            "answer4": "La reincidencia",
            "correct": 1,
            "tip": "El artículo 139 CP establece que la alevosía es una circunstancia de asesinato",
            "article": "Artículo 139: Será castigado con la pena de prisión de quince a veinticinco años, como reo de asesinato, el que matare a otro concurriendo alguna de las circunstancias siguientes: 1.ª Con alevosía.",
            "source_chunk_id": "test_chunk_002",
            "source_document": "codigo_penal.pdf",
            "faithfulness_score": 0.92,
            "relevancy_score": 0.88
        }
    ]

    result = persist_validated_questions(
        validated_questions=test_questions,
        topic=101,
        academy=1,
        batch_name="test_batch"
    )

    print(f"\n✅ Test completado: {result}")

    # Mostrar stats de la BD
    agent = AgentD()
    stats = agent.get_stats()
    print(f"\n📊 Stats BD: {stats['database']}")
