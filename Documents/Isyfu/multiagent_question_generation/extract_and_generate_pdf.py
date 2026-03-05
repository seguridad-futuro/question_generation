"""Script para extraer 100 preguntas de SQLite y generar PDF."""

import sqlite3
from pathlib import Path
from datetime import datetime

from models.question import Question
from agents.agent_e_pdf_generator import PDFGeneratorAgent, PDFFormatEnum


def extract_questions_from_db(db_path: Path, limit: int = 100) -> list[Question]:
    """Extrae preguntas de la base de datos SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Usar la vista questions_with_options que ya tiene las respuestas
    cursor = conn.execute("""
        SELECT * FROM questions_with_options
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    questions = []
    for row in rows:
        row_dict = dict(row)

        # Crear objeto Question
        q = Question(
            id=row_dict.get('id'),
            academy=row_dict.get('academy_id', 1),
            topic=row_dict.get('topic', 0),
            question=row_dict.get('question', ''),
            answer1=row_dict.get('answer1', ''),
            answer2=row_dict.get('answer2', ''),
            answer3=row_dict.get('answer3', ''),
            answer4=row_dict.get('answer4'),
            solution=row_dict.get('solution', 1),
            tip=row_dict.get('tip'),
            article=row_dict.get('article'),
            faithfulness_score=row_dict.get('faithfulness_score'),
            relevancy_score=row_dict.get('relevancy_score'),
            source_chunk_id=row_dict.get('source_chunk_id'),
            source_document=row_dict.get('source_document'),
            llm_model=row_dict.get('llm_model'),
        )
        questions.append(q)

    return questions


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXTRACCIÓN DE PREGUNTAS Y GENERACIÓN PDF                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Ruta a la base de datos
    db_path = Path("database/questions.db")

    if not db_path.exists():
        print(f"❌ No se encontró la base de datos en: {db_path}")
        return

    print(f"📂 Base de datos: {db_path}")

    # Extraer 100 preguntas
    print("\n📥 Extrayendo 100 preguntas...")
    questions = extract_questions_from_db(db_path, limit=100)

    print(f"✅ Extraídas {len(questions)} preguntas")

    if not questions:
        print("❌ No hay preguntas en la base de datos")
        return

    # Mostrar resumen por tema
    topics = {}
    for q in questions:
        topic = q.topic
        if topic not in topics:
            topics[topic] = 0
        topics[topic] += 1

    print("\n📊 Distribución por tema:")
    for topic, count in sorted(topics.items()):
        print(f"   • Tema {topic}: {count} preguntas")

    # Generar PDF
    print("\n📄 Generando PDF...")

    output_dir = Path("output/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = PDFGeneratorAgent()

    # Usar formato study_guide_with_chunks para incluir toda la información
    pdf_metadata = agent.generate_pdfs(
        questions=questions,
        output_dir=output_dir,
        pdf_format=PDFFormatEnum.STUDY_GUIDE,
    )

    print("\n" + "="*70)
    print("✅ GENERACIÓN COMPLETADA")
    print("="*70)

    for meta in pdf_metadata:
        print(f"""
   📄 {meta.file_name}
      • Tema: {meta.topic}
      • Preguntas: {meta.total_questions}
      • Formato: {meta.format.value}
      • Ruta: {meta.file_path}
""")


if __name__ == "__main__":
    main()