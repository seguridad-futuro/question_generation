"""Script para extraer preguntas de SQLite y generar Excels por tema."""

import sqlite3
from pathlib import Path

from models.question import Question
from agents.agent_f_excel_generator import ExcelGeneratorAgent, ExcelFormatEnum


def extract_questions_from_db(db_path: Path, limit: int = 100) -> list[Question]:
    """Extrae preguntas de la base de datos SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

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
========================================================================
         EXTRACCION DE PREGUNTAS Y GENERACION EXCEL POR TEMA
========================================================================
""")

    db_path = Path("database/questions.db")

    if not db_path.exists():
        print(f"No se encontro la base de datos en: {db_path}")
        return

    print(f"Base de datos: {db_path}")

    print("\nExtrayendo 100 preguntas...")
    questions = extract_questions_from_db(db_path, limit=100)

    print(f"Extraidas {len(questions)} preguntas")

    if not questions:
        print("No hay preguntas en la base de datos")
        return

    # Resumen por tema
    topics = {}
    for q in questions:
        topics[q.topic] = topics.get(q.topic, 0) + 1

    print("\nDistribucion por tema:")
    for topic, count in sorted(topics.items()):
        print(f"   Tema {topic}: {count} preguntas")

    # Generar Excels
    print("\nGenerando Excels...")

    output_dir = Path("output/excels")
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = ExcelGeneratorAgent()

    excel_metadata = agent.generate_excels(
        questions=questions,
        output_dir=output_dir,
        excel_format=ExcelFormatEnum.WITH_SOLUTIONS,
    )

    print("\n" + "=" * 70)
    print("GENERACION COMPLETADA")
    print("=" * 70)

    for meta in excel_metadata:
        print(f"""
   {meta.file_name}
      Tema: {meta.topic}
      Preguntas: {meta.total_questions}
      Formato: {meta.format.value}
      Ruta: {meta.file_path}
""")


if __name__ == "__main__":
    main()