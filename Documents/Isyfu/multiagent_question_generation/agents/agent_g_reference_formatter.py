"""Agente G: Reference Formatter - Limpia y formatea las referencias legales.

Detecta y corrige artefactos de OCR/libros en el campo 'article' de las preguntas:
- Marcadores de página ("--- Página X ---")
- Errores de OCR ("E1" -> "El", "l." -> "1.")
- Referencias a editoriales, URLs, emails
- Texto cortado a mitad de frase
- Artículos excesivamente largos (los resume)

Usa OpenAI para reformatear solo las referencias problemáticas.
"""

import logging
import os
import re
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from config.settings import get_settings

logger = logging.getLogger("AgentG")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)

# Patterns that indicate a problematic reference
ARTIFACT_PATTERNS = [
    r"---\s*Página\s*\d+\s*---",
    r"---\s*Pág\.?\s*\d+\s*---",
    r"Aspirantes\.es",
    r"info@",
    r"www\.",
    r"https?://",
    r"editorial",
    r"ISBN",
    r"©\s*\d{4}",
    r"\bE1\b",                     # OCR: "E1" instead of "El"
    r"(?<!\d)l\.\s+[A-Z]",        # OCR: "l." instead of "1."
    r"consentimi\s*ento",          # OCR word splits
    r"(?:Capítulo|CAPITULO)\s+[IVXLC]+\b",  # Chapter headers from book
]

ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)

# Max reasonable length for a formatted reference
MAX_ARTICLE_LENGTH = 800


def needs_formatting(article: str) -> tuple[bool, list[str]]:
    """Check if an article reference needs formatting.

    Returns (needs_fix, reasons).
    """
    if not article or not article.strip():
        return False, []

    reasons = []

    # Check artifact patterns
    matches = ARTIFACT_RE.findall(article)
    if matches:
        reasons.append(f"artefactos: {matches[:3]}")

    # Too long
    if len(article) > MAX_ARTICLE_LENGTH:
        reasons.append(f"demasiado largo ({len(article)} chars)")

    # Truncated mid-sentence (ends without punctuation)
    stripped = article.rstrip()
    if stripped and stripped[-1] not in ".;:)»\"'":
        # Check if it looks like it was cut
        if len(stripped) > 100:
            reasons.append("texto truncado")

    # Multiple newlines (formatting artifacts)
    if "\n\n\n" in article:
        reasons.append("saltos de linea excesivos")

    return bool(reasons), reasons


def format_reference_with_llm(client: OpenAI, model: str, article: str, question: str) -> str:
    """Use LLM to clean and reformat a problematic reference."""
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=2000,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un formateador de referencias legales para preguntas de examen de oposiciones. "
                    "Tu tarea es limpiar y reformatear la referencia legal dada, eliminando:\n"
                    "- Marcadores de página (--- Página X ---)\n"
                    "- Cabeceras de capítulos o secciones del libro\n"
                    "- Errores de OCR (E1->El, l.->1., palabras partidas)\n"
                    "- Referencias a editoriales, URLs, emails\n"
                    "- Texto irrelevante que no forma parte del artículo\n\n"
                    "Reglas:\n"
                    "1. Mantén SOLO el texto legal relevante del artículo citado\n"
                    "2. Si el artículo es muy largo, resume manteniendo los apartados clave "
                    "relacionados con la pregunta\n"
                    "3. Corrige errores de OCR evidentes\n"
                    "4. Formatea limpiamente: 'Artículo X. Título.\\n1. Contenido...'\n"
                    "5. Si el texto termina cortado, cierra con '(...)'\n"
                    "6. Máximo 500 caracteres aproximadamente\n"
                    "7. NO inventes texto. Solo limpia y recorta lo existente\n"
                    "8. Responde SOLO con el artículo formateado, sin explicaciones"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PREGUNTA: {question}\n\n"
                    f"REFERENCIA A LIMPIAR:\n{article}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


class AgentG:
    """Agente de formateo de referencias legales."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or str(settings.database_path)
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.stats = {
            "total_checked": 0,
            "needs_formatting": 0,
            "formatted": 0,
            "errors": 0,
            "skipped_ok": 0,
        }

    def scan_topic(self, topic: Optional[int] = None) -> list[dict]:
        """Scan questions and return those with problematic references."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if topic:
            rows = conn.execute(
                "SELECT id, topic, question, article FROM questions WHERE topic = ? AND article IS NOT NULL AND article != ''",
                (topic,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, topic, question, article FROM questions WHERE article IS NOT NULL AND article != ''"
            ).fetchall()
        conn.close()

        problematic = []
        for row in rows:
            self.stats["total_checked"] += 1
            needs_fix, reasons = needs_formatting(row["article"])
            if needs_fix:
                self.stats["needs_formatting"] += 1
                problematic.append({
                    "id": row["id"],
                    "topic": row["topic"],
                    "question": row["question"],
                    "article": row["article"],
                    "reasons": reasons,
                })
            else:
                self.stats["skipped_ok"] += 1

        return problematic

    def format_question(self, q_id: int, question: str, article: str) -> Optional[str]:
        """Format a single question's reference. Returns new article or None on error."""
        try:
            formatted = format_reference_with_llm(self.client, self.model, article, question)
            if formatted and len(formatted) > 10:
                self.stats["formatted"] += 1
                return formatted
            else:
                self.stats["errors"] += 1
                return None
        except Exception as e:
            logger.error(f"Error formatting Q{q_id}: {e}")
            self.stats["errors"] += 1
            return None

    def update_article(self, q_id: int, new_article: str):
        """Update the article field in the database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE questions SET article = ? WHERE id = ?", (new_article, q_id))
        conn.commit()
        conn.close()

    def format_topic(self, topic: Optional[int] = None, dry_run: bool = False) -> dict:
        """Scan and format all problematic references for a topic (or all topics).

        Args:
            topic: Topic number to format, or None for all topics
            dry_run: If True, don't update the database

        Returns:
            Stats dict
        """
        scope = f"Tema {topic}" if topic else "todos los temas"
        print(f"\n{'='*60}")
        print(f"  AGENTE G: Formateando referencias - {scope}")
        print(f"{'='*60}\n")

        problematic = self.scan_topic(topic)
        print(f"  Revisadas: {self.stats['total_checked']}")
        print(f"  Correctas: {self.stats['skipped_ok']}")
        print(f"  Necesitan formato: {len(problematic)}\n")

        if not problematic:
            print("  Todas las referencias son correctas.")
            return self.stats

        for i, q in enumerate(problematic, 1):
            print(f"  [{i}/{len(problematic)}] Q{q['id']} (Tema {q['topic']})")
            print(f"    Razones: {', '.join(q['reasons'])}")
            print(f"    Pregunta: {q['question'][:80]}...")

            formatted = self.format_question(q["id"], q["question"], q["article"])
            if formatted:
                print(f"    Antes:   {q['article'][:100]}...")
                print(f"    Despues: {formatted[:100]}...")
                if not dry_run:
                    self.update_article(q["id"], formatted)
                    print(f"    -> Actualizado en BD")
                else:
                    print(f"    -> [DRY RUN] No guardado")
            else:
                print(f"    -> Error formateando, se mantiene original")
            print()

        print(f"\n{'='*60}")
        print(f"  RESUMEN:")
        print(f"    Revisadas:   {self.stats['total_checked']}")
        print(f"    Correctas:   {self.stats['skipped_ok']}")
        print(f"    Formateadas: {self.stats['formatted']}")
        print(f"    Errores:     {self.stats['errors']}")
        print(f"{'='*60}\n")

        return self.stats


def format_references(topic: Optional[int] = None, dry_run: bool = False) -> dict:
    """Helper function to format references from the pipeline or GUI."""
    load_dotenv()
    agent = AgentG()
    return agent.format_topic(topic=topic, dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent G: Format legal references")
    parser.add_argument("--topic", type=int, default=None, help="Topic number (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    args = parser.parse_args()

    load_dotenv()
    format_references(topic=args.topic, dry_run=args.dry_run)
