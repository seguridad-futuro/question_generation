"""
Extract topic titles for Guardia Civil exam topics.

Strategy (in order):
  1. Web search: fetch official topic titles from internet
  2. Vision: render PDF cover pages and read the title from the concept map
  3. Chunks: infer the title from the first text chunks of the rewritten JSON

Usage:
    python scripts/extract_topic_titles.py [--input-dir INPUT_DIR] [--output OUTPUT] [--force] [--pages N] [--topics 1,3,5]
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# Resolve paths relative to the project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT_DIR = str(PROJECT_ROOT / "input_docs")
DEFAULT_REWRITTEN_DIR = str(PROJECT_ROOT / "input_docs" / "rewritten")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "topic_titles.json")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Titles that are too generic / likely a failed extraction
BAD_PATTERNS = [
    r"^tema\s*\d",
    r"^ingreso",
    r"^guardia\s*civil",
    r"^temario",
    r"^aspirantes",
    r"^cuerpo\s*de",
]

SYSTEM_PROMPT = (
    "Eres un asistente que extrae titulos de temas de oposiciones. "
    "Los titulos SIEMPRE son la materia o area tematica general, NUNCA el nombre de una ley. "
    "Ejemplos CORRECTOS: 'Derechos Humanos', 'La Constitucion Española', 'Derecho Penal', "
    "'Proteccion de Datos', 'Derecho Administrativo', 'Topografia', 'Violencia de Genero', "
    "'Extranjeria', 'Proteccion Civil', 'Telecomunicaciones', 'Armas y Explosivos'. "
    "Ejemplos INCORRECTOS (nunca respondas asi): "
    "'Ley Organica 3/2007...', 'Ley 39/2015...', 'Ley 31/1995...'. "
    "Responde SOLO con el titulo corto de la materia (2-6 palabras), "
    "sin numero de tema, sin 'Tema X', sin nombres de leyes, sin comillas, sin explicaciones."
)


def is_bad_title(title: str) -> bool:
    """Check if a title looks generic / not a real topic title."""
    if not title or len(title) < 3:
        return True
    low = title.lower().strip()
    for pat in BAD_PATTERNS:
        if re.match(pat, low):
            return True
    return False


def find_tema_pdfs(input_dir: str) -> dict[int, Path]:
    """Find all Tema-X-*.pdf files."""
    tema_files = {}
    for f in Path(input_dir).glob("Tema-*.pdf"):
        parts = f.stem.split("-")
        try:
            num = int(parts[1])
            tema_files[num] = f
        except (IndexError, ValueError):
            continue
    return dict(sorted(tema_files.items()))


def find_tema_jsons(rewritten_dir: str) -> dict[int, Path]:
    """Find rewritten JSON files (excluding coord and chunks_metadata)."""
    tema_files = {}
    for f in Path(rewritten_dir).glob("Tema-*-Ingreso-*.json"):
        name = f.name
        if "coord" in name or "chunks_metadata" in name:
            continue
        parts = name.split("-")
        try:
            num = int(parts[1])
            tema_files[num] = f
        except (IndexError, ValueError):
            continue
    return dict(sorted(tema_files.items()))


def render_pages_to_base64(pdf_path: Path, num_pages: int = 4, dpi: int = 200) -> list[str]:
    """Render the first N pages of a PDF as base64-encoded PNG images."""
    doc = fitz.open(str(pdf_path))
    images = []
    for i in range(min(num_pages, len(doc))):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        images.append(b64)
    doc.close()
    return images


def get_first_content(json_path: Path, max_chunks: int = 5, max_chars: int = 4000) -> str:
    """Extract text from the first N chunks of a tema JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", [])[:max_chunks]
    text = ""
    for chunk in chunks:
        text += chunk.get("content", "") + "\n"
        if len(text) >= max_chars:
            break
    return text[:max_chars]


# ==========================================================
# Extraction methods
# ==========================================================

def extract_titles_from_web(client: OpenAI, topic_nums: list[int]) -> dict[int, str]:
    """Use web search to find official topic titles."""
    nums_str = ", ".join(str(n) for n in topic_nums)
    prompt = (
        f"Busca el temario oficial de ingreso a la Guardia Civil (Escala de Cabos y Guardias). "
        f"Necesito los titulos oficiales de los siguientes temas: {nums_str}. "
        f"Responde SOLO en formato JSON: {{\"1\": \"Derechos Humanos\", \"2\": \"Igualdad Efectiva\"...}} "
        f"Los titulos deben ser la MATERIA general (ej: 'Derecho Penal'), NO el nombre de la ley. "
        f"Usa titulos cortos de 2-6 palabras. Solo JSON, sin explicaciones."
    )
    try:
        resp = client.responses.create(
            model=MODEL,
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
        text = resp.output_text.strip()
        # Extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not json_match:
            # Try multiline with nested braces
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            raw = json.loads(json_match.group())
            result = {}
            for k, v in raw.items():
                try:
                    result[int(k)] = str(v).strip().strip('"').strip("'")
                except (ValueError, TypeError):
                    continue
            return result
    except Exception as e:
        print(f"  [web] Error: {e}", flush=True)
    return {}


def extract_title_with_vision(client: OpenAI, tema_num: int, images_b64: list[str]) -> str:
    """Use OpenAI Vision to extract the title from cover page images."""
    content_parts = [
        {
            "type": "text",
            "text": (
                f"Estas son las primeras paginas del Tema {tema_num} del temario de "
                f"ingreso a la Guardia Civil. "
                f"Una de las paginas contiene un mapa conceptual o esquema visual "
                f"con el titulo/materia principal del tema en grande. "
                f"Extrae ese titulo o materia principal que aparece destacado en el esquema. "
                f"Recuerda: el titulo es la MATERIA (ej: 'Derecho Penal'), NO el nombre de una ley."
            ),
        }
    ]
    for b64 in images_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


def extract_title_from_chunks(client: OpenAI, tema_num: int, content: str) -> str:
    """Fallback: infer the topic title from text content of the first chunks."""
    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Este es el contenido inicial del Tema {tema_num} del temario de "
                    f"ingreso a la Guardia Civil. Infiere el titulo de la materia que trata "
                    f"(NO el nombre de la ley, sino la materia general):\n\n{content}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Extract topic titles (web + vision + chunks fallback)")
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT_DIR,
        help=f"Directory with PDF files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--rewritten-dir", default=DEFAULT_REWRITTEN_DIR,
        help=f"Directory with rewritten JSON files for fallback (default: {DEFAULT_REWRITTEN_DIR})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSON file for titles (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-extraction even if output file exists",
    )
    parser.add_argument(
        "--pages", type=int, default=4,
        help="Number of pages to render per PDF (default: 4)",
    )
    parser.add_argument(
        "--topics", type=str, default=None,
        help="Comma-separated list of topic numbers to extract (default: all). E.g. --topics 1,3,5",
    )
    args = parser.parse_args()

    # Parse topic filter
    topic_filter = None
    if args.topics:
        topic_filter = set()
        for part in args.topics.split(","):
            try:
                topic_filter.add(int(part.strip()))
            except ValueError:
                pass

    # Load existing cache
    output_path = Path(args.output)
    existing_titles = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing_titles = json.load(f)

    # If no --force and no specific topics, just show cache
    if existing_titles and not args.force and not topic_filter:
        print(f"Cached titles found at {output_path}. Use --force to re-extract.")
        for num, title in sorted(existing_titles.items(), key=lambda x: int(x[0])):
            print(f"  Tema {num}: {title}")
        return existing_titles

    # Find tema PDFs and rewritten JSONs
    tema_pdfs = find_tema_pdfs(args.input_dir)
    tema_jsons = find_tema_jsons(args.rewritten_dir)

    # All available tema numbers
    all_nums = sorted(set(tema_pdfs.keys()) | set(tema_jsons.keys()))
    if not all_nums:
        print(f"No tema files found in {args.input_dir} or {args.rewritten_dir}")
        sys.exit(1)

    # Filter to requested topics
    if topic_filter:
        all_nums = [n for n in all_nums if n in topic_filter]
        if not all_nums:
            print(f"No files found for topics: {topic_filter}")
            sys.exit(1)

    print(f"Extracting titles for {len(all_nums)} temas")
    print(f"Using model: {MODEL} | Pages per PDF: {args.pages}")

    # Initialize OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    new_titles = {}

    # --- Step 0: Web search (single call for all topics) ---
    print(f"\n[1/3] Buscando titulos oficiales en internet...", flush=True)
    web_titles = extract_titles_from_web(client, all_nums)
    for num in all_nums:
        if num in web_titles and not is_bad_title(web_titles[num]):
            new_titles[num] = web_titles[num]
            print(f"  Tema {num}: [web] {web_titles[num]}", flush=True)

    pending = [n for n in all_nums if n not in new_titles]
    if not pending:
        print("  Todos los titulos encontrados via web.", flush=True)
    else:
        print(f"  {len(pending)} temas pendientes: {pending}", flush=True)

    # --- Step 1: Vision on PDF for remaining ---
    if pending:
        print(f"\n[2/3] Extrayendo titulos por vision (PDF)...", flush=True)
    for num in list(pending):
        if num not in tema_pdfs:
            continue
        print(f"  Tema {num}: [vision] rendering {args.pages} pages...", end=" ", flush=True)
        try:
            images_b64 = render_pages_to_base64(tema_pdfs[num], num_pages=args.pages)
            title = extract_title_with_vision(client, num, images_b64)
            print(title, flush=True)
            if not is_bad_title(title):
                new_titles[num] = title
                pending.remove(num)
            else:
                print(f"           Titulo generico: '{title}'", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # --- Step 2: Chunks fallback for remaining ---
    if pending:
        print(f"\n[3/3] Infiriendo titulos desde chunks de texto...", flush=True)
    for num in list(pending):
        if num not in tema_jsons:
            print(f"  Tema {num}: sin JSON reescrito para fallback", flush=True)
            continue
        print(f"  Tema {num}: [chunks] inferiendo...", end=" ", flush=True)
        try:
            content = get_first_content(tema_jsons[num])
            title = extract_title_from_chunks(client, num, content)
            print(title, flush=True)
            if not is_bad_title(title):
                new_titles[num] = title
                pending.remove(num)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Report failures
    if pending:
        print(f"\nNO SE PUDO EXTRAER TITULO para: {pending}", flush=True)

    # Merge with existing cache (new overwrites old)
    merged = {int(k): v for k, v in existing_titles.items()}
    merged.update(new_titles)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(merged.items())}, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*50}")
    for num, title in sorted(merged.items()):
        src = "[web]" if num in web_titles and web_titles.get(num) == title else "[local]"
        print(f"  Tema {num}: {title}  {src}")
    print(f"{'='*50}")
    print(f"Titles saved to {output_path}")
    return merged


if __name__ == "__main__":
    main()
