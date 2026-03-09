#!/usr/bin/env python3
"""Script interactivo para elegir temas y generar preguntas.

Lee los JSON de chunks disponibles en input_docs/rewritten/,
muestra un menu para seleccionar uno o varios temas,
y lanza el pipeline BCDE para cada uno.

Uso:
    python scripts/select_topic_and_generate.py
    python scripts/select_topic_and_generate.py --questions-per-chunk 2
    python scripts/select_topic_and_generate.py --skip-agent-c
"""

import re
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REWRITTEN_DIR = PROJECT_ROOT / "input_docs" / "rewritten"


def discover_topics() -> list[dict]:
    """Descubre temas disponibles a partir de los JSON de chunks."""
    if not REWRITTEN_DIR.exists():
        return []

    topic_re = re.compile(r"Tema-(\d+)")
    topics: dict[int, dict] = {}

    for json_file in sorted(REWRITTEN_DIR.glob("*.json")):
        # Saltar archivos auxiliares
        if json_file.name.endswith("_chunks_metadata.json"):
            continue
        if json_file.name.endswith(".coord.json"):
            continue

        match = topic_re.search(json_file.stem)
        if not match:
            continue

        topic_num = int(match.group(1))
        if topic_num in topics:
            continue

        # Contar chunks sin cargar todo el JSON
        import json
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_chunks = len(data.get("chunks", []))
        except Exception:
            total_chunks = 0

        topics[topic_num] = {
            "num": topic_num,
            "file": json_file,
            "chunks": total_chunks,
            "name": json_file.stem,
        }

    return [topics[k] for k in sorted(topics)]


def print_topic_table(topics: list[dict]):
    """Muestra tabla de temas disponibles."""
    print(f"\n  {'#':<5} {'Tema':<8} {'Chunks':<10} {'Archivo'}")
    print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*50}")
    for i, t in enumerate(topics, 1):
        print(f"  {i:<5} Tema {t['num']:<3} {t['chunks']:<10} {t['file'].name}")
    print()


def parse_selection(text: str, max_val: int) -> list[int]:
    """Parsea seleccion tipo '1,3,5-8' y devuelve lista de indices (1-based)."""
    indices = set()
    text = text.strip()

    if text.lower() == "all":
        return list(range(1, max_val + 1))

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                start, end = int(bounds[0]), int(bounds[1])
                for i in range(start, end + 1):
                    if 1 <= i <= max_val:
                        indices.add(i)
            except ValueError:
                pass
        else:
            try:
                val = int(part)
                if 1 <= val <= max_val:
                    indices.add(val)
            except ValueError:
                pass

    return sorted(indices)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seleccionar temas y generar preguntas")
    parser.add_argument("--questions-per-chunk", type=int, default=1,
                        help="Preguntas por chunk (default: 1)")
    parser.add_argument("--academy", type=int, default=1, help="Academy ID (default: 1)")
    parser.add_argument("--skip-agent-c", action="store_true",
                        help="Omitir validacion del Agente C")
    parser.add_argument("--single-pdf", action="store_true",
                        help="Generar un unico PDF combinado")
    parser.add_argument("--force", action="store_true",
                        help="Ignorar cache de chunks usados")
    parser.add_argument("--limit-chunks", type=int, default=0,
                        help="Limitar numero de chunks a procesar")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar que se ejecutaria sin ejecutar")

    args = parser.parse_args()

    print("=" * 70)
    print("  SELECTOR DE TEMAS PARA GENERACION DE PREGUNTAS")
    print("=" * 70)

    topics = discover_topics()

    if not topics:
        print(f"\nNo se encontraron JSON de chunks en: {REWRITTEN_DIR}")
        print("Ejecuta primero el Agente Z (rewriter) para generar los chunks.")
        return 1

    print(f"\nTemas disponibles ({len(topics)}):")
    print_topic_table(topics)

    print("Selecciona temas a procesar:")
    print("  - Numeros separados por coma: 1,3,5")
    print("  - Rangos: 1-5")
    print("  - Combinaciones: 1,3,5-8")
    print("  - Todos: all")
    print()

    try:
        selection = input("Tu seleccion: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelado.")
        return 0

    if not selection:
        print("No se selecciono nada.")
        return 0

    indices = parse_selection(selection, len(topics))

    if not indices:
        print("Seleccion invalida.")
        return 1

    selected = [topics[i - 1] for i in indices]

    print(f"\nTemas seleccionados ({len(selected)}):")
    for t in selected:
        print(f"  Tema {t['num']} ({t['chunks']} chunks)")

    # Construir y ejecutar comandos
    print(f"\n{'=' * 70}")
    print("  CONFIGURACION")
    print(f"{'=' * 70}")
    print(f"  Preguntas/chunk: {args.questions_per_chunk}")
    print(f"  Academy: {args.academy}")
    print(f"  Agente C: {'DESACTIVADO' if args.skip_agent_c else 'ACTIVADO'}")
    print(f"  Force: {'SI' if args.force else 'NO'}")
    if args.limit_chunks:
        print(f"  Limite chunks: {args.limit_chunks}")
    print()

    if not args.dry_run:
        try:
            confirm = input("Iniciar generacion? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelado.")
            return 0
        if confirm != "y":
            print("Cancelado.")
            return 0

    for t in selected:
        topic_num = t["num"]
        doc_filter = f"Tema-{topic_num}"

        cmd = [
            sys.executable, str(PROJECT_ROOT / "run_bcde_pipeline.py"),
            "--topic", str(topic_num),
            "--academy", str(args.academy),
            "--doc", doc_filter,
            "--questions-per-chunk", str(args.questions_per_chunk),
        ]

        if args.skip_agent_c:
            cmd.append("--skip-agent-c")
        if args.single_pdf:
            cmd.append("--single-pdf")
        if args.force:
            cmd.append("--force")
        if args.limit_chunks:
            cmd.extend(["--limit-chunks", str(args.limit_chunks)])

        print(f"\n{'=' * 70}")
        print(f"  PROCESANDO TEMA {topic_num}")
        print(f"{'=' * 70}")
        print(f"  Comando: {' '.join(cmd)}")

        if args.dry_run:
            print("  [DRY RUN] No se ejecuta.")
            continue

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            print(f"\n  Error procesando Tema {topic_num} (exit code: {result.returncode})")
            try:
                cont = input("  Continuar con el siguiente tema? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAbortado.")
                return 1
            if cont != "y":
                return 1

    print(f"\n{'=' * 70}")
    print("  PROCESO COMPLETADO")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())