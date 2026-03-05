#!/usr/bin/env python3
"""Run Agent Z on pending documents (no cache yet)."""

import argparse
import re
from pathlib import Path
from typing import List, Optional

from agents.agent_z_rewriter import RewriterAgent, get_cache_path
from config.settings import get_settings


DEFAULT_EXTENSIONS = [".pdf", ".txt", ".md"]


def _normalize_extensions(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_EXTENSIONS
    exts = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("."):
            part = f".{part}"
        exts.append(part.lower())
    return sorted(set(exts))


def _collect_files(input_dir: Path, extensions: List[str]) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(sorted(input_dir.glob(f"*{ext}")))
    return sorted({f for f in files if f.is_file()})


def _is_processed(file_path: Path) -> bool:
    return get_cache_path(file_path).exists()


def _infer_topic_from_filename(file_name: str) -> int:
    match = re.search(r'(?i)tema[\s_-]*([0-9]{1,3})', file_name)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Process pending documents with Agent Z"
    )
    parser.add_argument(
        "--input-dir",
        default=str(settings.input_docs_dir),
        help="Directory with input documents"
    )
    parser.add_argument(
        "--output-subdir",
        default="rewritten",
        help="Subdirectory for rewritten outputs"
    )
    parser.add_argument(
        "--topic",
        type=int,
        default=1,
        help="Topic number for metadata"
    )
    parser.add_argument(
        "--auto-topic",
        action="store_true",
        help="Infer topic from filename when possible (e.g. Tema-14...)"
    )
    parser.add_argument(
        "--extensions",
        default="pdf,txt,md",
        help="Comma-separated extensions to scan (default: pdf,txt,md)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process even if cache exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pending files and exit"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to process"
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=0,
        help="Limit number of chunks per document (Agent Z only)"
    )
    parser.add_argument(
        "--doc-timeout",
        type=int,
        default=settings.rewriter_doc_timeout_seconds,
        help="Hard timeout (seconds) per document. 0 disables."
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    if input_dir.resolve() != settings.input_docs_dir.resolve():
        print(
            "Note: cache directory is tied to settings.input_docs_dir "
            f"({settings.input_docs_dir})"
        )

    extensions = _normalize_extensions(args.extensions)
    files = _collect_files(input_dir, extensions)

    if args.force:
        pending = files
    else:
        pending = [f for f in files if not _is_processed(f)]

    if args.limit and args.limit > 0:
        pending = pending[: args.limit]

    if args.list:
        print("Pending files:")
        for f in pending:
            print(f" - {f.name}")
        print(f"Total pending: {len(pending)}")
        return 0

    if not pending:
        print("No pending files to process.")
        return 0

    output_dir = input_dir / args.output_subdir

    processed = []
    errors = []

    def _process_file(file_path: Path) -> tuple[Path, int, Optional[str]]:
        topic = args.topic
        if args.auto_topic:
            inferred = _infer_topic_from_filename(file_path.name)
            if inferred > 0:
                topic = inferred
        agent = RewriterAgent()
        agent.rewrite_document(file_path, output_dir, topic, limit_chunks=args.limit_chunks)
        return file_path, topic, None


    if args.doc_timeout and args.doc_timeout > 0:
        print("⚠️ doc-timeout ignorado: ejecución secuencial forzada (sin procesos en paralelo).")

    agent = RewriterAgent()
    for idx, file_path in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] Processing: {file_path.name}")
        try:
            topic = args.topic
            if args.auto_topic:
                inferred = _infer_topic_from_filename(file_path.name)
                if inferred > 0:
                    topic = inferred
            agent.rewrite_document(file_path, output_dir, topic, limit_chunks=args.limit_chunks)
            processed.append(file_path)
        except Exception as exc:
            errors.append((file_path, str(exc)))
            print(f"Error processing {file_path.name}: {exc}")

    print("\nSummary:")
    print(f" - processed: {len(processed)}")
    print(f" - failed: {len(errors)}")
    if errors:
        for file_path, err in errors:
            print(f"   * {file_path.name}: {err}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
