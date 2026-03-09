"""
Upload locally generated questions to the corresponding Supabase topics.

This script:
1. Scans output/json/ for all generated question files
2. Groups questions by tema number (extracted from source_doc_ids)
3. Fetches existing topics from Supabase for the selected topic_type
4. Maps tema numbers to topic IDs using the 'order' field
5. Presents a summary and asks for confirmation
6. Inserts questions + question_options into Supabase

Usage:
    python scripts/upload_questions_supabase.py [--output-dir OUTPUT_DIR] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "json")


def supabase_request(method: str, table: str, params: dict = None, data=None):
    """Make a request to the Supabase REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    if method == "GET":
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
    elif method == "POST":
        resp = httpx.post(url, headers=headers, params=params, json=data, timeout=60)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if resp.status_code >= 400:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    return resp.json()


def fetch_study_topic_types() -> list[dict]:
    """Fetch all topic_types with level='Study'."""
    return supabase_request(
        "GET", "topic_type",
        params={"level": "eq.Study", "select": "id,topic_type_name,description,academy_id"},
    )


def fetch_topics_for_type(topic_type_id: int) -> list[dict]:
    """Fetch all topics for a given topic_type, ordered by 'order'."""
    return supabase_request(
        "GET", "topic",
        params={
            "topic_type_id": f"eq.{topic_type_id}",
            "select": "id,topic_name,topic_short_name,order,academy_id,total_questions",
            "order": "order.asc",
        },
    )


def extract_tema_number(source_doc_ids: list[str]) -> int | None:
    """Extract tema number from source_doc_ids like 'Tema-10-Ingreso-...'."""
    for doc_id in source_doc_ids:
        m = re.search(r"Tema-(\d+)-", doc_id)
        if m:
            return int(m.group(1))
    return None


def scan_questions(output_dir: str) -> dict[int, list[dict]]:
    """Scan all JSON files and group questions by tema number."""
    questions_by_tema: dict[int, list[dict]] = {}

    for root, _, files in os.walk(output_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(root, fname)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            for q in data.get("questions", []):
                tema_num = extract_tema_number(q.get("source_doc_ids", []))
                if tema_num is None:
                    continue
                questions_by_tema.setdefault(tema_num, []).append(q)

    return dict(sorted(questions_by_tema.items()))


def select_topic_type() -> dict:
    """Interactive selection of topic_type."""
    topic_types = fetch_study_topic_types()
    if not topic_types:
        print("No topic_types with level 'Study' found.")
        sys.exit(1)

    print("Available Study topic_types:")
    print("-" * 60)
    for i, tt in enumerate(topic_types, 1):
        print(f"  [{i}] {tt['topic_type_name']} (id={tt['id']}, academy_id={tt['academy_id']})")
        if tt.get("description"):
            print(f"      {tt['description']}")
    print("-" * 60)

    if len(topic_types) == 1:
        print(f"\nAuto-selected: {topic_types[0]['topic_type_name']}")
        return topic_types[0]

    try:
        choice = int(input("\nSelect topic_type number: "))
        if choice < 1 or choice > len(topic_types):
            print("Invalid selection")
            sys.exit(1)
    except (ValueError, EOFError):
        print("Invalid input")
        sys.exit(1)

    return topic_types[choice - 1]


def select_temas(available: list[int]) -> list[int]:
    """Let user select which temas to upload."""
    print(f"\nAvailable temas: {', '.join(str(t) for t in available)}")
    print("Options:")
    print("  - Press Enter to upload ALL temas")
    print("  - Type specific numbers separated by commas (e.g., 1,2,3)")
    print("  - Type a range (e.g., 1-5)")

    try:
        selection = input("\nSelect temas to upload: ").strip()
    except EOFError:
        selection = ""

    if not selection:
        return available

    selected = set()
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            for n in range(int(start), int(end) + 1):
                if n in available:
                    selected.add(n)
        else:
            n = int(part)
            if n in available:
                selected.add(n)

    return sorted(selected)


def upload_questions(
    questions: list[dict],
    topic_id: int,
    academy_id: int,
) -> int:
    """Upload questions and their options to Supabase. Returns count of inserted."""
    inserted = 0

    for q in questions:
        # Insert question
        question_data = {
            "question": q["question"],
            "tip": q.get("tip", ""),
            "topic": topic_id,
            "article": q.get("article", ""),
            "academy_id": academy_id,
            "published": True,
            "order": 0,
        }

        result = supabase_request("POST", "questions", data=question_data)
        if not result:
            continue

        question_id = result[0]["id"]

        # Insert question options
        correct_num = q.get("correct", 1)
        options_data = []
        for i in range(1, 5):
            answer_key = f"answer{i}"
            answer_text = q.get(answer_key, "")
            if not answer_text:
                continue
            options_data.append({
                "question_id": question_id,
                "answer": answer_text,
                "is_correct": i == correct_num,
                "option_order": i,
            })

        if options_data:
            supabase_request("POST", "question_options", data=options_data)

        inserted += 1

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Upload questions to Supabase")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory with generated question JSONs (default: output/json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without actually inserting",
    )
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)

    # 1. Scan local questions
    print("Scanning local question files...\n")
    questions_by_tema = scan_questions(args.output_dir)

    if not questions_by_tema:
        print(f"No questions found in {args.output_dir}")
        sys.exit(1)

    total_local = sum(len(qs) for qs in questions_by_tema.values())
    print(f"Found {total_local} questions across {len(questions_by_tema)} temas\n")

    # 2. Select topic_type
    selected_tt = select_topic_type()
    tt_id = selected_tt["id"]
    academy_id = selected_tt["academy_id"]
    print(f"\nSelected: {selected_tt['topic_type_name']} (id={tt_id})\n")

    # 3. Fetch topics from Supabase
    topics = fetch_topics_for_type(tt_id)
    if not topics:
        print("No topics found for this topic_type. Create topics first with create_topics_supabase.py")
        sys.exit(1)

    # Build mapping: order (tema number) -> topic
    topic_by_order = {t["order"]: t for t in topics if t.get("order")}

    # 4. Check which temas have matching topics
    matched = []
    unmatched = []
    for tema_num in questions_by_tema:
        if tema_num in topic_by_order:
            matched.append(tema_num)
        else:
            unmatched.append(tema_num)

    if unmatched:
        print(f"WARNING: No matching topic found for temas: {', '.join(str(t) for t in unmatched)}")
        print("These questions will be skipped.\n")

    if not matched:
        print("No temas match any existing topics. Aborting.")
        sys.exit(1)

    # 5. Let user select which temas to upload
    selected_temas = select_temas(matched)

    if not selected_temas:
        print("No temas selected. Aborting.")
        sys.exit(1)

    # 6. Present summary
    print(f"\n{'='*75}")
    print(f"  UPLOAD SUMMARY")
    print(f"  Topic Type: {selected_tt['topic_type_name']} (id={tt_id})")
    print(f"{'='*75}")
    print(f"  {'Tema':<7} {'Questions':<12} {'Target Topic':<40} {'ID'}")
    print(f"  {'-'*7} {'-'*12} {'-'*40} {'-'*5}")

    total_to_upload = 0
    for tema_num in selected_temas:
        topic = topic_by_order[tema_num]
        n_questions = len(questions_by_tema[tema_num])
        total_to_upload += n_questions
        print(f"  {tema_num:<7} {n_questions:<12} {topic['topic_name'][:40]:<40} {topic['id']}")

    print(f"  {'-'*7} {'-'*12} {'-'*40} {'-'*5}")
    print(f"  {'TOTAL':<7} {total_to_upload:<12}")
    print(f"{'='*75}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        return

    try:
        confirm = input("\nConfirm upload? (y/N): ").strip().lower()
    except EOFError:
        confirm = "n"
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # 7. Upload
    print("\nUploading questions...")
    total_inserted = 0
    for tema_num in selected_temas:
        topic = topic_by_order[tema_num]
        questions = questions_by_tema[tema_num]
        print(f"  Tema {tema_num} -> topic '{topic['topic_short_name']}' (id={topic['id']})...", end=" ")
        count = upload_questions(questions, topic["id"], academy_id)
        total_inserted += count
        print(f"{count} questions inserted")

    print(f"\nDone! {total_inserted} questions uploaded to Supabase.")


if __name__ == "__main__":
    main()