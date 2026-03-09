"""
Create topics in Supabase for each tema extracted from the study material.

This script:
1. Fetches topic_types of level 'Study' from Supabase
2. Lets the user choose which topic_type to use
3. Optionally runs extract_topic_titles.py to get titles via OpenAI
4. Presents the topics to be created for confirmation
5. Inserts the topics into the 'topic' table

Usage:
    python scripts/create_topics_supabase.py [--titles-file TITLES_FILE] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve paths relative to the project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

TITLES_FILE_DEFAULT = str(SCRIPT_DIR / "topic_titles.json")


def get_supabase_client():
    """Initialize Supabase client using REST API with httpx."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)
    return {"url": SUPABASE_URL, "key": SUPABASE_KEY}


def supabase_request(client: dict, method: str, table: str, params: dict = None, data=None):
    """Make a request to the Supabase REST API."""
    import httpx

    url = f"{client['url']}/rest/v1/{table}"
    headers = {
        "apikey": client["key"],
        "Authorization": f"Bearer {client['key']}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    if method == "GET":
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
    elif method == "POST":
        resp = httpx.post(url, headers=headers, params=params, json=data, timeout=30)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if resp.status_code >= 400:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    return resp.json()


def fetch_study_topic_types(client: dict) -> list[dict]:
    """Fetch all topic_types with level='Study'."""
    return supabase_request(
        client, "GET", "topic_type",
        params={"level": "eq.Study", "select": "id,topic_type_name,description,academy_id"},
    )


def fetch_existing_topics(client: dict, topic_type_id: int) -> list[dict]:
    """Fetch existing topics for a given topic_type."""
    return supabase_request(
        client, "GET", "topic",
        params={"topic_type_id": f"eq.{topic_type_id}", "select": "id,topic_name,order,topic_short_name"},
    )


def load_or_extract_titles(titles_file: str) -> dict[int, str]:
    """Load titles from cache or run extraction."""
    titles_path = Path(titles_file)

    if titles_path.exists():
        with open(titles_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    print(f"Titles file not found at {titles_file}.")
    print("Running extract_topic_titles.py to generate titles...")
    import subprocess
    extract_script = str(SCRIPT_DIR / "extract_topic_titles.py")
    result = subprocess.run(
        [sys.executable, extract_script],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    if not titles_path.exists():
        print("Error: titles file was not created")
        sys.exit(1)

    with open(titles_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="Create topics in Supabase")
    parser.add_argument(
        "--titles-file", default=TITLES_FILE_DEFAULT,
        help=f"Path to topic titles JSON (default: {TITLES_FILE_DEFAULT})",
    )
    parser.add_argument(
        "--topic-type-id", type=int, default=None,
        help="Directly specify the topic_type ID to use (skip interactive selection)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without actually inserting",
    )
    args = parser.parse_args()

    client = get_supabase_client()

    # 1. Fetch Study topic_types
    print("Fetching topic_types with level='Study'...\n")
    topic_types = fetch_study_topic_types(client)

    if not topic_types:
        print("No topic_types with level 'Study' found in the database.")
        sys.exit(1)

    # 2. Let user choose (or use --topic-type-id)
    if args.topic_type_id:
        matches = [tt for tt in topic_types if tt["id"] == args.topic_type_id]
        if not matches:
            print(f"Error: topic_type_id={args.topic_type_id} not found among Study types.")
            sys.exit(1)
        selected_tt = matches[0]
    else:
        print("Available Study topic_types:")
        print("-" * 60)
        for i, tt in enumerate(topic_types, 1):
            print(f"  [{i}] {tt['topic_type_name']} (id={tt['id']}, academy_id={tt['academy_id']})")
            if tt.get("description"):
                print(f"      {tt['description']}")
        print("-" * 60)

        if len(topic_types) == 1:
            choice = 1
            print(f"\nOnly one topic_type found, selecting automatically: {topic_types[0]['topic_type_name']}")
        else:
            try:
                choice = int(input("\nSelect topic_type number: "))
                if choice < 1 or choice > len(topic_types):
                    print("Invalid selection")
                    sys.exit(1)
            except (ValueError, EOFError):
                print("Invalid input")
                sys.exit(1)

        selected_tt = topic_types[choice - 1]
    tt_id = selected_tt["id"]
    academy_id = selected_tt["academy_id"]
    print(f"\nSelected: {selected_tt['topic_type_name']} (id={tt_id})")

    # 3. Check for existing topics
    existing = fetch_existing_topics(client, tt_id)
    if existing:
        print(f"\nWARNING: {len(existing)} topics already exist for this topic_type:")
        for t in sorted(existing, key=lambda x: x.get("order") or 0):
            print(f"  - [{t.get('order', '?')}] {t['topic_name']} (id={t['id']})")

        if not args.dry_run:
            try:
                confirm = input("\nDo you want to continue and add new topics? (y/N): ").strip().lower()
            except EOFError:
                confirm = "n"
            if confirm != "y":
                print("Aborted.")
                sys.exit(0)

    # 4. Load or extract titles
    titles = load_or_extract_titles(args.titles_file)
    if not titles:
        print("No titles found. Aborting.")
        sys.exit(1)

    # 5. Build topics to create
    topics_to_create = []
    for num in sorted(titles.keys()):
        topic_data = {
            "topic_type_id": tt_id,
            "topic_name": titles[num],
            "topic_short_name": f"Tema {num}",
            "order": num,
            "academy_id": academy_id,
            "options": selected_tt.get("default_number_options", 4),
            "is_premium": False,
        }
        topics_to_create.append(topic_data)

    # 6. Present for confirmation
    print(f"\n{'='*70}")
    print(f"  TOPICS TO CREATE ({len(topics_to_create)} topics)")
    print(f"  Topic Type: {selected_tt['topic_type_name']} (id={tt_id})")
    print(f"  Academy ID: {academy_id}")
    print(f"{'='*70}")
    print(f"  {'Order':<7} {'Short Name':<12} {'Title'}")
    print(f"  {'-'*7} {'-'*12} {'-'*45}")
    for t in topics_to_create:
        print(f"  {t['order']:<7} {t['topic_short_name']:<12} {t['topic_name']}")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        return

    try:
        confirm = input("\nConfirm creation? (y/N): ").strip().lower()
    except EOFError:
        confirm = "n"
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # 7. Insert topics
    print("\nInserting topics...")
    created = supabase_request(client, "POST", "topic", data=topics_to_create)

    print(f"\nSuccessfully created {len(created)} topics:")
    for t in created:
        print(f"  id={t['id']}: [{t.get('order', '?')}] {t['topic_name']}")

    print("\nDone!")


if __name__ == "__main__":
    main()