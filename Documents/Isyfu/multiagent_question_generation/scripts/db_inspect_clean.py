"""
Inspect and clean the local SQLite database (database/questions.db).

Shows current state of questions, options, and batches,
and provides options to selectively or fully clean them.

Usage:
    python scripts/db_inspect_clean.py                # inspect only
    python scripts/db_inspect_clean.py --clean        # interactive clean
    python scripts/db_inspect_clean.py --clean-all    # wipe everything (with confirmation)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "database" / "questions.db"


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def query(conn: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def scalar(conn: sqlite3.Connection, sql: str, params=()) -> int:
    return conn.execute(sql, params).fetchone()[0]


def inspect(conn: sqlite3.Connection):
    """Show database content summary."""
    n_questions = scalar(conn, "SELECT COUNT(*) FROM questions")
    n_options = scalar(conn, "SELECT COUNT(*) FROM question_options")
    n_batches = scalar(conn, "SELECT COUNT(*) FROM batches")
    n_duplicates = scalar(conn, "SELECT COUNT(*) FROM questions WHERE is_duplicate = 1")
    n_manual = scalar(conn, "SELECT COUNT(*) FROM questions WHERE needs_manual_review = 1")

    print("=" * 70)
    print("  LOCAL DATABASE INSPECTION")
    print(f"  Path: {DB_PATH}")
    print("=" * 70)

    # General totals
    print(f"\n  TOTALS")
    print(f"  {'-'*40}")
    print(f"  Questions:        {n_questions}")
    print(f"  Question Options: {n_options}")
    print(f"  Batches:          {n_batches}")
    print(f"  Duplicates:       {n_duplicates}")
    print(f"  Manual Review:    {n_manual}")

    # Questions by topic
    if n_questions > 0:
        by_topic = query(conn, """
            SELECT topic, COUNT(*) as count,
                   ROUND(AVG(faithfulness_score), 2) as avg_faith,
                   ROUND(AVG(relevancy_score), 2) as avg_rel
            FROM questions
            GROUP BY topic
            ORDER BY topic
        """)
        print(f"\n  QUESTIONS BY TOPIC")
        print(f"  {'-'*60}")
        print(f"  {'Topic':<8} {'Count':<8} {'Avg Faith':<12} {'Avg Rel':<12}")
        print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
        for r in by_topic:
            print(f"  {r['topic']:<8} {r['count']:<8} {r['avg_faith'] or '-':<12} {r['avg_rel'] or '-':<12}")

    # Questions by generation method
    if n_questions > 0:
        by_method = query(conn, """
            SELECT COALESCE(generation_method, 'unknown') as method, COUNT(*) as count
            FROM questions
            GROUP BY generation_method
            ORDER BY count DESC
        """)
        print(f"\n  QUESTIONS BY GENERATION METHOD")
        print(f"  {'-'*40}")
        for r in by_method:
            print(f"  {r['method']:<25} {r['count']}")

    # Batches
    if n_batches > 0:
        batches = query(conn, """
            SELECT batch_name, topic, total_questions, unique_questions,
                   duplicates, status, created_at
            FROM batches
            ORDER BY created_at DESC
            LIMIT 20
        """)
        print(f"\n  RECENT BATCHES (last 20)")
        print(f"  {'-'*70}")
        print(f"  {'Batch':<35} {'Topic':<7} {'Total':<7} {'Unique':<8} {'Dup':<5} {'Status'}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*5} {'-'*10}")
        for b in batches:
            name = b['batch_name'][:35]
            print(f"  {name:<35} {b['topic']:<7} {b['total_questions']:<7} "
                  f"{b['unique_questions']:<8} {b['duplicates']:<5} {b['status']}")

    # Sample questions (last 5)
    if n_questions > 0:
        samples = query(conn, """
            SELECT q.id, q.question, q.topic, q.faithfulness_score, q.relevancy_score,
                   q.needs_manual_review, q.is_duplicate, q.created_at
            FROM questions q
            ORDER BY q.created_at DESC
            LIMIT 5
        """)
        print(f"\n  LATEST 5 QUESTIONS")
        print(f"  {'-'*70}")
        for s in samples:
            q_text = s['question'][:65]
            dup = " [DUP]" if s['is_duplicate'] else ""
            rev = " [REVIEW]" if s['needs_manual_review'] else ""
            print(f"  id={s['id']} topic={s['topic']} F={s['faithfulness_score']} R={s['relevancy_score']}{dup}{rev}")
            print(f"    {q_text}")

    print(f"\n{'='*70}")


def clean_interactive(conn: sqlite3.Connection):
    """Interactive cleaning menu."""
    print("\n  CLEAN OPTIONS")
    print("  " + "-" * 45)
    print("  [1] Delete questions for a specific topic")
    print("  [2] Delete a specific batch (+ its questions)")
    print("  [3] Delete duplicate questions")
    print("  [4] Delete questions needing manual review")
    print("  [5] Delete ALL questions and batches")
    print("  [6] Delete EVERYTHING and vacuum")
    print("  [0] Cancel")

    try:
        choice = int(input("\n  Select option: "))
    except (ValueError, EOFError):
        print("  Cancelled.")
        return

    if choice == 0:
        return
    elif choice == 1:
        _clean_by_topic(conn)
    elif choice == 2:
        _clean_by_batch(conn)
    elif choice == 3:
        _clean_duplicates(conn)
    elif choice == 4:
        _clean_manual_review(conn)
    elif choice == 5:
        _clean_all_questions(conn)
    elif choice == 6:
        _clean_everything(conn)
    else:
        print("  Invalid option.")


def _clean_by_topic(conn: sqlite3.Connection):
    topics = query(conn, "SELECT topic, COUNT(*) as cnt FROM questions GROUP BY topic ORDER BY topic")
    if not topics:
        print("  No questions found.")
        return

    print("\n  Topics with questions:")
    for t in topics:
        print(f"    Topic {t['topic']}: {t['cnt']} questions")

    try:
        topic_id = int(input("\n  Enter topic number to delete: "))
    except (ValueError, EOFError):
        print("  Cancelled.")
        return

    count = scalar(conn, "SELECT COUNT(*) FROM questions WHERE topic = ?", (topic_id,))
    if count == 0:
        print(f"  No questions for topic {topic_id}.")
        return

    print(f"\n  Will delete {count} questions for topic {topic_id}")
    if not _confirm():
        return

    # Options cascade via FK, but SQLite needs explicit delete or PRAGMA foreign_keys
    conn.execute("DELETE FROM question_options WHERE question_id IN (SELECT id FROM questions WHERE topic = ?)", (topic_id,))
    conn.execute("DELETE FROM questions WHERE topic = ?", (topic_id,))
    conn.commit()
    print(f"  Deleted {count} questions for topic {topic_id}.")


def _clean_by_batch(conn: sqlite3.Connection):
    batches = query(conn, "SELECT batch_name, topic, total_questions FROM batches ORDER BY created_at DESC LIMIT 20")
    if not batches:
        print("  No batches found.")
        return

    print("\n  Batches:")
    for i, b in enumerate(batches, 1):
        print(f"    [{i}] {b['batch_name']} (topic={b['topic']}, {b['total_questions']} questions)")

    try:
        choice = int(input("\n  Select batch number: "))
        if choice < 1 or choice > len(batches):
            print("  Invalid.")
            return
    except (ValueError, EOFError):
        print("  Cancelled.")
        return

    batch = batches[choice - 1]
    print(f"\n  Will delete batch '{batch['batch_name']}' and associated questions")
    if not _confirm():
        return

    # Delete questions whose source_document contains the batch name
    conn.execute(
        "DELETE FROM question_options WHERE question_id IN "
        "(SELECT id FROM questions WHERE source_document LIKE ?)",
        (f"%{batch['batch_name']}%",),
    )
    conn.execute("DELETE FROM questions WHERE source_document LIKE ?", (f"%{batch['batch_name']}%",))
    conn.execute("DELETE FROM batches WHERE batch_name = ?", (batch['batch_name'],))
    conn.commit()
    print(f"  Batch deleted.")


def _clean_duplicates(conn: sqlite3.Connection):
    count = scalar(conn, "SELECT COUNT(*) FROM questions WHERE is_duplicate = 1")
    if count == 0:
        print("  No duplicate questions found.")
        return

    print(f"\n  Will delete {count} duplicate questions")
    if not _confirm():
        return

    conn.execute("DELETE FROM question_options WHERE question_id IN (SELECT id FROM questions WHERE is_duplicate = 1)")
    conn.execute("DELETE FROM questions WHERE is_duplicate = 1")
    conn.commit()
    print(f"  Deleted {count} duplicates.")


def _clean_manual_review(conn: sqlite3.Connection):
    count = scalar(conn, "SELECT COUNT(*) FROM questions WHERE needs_manual_review = 1")
    if count == 0:
        print("  No questions needing manual review found.")
        return

    print(f"\n  Will delete {count} questions needing manual review")
    if not _confirm():
        return

    conn.execute("DELETE FROM question_options WHERE question_id IN (SELECT id FROM questions WHERE needs_manual_review = 1)")
    conn.execute("DELETE FROM questions WHERE needs_manual_review = 1")
    conn.commit()
    print(f"  Deleted {count} questions.")


def _clean_all_questions(conn: sqlite3.Connection):
    n_q = scalar(conn, "SELECT COUNT(*) FROM questions")
    n_b = scalar(conn, "SELECT COUNT(*) FROM batches")
    print(f"\n  Will delete ALL:")
    print(f"    - {n_q} questions (+ options)")
    print(f"    - {n_b} batches")

    if not _confirm():
        return

    conn.execute("DELETE FROM question_options")
    conn.execute("DELETE FROM questions")
    conn.execute("DELETE FROM batches")
    conn.commit()
    print("  All questions and batches deleted.")


def _clean_everything(conn: sqlite3.Connection):
    print(f"\n  WARNING: This will delete ALL data and vacuum the database.")
    if not _confirm("  Type 'DELETE ALL' to confirm: ", match="DELETE ALL"):
        return

    conn.execute("DELETE FROM question_options")
    conn.execute("DELETE FROM questions")
    conn.execute("DELETE FROM batches")
    conn.commit()
    conn.execute("VACUUM")
    print("  Database completely cleaned and compacted.")


def _confirm(prompt: str = "  Confirm? (y/N): ", match: str = "y") -> bool:
    try:
        return input(prompt).strip().lower() == match.lower()
    except EOFError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect and clean local SQLite database")
    parser.add_argument("--clean", action="store_true", help="Enter interactive cleaning mode")
    parser.add_argument("--clean-all", action="store_true", help="Delete ALL data (with confirmation)")
    args = parser.parse_args()

    conn = get_conn()
    try:
        inspect(conn)

        if args.clean_all:
            _clean_everything(conn)
        elif args.clean:
            clean_interactive(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()