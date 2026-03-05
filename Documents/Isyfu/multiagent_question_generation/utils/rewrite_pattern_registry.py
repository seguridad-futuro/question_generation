"""Registry for learned rewrite patterns (regex-based cleanup)."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RewritePatternRegistry:
    """Tracks regex patterns learned from rewrite cleanups."""

    def __init__(
        self,
        path: Path,
        min_hits: int = 2,
        min_len: int = 6,
        max_len: int = 160,
    ) -> None:
        self.path = path
        self.min_hits = min_hits
        self.min_len = min_len
        self.max_len = max_len
        self.patterns: Dict[str, Dict[str, object]] = {}
        self._compiled: Dict[str, re.Pattern] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            patterns = data.get("patterns", {})
            if isinstance(patterns, dict):
                self.patterns = patterns
        except Exception as exc:
            logger.warning(f"Failed to load rewrite patterns: {exc}")

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "patterns": self.patterns,
            }
            self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def apply(self, text: str) -> Tuple[str, List[str]]:
        """Apply learned patterns to text, returning cleaned text + matches."""
        if not text or not self.patterns:
            return text, []

        cleaned = text
        matched_patterns: List[str] = []

        for regex, entry in self.patterns.items():
            if entry.get("count", 0) < self.min_hits:
                continue
            compiled = self._compiled.get(regex)
            if compiled is None:
                try:
                    compiled = re.compile(regex, flags=re.MULTILINE)
                    self._compiled[regex] = compiled
                except re.error:
                    continue
            cleaned_next, count = compiled.subn("", cleaned)
            if count > 0:
                cleaned = cleaned_next
                matched_patterns.append(regex)

        if matched_patterns:
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r" {2,}", " ", cleaned)
            cleaned = cleaned.strip()

        return cleaned, matched_patterns

    def learn_from_removed_lines(
        self,
        lines: List[str],
        doc_id: Optional[str],
        reason: str = ""
    ) -> int:
        """Learn regex patterns from removed lines."""
        if not lines:
            return 0

        with self._lock:
            now = datetime.now().isoformat()
            updated = False
            learned = 0

            for line in sorted(set(lines)):
                normalized = self._normalize_line(line)
                if not self._should_learn(normalized):
                    continue
                regex = self._line_to_regex(normalized)
                entry = self.patterns.get(regex)
                if entry is None:
                    entry = {
                        "sample": normalized,
                        "count": 0,
                        "last_seen": now,
                        "doc_ids": [],
                        "reason": reason or ""
                    }
                    self.patterns[regex] = entry
                    learned += 1

                entry["count"] = int(entry.get("count", 0)) + 1
                entry["last_seen"] = now
                if doc_id:
                    doc_ids = entry.get("doc_ids", [])
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
                        entry["doc_ids"] = doc_ids
                if reason and not entry.get("reason"):
                    entry["reason"] = reason
                updated = True

            if updated:
                self.save()
            return learned

    def _should_learn(self, line: str) -> bool:
        if not line:
            return False
        length = len(line)
        if length < self.min_len or length > self.max_len:
            return False
        lower = self._strip_accents(line.lower())
        for keyword in ("articulo", "capitulo", "seccion", "titulo"):
            if keyword in lower:
                return False
        return True

    @staticmethod
    def _normalize_line(line: str) -> str:
        return re.sub(r"\s+", " ", line).strip()

    @staticmethod
    def _strip_accents(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @staticmethod
    def _line_to_regex(line: str) -> str:
        collapsed = RewritePatternRegistry._normalize_line(line)
        collapsed = re.sub(r"\d+", "<NUM>", collapsed)
        escaped = re.escape(collapsed)
        escaped = escaped.replace(r"\ ", r"\s+")
        escaped = escaped.replace("<NUM>", r"\\d+")
        return rf"^\s*{escaped}\s*$"
