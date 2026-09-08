"""Small offline-first cache for literature metadata and PDFs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

class ResearchCache:
    def __init__(self, root: Path | None = None, cache_dir: Path | None = None):
        self.root = root or cache_dir or Path("data/cache")
        self.root.mkdir(parents=True, exist_ok=True)
        self.query_file = self.root / "queries.json"
        self._queries = json.loads(self.query_file.read_text(encoding="utf-8")) if self.query_file.exists() else {}
    def _save(self):
        self.query_file.write_text(json.dumps(self._queries, indent=2), encoding="utf-8")
    def key(self, query: str) -> str: return hashlib.sha256(query.strip().lower().encode()).hexdigest()
    def get_query(self, query: str) -> Any: return self._queries.get(self.key(query))
    def put_query(self, query: str, value: Any) -> None:
        self._queries[self.key(query)] = value; self._save()
    def paper_path(self, paper_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in paper_id)
        return self.root / "papers" / f"{safe}.pdf"

