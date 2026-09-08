"""Semantic Scholar Graph API client, using the same cache contract."""
from __future__ import annotations
from typing import Any
import httpx
from .arxiv_client import PaperRecord
from .cache import ResearchCache

class SemanticScholarClient:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    def __init__(self, cache: ResearchCache | None = None, client: Any | None = None, timeout: float = 30): self.cache, self.client, self.timeout = cache, client or httpx.Client(), timeout
    def search(self, query: str, limit: int = 10) -> list[PaperRecord]:
        key=f"semantic:{query}:{limit}"; cached=self.cache.get_query(key) if self.cache else None
        if cached is not None: return [PaperRecord(**x) for x in cached]
        r=self.client.get(self.endpoint, params={"query":query,"limit":limit,"fields":"paperId,title,authors,abstract,year,openAccessPdf"}, timeout=self.timeout); r.raise_for_status()
        out=[]
        for x in r.json().get("data",[]):
            pdf=(x.get("openAccessPdf") or {}).get("url","")
            out.append(PaperRecord(x.get("paperId",""),x.get("title","") or "",[a.get("name","") for a in x.get("authors",[])],x.get("abstract","") or "",str(x.get("year") or ""),pdf,"semantic_scholar"))
        if self.cache: self.cache.put_query(key,[x.to_dict() for x in out])
        return out

