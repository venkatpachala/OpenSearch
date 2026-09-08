"""Live arXiv Atom client with cache and injectable HTTP transport."""
from __future__ import annotations
import urllib.parse, xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Any
import httpx
from .cache import ResearchCache

@dataclass
class PaperRecord:
    id: str; title: str; authors: list[str]; abstract: str; published: str = ""; pdf_url: str = ""; source: str = "arxiv"
    def to_dict(self): return asdict(self)

class ArxivClient:
    endpoint = "https://export.arxiv.org/api/query"
    def __init__(self, cache: ResearchCache | None = None, client: Any | None = None, timeout: float = 30):
        self.cache, self.client, self.timeout = cache, client or httpx.Client(), timeout
    def search(self, query: str, category: str | None = None, max_results: int = 10) -> list[PaperRecord]:
        full = f"{query} category:{category}" if category else query
        cache_key = f"arxiv:{full}:{max_results}"
        cached = self.cache.get_query(cache_key) if self.cache else None
        if cached is not None: return [PaperRecord(**x) for x in cached]
        params = {"search_query": f"all:{full}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
        response = self.client.get(self.endpoint, params=params, timeout=self.timeout); response.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}; out=[]
        for entry in ET.fromstring(response.text).findall("a:entry", ns):
            pid = (entry.findtext("a:id", "", ns).rsplit("/", 1)[-1])
            links = entry.findall("a:link", ns)
            pdf = next((x.attrib.get("href", "") for x in links if x.attrib.get("title") == "pdf"), "")
            out.append(PaperRecord(pid, " ".join(entry.findtext("a:title", "", ns).split()), [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)], " ".join(entry.findtext("a:summary", "", ns).split()), entry.findtext("a:published", "", ns), pdf))
        if self.cache: self.cache.put_query(cache_key, [x.to_dict() for x in out])
        return out
    def download_pdf(self, paper: PaperRecord | str) -> Any:
        pid, url = (paper.id, paper.pdf_url) if isinstance(paper, PaperRecord) else (paper, f"https://arxiv.org/pdf/{paper}.pdf")
        path = self.cache.paper_path(pid) if self.cache else None
        if path and path.exists(): return path
        response = self.client.get(url, timeout=self.timeout); response.raise_for_status()
        if path is None: raise ValueError("download_pdf requires a cache")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(response.content); return path

