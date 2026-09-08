"""Typed tools backed by live literature services and local PDF cache."""
from __future__ import annotations
from pathlib import Path
from .base import Tool, ToolRequest, ToolResponse
from ..research import ArxivClient, SemanticScholarClient, ResearchCache, PdfSectionParser, MethodologyExtractor

class LiveSearchRequest(ToolRequest):
    query: str
    max_results: int = 5
    source: str = "arxiv"

class LiveSearchLiteratureTool(Tool):
    name="search_literature"; description="Search live arXiv or Semantic Scholar for papers matching the research goal."; request_model=LiveSearchRequest
    def __init__(self, cache_dir: Path):
        cache=ResearchCache(cache_dir); self.arxiv=ArxivClient(cache); self.semantic=SemanticScholarClient(cache)
    def execute(self, request):
        papers=(self.semantic.search(request.query,request.max_results) if request.source == "semantic_scholar" else self.arxiv.search(request.query,max_results=request.max_results))
        return ToolResponse.ok(papers=[p.to_dict() for p in papers],count=len(papers),source=request.source)

class LiveRetrieveRequest(ToolRequest):
    paper_id: str
    pdf_url: str = ""

class LiveRetrievePaperTool(Tool):
    name="retrieve_paper"; description="Download a real paper PDF and extract its section text into the local cache."; request_model=LiveRetrieveRequest
    def __init__(self, cache_dir: Path): self.cache=ResearchCache(cache_dir); self.client=ArxivClient(self.cache); self.parser=PdfSectionParser()
    def execute(self, request):
        try:
            from ..research.arxiv_client import PaperRecord
            path=self.client.download_pdf(PaperRecord(request.paper_id, "", [], "", pdf_url=request.pdf_url) if request.pdf_url else request.paper_id)
            sections=self.parser.extract(path)
            return ToolResponse.ok(id=request.paper_id,title=request.paper_id,abstract=sections.get("abstract",""),sections=sections,pdf_path=str(path),url=request.pdf_url)
        except Exception as exc: return ToolResponse.fail(f"Paper retrieval failed: {exc}")

class LiveExtractRequest(ToolRequest):
    paper_id: str
    sections: dict[str,str] = {}

class LiveExtractMethodologyTool(Tool):
    name="extract_methodology"; description="Extract reproducible methodology fields from retrieved paper sections."; request_model=LiveExtractRequest
    def execute(self, request): return ToolResponse.ok(**MethodologyExtractor().extract(request.paper_id,request.sections).model_dump())

