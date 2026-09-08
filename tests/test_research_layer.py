"""
Tests for Phase 7: Research Layer (ArxivClient, SemanticScholarClient, PdfSectionParser, MethodologyExtractor, ResearchCache).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from research_repro.research.cache import ResearchCache
from research_repro.research.arxiv_client import ArxivClient, PaperRecord
from research_repro.research.pdf_parser import PdfSectionParser
from research_repro.research.extractor import MethodologyExtractor
from research_repro.tools.research import (
    LiveSearchLiteratureTool, LiveRetrievePaperTool, LiveExtractMethodologyTool
)


class TestResearchCache:
    def test_cache_put_and_get(self, tmp_path):
        cache = ResearchCache(cache_dir=tmp_path)
        data = [{"id": "p1", "title": "Test Paper"}]
        cache.put_query("test_query", data)
        retrieved = cache.get_query("test_query")
        assert retrieved == data
        assert cache.get_query("nonexistent") is None

    def test_paper_path(self, tmp_path):
        cache = ResearchCache(cache_dir=tmp_path)
        path = cache.paper_path("2304.09871")
        assert path.name == "2304.09871.pdf"
        assert path.parent.name == "papers"


class TestArxivClient:
    def test_search_with_cache_hit(self, tmp_path):
        cache = ResearchCache(cache_dir=tmp_path)
        cache.put_query("arxiv:rag:5", [{"id": "2304.09871", "title": "RAG Reranking", "authors": ["Author A"], "abstract": "Test abstract"}])

        client = ArxivClient(cache=cache)
        results = client.search(query="rag", max_results=5)
        assert len(results) == 1
        assert results[0].id == "2304.09871"
        assert results[0].title == "RAG Reranking"


class TestPdfSectionParser:
    def test_extract_sections_from_pdf(self, tmp_path):
        import fitz
        pdf_path = tmp_path / "sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Abstract\nWe present a two-stage retrieval pipeline.")
        page.insert_text((50, 150), "Methodology\nWe use a cross-encoder reranker with BGE-Reranker-Large.")
        page.insert_text((50, 250), "Results\nOur approach achieves accuracy of 0.942 on the test set.")
        doc.save(pdf_path)
        doc.close()

        parser = PdfSectionParser()
        sections = parser.extract(pdf_path)
        assert "abstract" in sections
        assert "methodology" in sections
        assert "cross-encoder" in sections["methodology"]
        assert "results" in sections


class TestMethodologyExtractor:
    def test_extract_regex_fallback(self):
        extractor = MethodologyExtractor()
        sections = {
            "methodology": "We train with batch size of 64 and learning rate: 0.001 for 20 epochs.",
            "results": "The model achieved accuracy of 0.945 and f1 of 0.932.",
        }
        methodology = extractor.extract("paper_001", sections)
        assert methodology.paper_id == "paper_001"
        assert methodology.hyperparameters.get("batch_size") == 64
        assert methodology.hyperparameters.get("learning_rate") == pytest.approx(0.001)
        assert methodology.reported_metrics.get("accuracy") == pytest.approx(0.945)
        assert methodology.reported_metrics.get("f1") == pytest.approx(0.932)


class TestLiveResearchTools:
    def test_live_extract_methodology_tool(self):
        tool = LiveExtractMethodologyTool()
        req = tool.request_model(
            paper_id="p123",
            sections={"methodology": "batch size 32, lr: 0.05", "results": "accuracy of 0.965"}
        )
        resp = tool.execute(req)
        assert resp.success is True
        assert resp.data["paper_id"] == "p123"
        assert resp.data["reported_metrics"]["accuracy"] == pytest.approx(0.965)
