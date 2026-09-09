"""
Stub tool implementations for testing the agent loop without real ML.

These return deterministic, plausible results for both:
  1. Image Classification (CIFAR-10 / MNIST)
  2. RAG & Information Retrieval (Dense Retrieval vs Cross-Encoder Reranking)

Stub behaviour is designed to trigger the full self-correction flow:
  - search_literature / retrieve_paper / extract_methodology → populate memory
  - run_experiment → baseline result is deliberately below target
  - compare_results → flags DISCREPANCY, triggering diagnosis and recovery
  - inspect_methodology → reveals missing technique (normalization or cross-encoder reranking)
  - re-run → target achieved!
"""
from __future__ import annotations

from typing import Any
from .base import Tool, ToolRequest, ToolResponse, ToolRegistry


# ---------------------------------------------------------------------------
# search_literature
# ---------------------------------------------------------------------------


class SearchLiteratureRequest(ToolRequest):
    query: str
    max_results: int = 5


class SearchLiteratureTool(Tool):
    name = "search_literature"
    description = (
        "Search for relevant scientific papers given a research query. "
        "Returns a list of papers with titles, authors, and abstracts."
    )
    request_model = SearchLiteratureRequest

    def execute(self, request: SearchLiteratureRequest) -> ToolResponse:
        q = request.query.lower()
        is_rag = any(kw in q for kw in ("rag", "rerank", "retrieval", "passage", "ndcg", "mrr", "embedding"))

        if is_rag:
            papers = [
                {
                    "id": "paper_rag_01",
                    "title": "Improving RAG Retrieval Quality with Cross-Encoder Reranking",
                    "authors": ["Nogueira et al.", "Lin, J."],
                    "abstract": (
                        "We investigate the effect of two-stage retrieval for Retrieval-Augmented Generation (RAG). "
                        "While dense bi-encoders retrieve candidate passages with NDCG@10 of 0.650, adding a "
                        "cross-encoder reranker improves NDCG@10 to 0.735 (+13.1% relative gain) on MS-MARCO."
                    ),
                    "url": "https://arxiv.org/abs/2304.09871",
                },
                {
                    "id": "paper_rag_02",
                    "title": "Dense Passage Retrieval vs Two-Stage Hybrid Reranking",
                    "authors": ["Karpukhin et al."],
                    "abstract": (
                        "Dense embeddings offer fast candidate retrieval, but shallow interaction limits ranking accuracy. "
                        "Cross-encoder reranking over the top-50 passages consistently yields a >10% retrieval quality lift."
                    ),
                    "url": "https://arxiv.org/abs/2004.04906",
                },
            ]
        else:
            papers = [
                {
                    "id": f"paper_{i}",
                    "title": f"ResNet-based {request.query} — Variant {i}",
                    "authors": ["He et al."],
                    "abstract": (
                        f"We present a residual network approach achieving "
                        f"94.{i}% accuracy on CIFAR-10 using cosine LR scheduling."
                    ),
                    "url": f"https://arxiv.org/abs/200{i}.1234{i}",
                }
                for i in range(1, min(request.max_results, 3) + 1)
            ]

        return ToolResponse.ok(papers=papers, count=len(papers))


# ---------------------------------------------------------------------------
# retrieve_paper
# ---------------------------------------------------------------------------


class RetrievePaperRequest(ToolRequest):
    paper_id: str


class RetrievePaperTool(Tool):
    name = "retrieve_paper"
    description = (
        "Retrieve full details for a paper given its ID, "
        "including methodology sections and reported results."
    )
    request_model = RetrievePaperRequest

    def execute(self, request: RetrievePaperRequest) -> ToolResponse:
        pid = request.paper_id.lower()
        if "rag" in pid or "rerank" in pid:
            return ToolResponse.ok(
                id=request.paper_id,
                title="Improving RAG Retrieval Quality with Cross-Encoder Reranking",
                authors=["Nogueira, R.", "Lin, J."],
                abstract=(
                    "We investigate whether cross-encoder reranking improves RAG retrieval quality. "
                    "Dense retrieval baseline achieves NDCG@10 of 0.650. Applying a cross-encoder "
                    "reranker improves NDCG@10 to 0.735, a relative gain of 13.1%."
                ),
                sections={
                    "methodology": (
                        "Two-stage retrieval pipeline: Stage 1 uses dense bi-encoder (BGE-small-en) "
                        "to retrieve top-50 candidates. Stage 2 applies BGE-Reranker-Large cross-encoder "
                        "to score all query-passage pairs and rerank top-10."
                    ),
                    "experimental_setup": (
                        "Dataset: MS-MARCO QA test subset (1,000 queries). "
                        "Baseline: Bi-encoder only (top-k=10). "
                        "Reranked: Bi-encoder top-50 -> Cross-encoder reranker top-10. "
                        "Metrics: NDCG@10, MRR@10, Relative Improvement."
                    ),
                    "preprocessing": "Passage chunking (512 tokens with 50-token overlap).",
                    "results": (
                        "Dense baseline NDCG@10: 0.650. Reranked NDCG@10: 0.735. "
                        "Relative quality improvement: +13.1% (p < 0.001)."
                    ),
                },
            )

        return ToolResponse.ok(
            id=request.paper_id,
            title="Deep Residual Learning for CIFAR-10 Classification",
            authors=["He, K.", "Zhang, X.", "Ren, S.", "Sun, J."],
            abstract=(
                "We achieve 94.2% top-1 accuracy on CIFAR-10 using residual networks "
                "with a cosine learning-rate schedule and standard data augmentation."
            ),
            sections={
                "methodology": (
                    "ResNet-20 with batch normalization. "
                    "SGD optimizer, lr=0.1, momentum=0.9, weight_decay=1e-4."
                ),
                "experimental_setup": (
                    "Dataset: CIFAR-10. Optimizer: SGD. LR: 0.1 (cosine schedule). "
                    "Epochs: 200. Batch size: 128."
                ),
                "preprocessing": (
                    "RandomCrop(32, padding=4), RandomHorizontalFlip(), "
                    "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])."
                ),
                "results": "Top-1 accuracy: 94.2% on CIFAR-10 test set.",
            },
        )


# ---------------------------------------------------------------------------
# extract_methodology
# ---------------------------------------------------------------------------


class ExtractMethodologyRequest(ToolRequest):
    paper_id: str
    section: str = "experimental_setup"


class ExtractMethodologyTool(Tool):
    name = "extract_methodology"
    description = (
        "Extract the experimental methodology from a specific section of a paper. "
        "Returns structured hyperparameters, preprocessing steps, and reported metrics."
    )
    request_model = ExtractMethodologyRequest

    def execute(self, request: ExtractMethodologyRequest) -> ToolResponse:
        pid = request.paper_id.lower()
        if "rag" in pid or "rerank" in pid:
            return ToolResponse.ok(
                paper_id=request.paper_id,
                dataset="MS-MARCO QA",
                model_architecture="Bi-Encoder (BGE-Small) + Cross-Encoder Reranker (BGE-Reranker-Large)",
                hyperparameters={
                    "stage1_retriever": "bge-small-en-v1.5",
                    "stage1_top_k": 50,
                    "stage2_reranker": "bge-reranker-large",
                    "final_top_k": 10,
                    "rerank": True,
                },
                preprocessing=["Chunk size 512", "Overlap 50"],
                training_details={"reranker_type": "cross-encoder"},
                reported_metrics={
                    "ndcg@10": 0.735,
                    "mrr@10": 0.712,
                    "improvement_pct": 0.131,
                    "accuracy": 0.735,
                },
            )

        return ToolResponse.ok(
            paper_id=request.paper_id,
            dataset="CIFAR-10",
            model_architecture="ResNet-20",
            hyperparameters={
                "optimizer": "SGD",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "epochs": 200,
                "batch_size": 128,
                "lr_schedule": "cosine",
            },
            preprocessing=[
                "RandomCrop(32, padding=4)",
                "RandomHorizontalFlip()",
                "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
            ],
            reported_metrics={"accuracy": 0.942},
        )


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


class RunExperimentRequest(ToolRequest):
    experiment_id: str
    parameters: dict = {}
    timeout_seconds: int = 300
    seed: int = 42


class RunExperimentTool(Tool):
    name = "run_experiment"
    description = (
        "Execute an experiment with the given parameters and return metrics. "
        "Supports classification (accuracy, f1) and RAG retrieval (ndcg@10, mrr@10, improvement_pct). "
        "Use experiment_id to track results. Set seed for reproducibility."
    )
    request_model = RunExperimentRequest

    def execute(self, request: RunExperimentRequest) -> ToolResponse:
        params_str = str(request.parameters).lower()

        # Check if this is a RAG / reranking experiment
        is_rag = any(kw in params_str for kw in ("rerank", "cross_encoder", "retriever", "bge", "ndcg", "mrr", "top_k"))
        has_rerank = any(kw in params_str for kw in ("rerank", "cross_encoder", "stage2")) and "false" not in params_str

        if is_rag or "rag" in request.experiment_id.lower():
            if has_rerank:
                ndcg = 0.735
                mrr = 0.712
                improvement = 0.131  # +13.1% relative improvement (> 10% target!)
                summary = "RAG with Cross-Encoder Reranking: NDCG@10=0.7350, MRR@10=0.7120 (+13.1% over baseline)"
            else:
                ndcg = 0.650
                mrr = 0.610
                improvement = 0.000  # Baseline: no improvement
                summary = "Baseline Dense Retrieval (no reranking): NDCG@10=0.6500, MRR@10=0.6100"

            return ToolResponse.ok(
                experiment_id=request.experiment_id,
                ndcg=ndcg,
                ndcg10=ndcg,
                mrr=mrr,
                improvement_pct=improvement,
                relative_improvement=improvement,
                accuracy=ndcg,  # compatibility alias
                runtime_seconds=8.5,
                stdout_summary=summary,
                metrics={
                    "ndcg@10": ndcg,
                    "mrr@10": mrr,
                    "improvement_pct": improvement,
                    "accuracy": ndcg,
                },
            )

        # Image classification fallback — complexity-aware latency proxy
        from ..agent.planning_context import canonicalize_parameters, normalize_parameters

        params = normalize_parameters(canonicalize_parameters(request.parameters or {}))
        has_normalize = bool(params.get("normalize")) or "normalize" in params_str
        has_cosine = "cosine" in params_str
        hidden = int(params.get("hidden_size") or 128)
        layers = int(params.get("hidden_layers") or 2)
        complexity = hidden * layers
        if complexity >= 256:
            accuracy, latency_ms = 0.961, 156.9
        elif hidden >= 128:
            accuracy, latency_ms = 0.9515, 137.1
        else:
            accuracy, latency_ms = 0.9125, 76.1
        if has_normalize:
            accuracy = min(0.97, round(accuracy + 0.045, 4))
        if has_cosine:
            accuracy = min(0.97, round(accuracy + 0.01, 4))

        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            accuracy=round(accuracy, 4),
            f1=round(accuracy - 0.005, 4),
            loss=round(0.42 - (accuracy - 0.867) * 0.5, 4),
            runtime_seconds=12.3,
            latency_ms=latency_ms,
            artifacts=[f"runs/{request.experiment_id}/model.pt"],
            stdout_summary=f"Epoch 200/200 — acc: {accuracy:.4f} latency_ms={latency_ms}",
            metrics={"accuracy": round(accuracy, 4), "latency_ms": latency_ms},
        )


# ---------------------------------------------------------------------------
# collect_metrics
# ---------------------------------------------------------------------------


class RunIndependentEvalRequest(ToolRequest):
    experiment_id: str
    timeout_seconds: int = 120
    seed: int = 1042


class StubIndependentEvalTool(Tool):
    name = "run_independent_evaluation"
    description = (
        "Re-score a completed experiment independently. Stub mode does not "
        "fabricate metrics; it reports that model artifacts are unavailable."
    )
    request_model = RunIndependentEvalRequest

    def execute(self, request: RunIndependentEvalRequest) -> ToolResponse:
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            eval_accuracy=None,
            training_accuracy=None,
            discrepancy=None,
            consistent=None,
            independently_verified=None,
            limitation="Stub independent evaluation does not re-score a trained model",
        )


class CollectMetricsRequest(ToolRequest):
    experiment_id: str
    metric_names: list[str] = ["accuracy", "f1", "loss", "ndcg@10", "improvement_pct"]


class CollectMetricsTool(Tool):
    name = "collect_metrics"
    description = "Collect and return metrics from a completed experiment."
    request_model = CollectMetricsRequest

    def execute(self, request: CollectMetricsRequest) -> ToolResponse:
        all_metrics = {
            "accuracy": 0.867,
            "f1": 0.862,
            "loss": 0.42,
            "ndcg@10": 0.735,
            "improvement_pct": 0.131,
        }
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            metrics={k: v for k, v in all_metrics.items() if not request.metric_names or k in request.metric_names},
        )


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class CompareResultsRequest(ToolRequest):
    experiment_id: str
    reported_value: float | None = None
    observed_value: float | None = None
    reported_accuracy: float | None = None
    observed_accuracy: float | None = None
    threshold: float = 0.005


class CompareResultsTool(Tool):
    name = "compare_results"
    description = (
        "Compare observed experiment results against the paper's reported results. "
        "Supports any metric (accuracy, ndcg@10, relative_improvement). "
        "Returns whether the result is within the acceptable threshold."
    )
    request_model = CompareResultsRequest

    def execute(self, request: CompareResultsRequest) -> ToolResponse:
        rep = request.reported_value if request.reported_value is not None else (request.reported_accuracy or 0.0)
        obs = request.observed_value if request.observed_value is not None else (request.observed_accuracy or 0.0)

        diff = abs(rep - obs)
        within = diff <= request.threshold
        pct = diff / max(rep, 1e-9) * 100
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            reported=round(rep, 4),
            observed=round(obs, 4),
            absolute_difference=round(diff, 4),
            relative_difference_pct=round(pct, 2),
            within_threshold=within,
            status="REPRODUCED" if within else "DISCREPANCY",
        )


# ---------------------------------------------------------------------------
# inspect_methodology
# ---------------------------------------------------------------------------


class InspectMethodologyRequest(ToolRequest):
    paper_id: str
    topic: str = "preprocessing"
    """Topic to inspect: 'preprocessing', 'optimizer', 'lr_schedule', 'reranker', 'retrieval'"""


class InspectMethodologyTool(Tool):
    name = "inspect_methodology"
    description = (
        "Inspect a specific aspect of a paper's methodology in detail. "
        "Use this after detecting a discrepancy to find the root cause."
    )
    request_model = InspectMethodologyRequest

    def execute(self, request: InspectMethodologyRequest) -> ToolResponse:
        topic_lower = request.topic.lower()
        if any(kw in topic_lower for kw in ("rerank", "retrieval", "cross_encoder", "rank")):
            details = {
                "pipeline": "Two-stage retrieval: dense candidate retrieval -> cross-encoder reranking.",
                "reranker_model": "bge-reranker-large",
                "candidate_pool_size": 50,
                "final_top_k": 10,
                "note": (
                    "Cross-encoder reranker computes cross-attention over query-passage pairs. "
                    "Omitting reranking drops NDCG@10 from 0.735 to 0.650 (-13.1% drop)."
                ),
            }
        else:
            details = {
                "preprocessing": {
                    "steps": [
                        "RandomCrop(32, padding=4)",
                        "RandomHorizontalFlip()",
                        "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
                    ],
                    "note": "Normalization is essential — omitting it typically costs 5-7% accuracy.",
                },
                "optimizer": {
                    "type": "SGD",
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "note": "Weight decay of 1e-4 is critical for CIFAR-10 with ResNet.",
                },
                "lr_schedule": {
                    "type": "cosine",
                    "epochs": 200,
                    "note": "Step LR loses ~3% accuracy compared to cosine on this task.",
                },
            }.get(
                request.topic,
                {"note": f"Details for topic: {request.topic}"},
            )

        return ToolResponse.ok(
            paper_id=request.paper_id,
            topic=request.topic,
            details=details,
        )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def build_stub_registry() -> ToolRegistry:
    """Build and return a ToolRegistry with all stub tools registered."""
    registry = ToolRegistry()
    registry.register(SearchLiteratureTool())
    registry.register(RetrievePaperTool())
    registry.register(ExtractMethodologyTool())
    registry.register(RunExperimentTool())
    registry.register(CollectMetricsTool())
    registry.register(CompareResultsTool())
    registry.register(InspectMethodologyTool())
    registry.register(StubIndependentEvalTool())
    return registry
