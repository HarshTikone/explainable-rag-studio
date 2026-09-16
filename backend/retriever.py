"""Inspectable dense, MMR, and hybrid retrieval strategies."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .vectorstore import FaissStore
from .config import SETTINGS
from .reranker import Reranker, get_default_reranker
from .security_models import PUBLIC_ORGANIZATION_ID, RetrievalScope, SecurityBoundaryError
from .observability import span

RetrievalStrategy = Literal["lexical", "dense", "dense_mmr", "hybrid_rrf", "hybrid_rerank"]
VALID_STRATEGIES = ("lexical", "dense", "dense_mmr", "hybrid_rrf", "hybrid_rerank")
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", flags=re.UNICODE)
BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "which", "with",
}
BM25_QUERY_ALIASES = {
    "log": ("logs", "event", "events"),
    "logs": ("log", "event", "events"),
    "retention": ("retain", "retained"),
    "retain": ("retention", "retained"),
    "period": ("duration",),
}
HISTORICAL_INTENT = {"former", "historical", "incident", "legacy", "old", "obsolete", "previous", "retired"}
CURRENT_INTENT = {"current", "currently", "now", "today"}
HISTORICAL_MARKERS = ("status: obsolete", "retired", "legacy")


@dataclass(frozen=True)
class RetrievalHit:
    item: Dict[str, Any]
    rank: int
    final_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_rank: int | None = None
    fusion_score: float | None = None
    reranker_rank: int | None = None
    reranker_score: float | None = None
    stages: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {**self.item, **{key: value for key, value in asdict(self).items() if key != "item"}}


@dataclass(frozen=True)
class RetrievalResult:
    strategy: RetrievalStrategy
    hits: List[RetrievalHit]
    candidate_count: int
    latency_ms: float
    dense_latency_ms: float = 0.0
    lexical_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    reranking_trace: List[Dict[str, Any]] = field(default_factory=list)
    organization_id: str | None = None

    def as_legacy(self) -> List[Tuple[float, Dict[str, Any]]]:
        return [(hit.final_score, hit.item) for hit in self.hits]


def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase Unicode tokens while retaining dotted/hyphenated identifiers."""
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text or "")]


def tokenize_query_for_bm25(text: str) -> List[str]:
    """Remove query filler and add a small, deterministic operational synonym set."""
    base = [token for token in tokenize_for_bm25(text) if token not in BM25_STOPWORDS]
    return list(dict.fromkeys(
        token for value in base for token in (value, *BM25_QUERY_ALIASES.get(value, ()))
    ))


def reciprocal_rank_fusion(
    rankings: Sequence[Tuple[str, Sequence[Tuple[str, float]]]], rrf_k: int = 60
) -> List[Tuple[str, float, Dict[str, Tuple[int, float]]]]:
    """Fuse named rankings with deterministic tie ordering."""
    contributions: Dict[str, Dict[str, Tuple[int, float]]] = {}
    fused: Dict[str, float] = {}
    for stage, ranking in rankings:
        seen = set()
        for rank, (chunk_id, raw_score) in enumerate(ranking, start=1):
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            contributions.setdefault(chunk_id, {})[stage] = (rank, float(raw_score))
    return [
        (chunk_id, score, contributions[chunk_id])
        for chunk_id, score in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _mmr_indices(relevance: np.ndarray, vectors: np.ndarray, k: int, lambda_mult: float = 0.5) -> List[int]:
    if len(relevance) == 0 or k <= 0:
        return []
    similarities = vectors @ vectors.T
    selected = [int(relevance.argmax())]
    while len(selected) < min(k, len(relevance)):
        remaining = [index for index in range(len(relevance)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                lambda_mult * float(relevance[index])
                - (1 - lambda_mult) * max(float(similarities[index, chosen]) for chosen in selected),
                -index,
            ),
        )
        selected.append(next_index)
    return selected


def _dense_candidates(store: FaissStore, embedder, query: str, candidate_count: int):
    query_vector = embedder.embed_query(query)
    return query_vector, store.search(query_vector, candidate_count)


def _lexical_candidates(items: Sequence[Dict[str, Any]], query: str, candidate_count: int):
    corpus = [tokenize_for_bm25(item.get("retrieval_text", item.get("text", ""))) for item in items]
    query_tokens = tokenize_query_for_bm25(query)
    if not corpus or not query_tokens or not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    query_terms = set(query_tokens)
    # A contrast such as "current rather than obsolete" asks for the active policy,
    # even though the query necessarily mentions an historical marker.
    historical_query = bool(query_terms & HISTORICAL_INTENT) and not bool(query_terms & CURRENT_INTENT)

    def adjusted_score(index: int) -> float:
        matched_terms = len(set(query_tokens) & set(corpus[index]))
        score = float(scores[index]) + 2.0 * matched_terms
        text = str(items[index].get("retrieval_text", items[index].get("text", ""))).casefold()
        if not historical_query and any(marker in text for marker in HISTORICAL_MARKERS):
            return score * 0.75
        return score

    ranked = sorted(range(len(items)), key=lambda index: (-adjusted_score(index), index))
    return [(adjusted_score(index), items[index]) for index in ranked[:candidate_count] if adjusted_score(index) > 0.0]


def _obsolete_for_query(query: str, item: Dict[str, Any]) -> bool:
    """Identify lifecycle-stale evidence unless the visitor explicitly asks for history."""
    query_terms = set(tokenize_for_bm25(query))
    historical_query = bool(query_terms & HISTORICAL_INTENT) and not bool(query_terms & CURRENT_INTENT)
    if historical_query:
        return False
    source_version = str(item.get("source_version", "")).casefold()
    text = str(item.get("retrieval_text", item.get("text", ""))).casefold()
    return source_version in {"legacy", "obsolete", "retired"} or any(marker in text for marker in HISTORICAL_MARKERS)


def validate_store_scope(store: FaissStore, scope: RetrievalScope) -> None:
    """Fail closed before scoring when a physical index contains mixed tenants."""
    items = store.meta.get("items", [])
    for item in items:
        item_organization = item.get("organization_id")
        if item_organization is None and scope.organization_id == PUBLIC_ORGANIZATION_ID:
            continue  # legacy sanitized public index
        if item_organization != scope.organization_id:
            raise SecurityBoundaryError("Tenant boundary violation detected in vector metadata.")


def retrieve(
    store: FaissStore,
    embedder,
    query: str,
    top_k: int,
    strategy: RetrievalStrategy = "dense_mmr",
    reranker: Reranker | None = None,
    rerank_candidates: int | None = None,
    scope: RetrievalScope | None = None,
) -> RetrievalResult:
    started = time.perf_counter()
    normalized_query = " ".join((query or "").split())
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    if strategy == "hybrid_rerank" and rerank_candidates is not None and rerank_candidates < top_k:
        raise ValueError("rerank_candidates must be greater than or equal to top_k.")
    if scope is not None:
        validate_store_scope(store, scope)
    if not normalized_query or top_k <= 0 or not store.meta.get("items"):
        return RetrievalResult(strategy, [], 0, (time.perf_counter() - started) * 1000,
                               organization_id=scope.organization_id if scope else None)

    candidate_count = max(50, top_k * 5)
    dense = []
    dense_latency = 0.0
    if strategy != "lexical":
        dense_started = time.perf_counter()
        with span("rag.retrieval.dense", {"rag.candidate_count": candidate_count}):
            _, dense = _dense_candidates(store, embedder, normalized_query, candidate_count)
        dense_latency = (time.perf_counter() - dense_started) * 1000
    lexical_latency = fusion_latency = reranking_latency = 0.0
    reranking_trace: List[Dict[str, Any]] = []

    if strategy == "lexical":
        lexical_started = time.perf_counter()
        with span("rag.retrieval.lexical", {"rag.candidate_count": candidate_count}):
            lexical = _lexical_candidates(store.meta["items"], normalized_query, candidate_count)
        lexical_latency = (time.perf_counter() - lexical_started) * 1000
        hits = [
            RetrievalHit(
                item, rank, score,
                lexical_rank=rank, lexical_score=score, stages=("lexical",),
            )
            for rank, (score, item) in enumerate(lexical[:top_k], start=1)
        ]
    elif strategy == "dense":
        hits = [
            RetrievalHit(item, rank, score, rank, score, stages=("dense",))
            for rank, (score, item) in enumerate(dense[:top_k], start=1)
        ]
    elif strategy == "dense_mmr":
        if not dense:
            hits = []
        else:
            vectors = embedder.embed_texts([item["text"] for _, item in dense])
            indices = _mmr_indices(np.asarray([score for score, _ in dense], dtype="float32"), vectors, top_k)
            hits = [
                RetrievalHit(dense[index][1], rank, dense[index][0], index + 1, dense[index][0], stages=("dense", "mmr"))
                for rank, index in enumerate(indices, start=1)
            ]
    else:
        lexical_started = time.perf_counter()
        with span("rag.retrieval.lexical", {"rag.candidate_count": candidate_count}):
            lexical = _lexical_candidates(store.meta["items"], normalized_query, candidate_count)
        lexical_latency = (time.perf_counter() - lexical_started) * 1000
        dense_ranking = [(item["chunk_id"], score) for score, item in dense]
        lexical_ranking = [(item["chunk_id"], score) for score, item in lexical]
        fusion_started = time.perf_counter()
        with span("rag.retrieval.fusion", {"rag.candidate_count": candidate_count}):
            fused = reciprocal_rank_fusion((("dense", dense_ranking), ("lexical", lexical_ranking)))
        fusion_latency = (time.perf_counter() - fusion_started) * 1000
        items_by_id = {item["chunk_id"]: item for item in store.meta["items"]}
        fused_rows = fused
        reranker_scores = None
        if strategy == "hybrid_rerank":
            pool_size = min(rerank_candidates or SETTINGS.rerank_candidates, len(fused))
            rerank_pool = fused[:pool_size]
            active_reranker = reranker or get_default_reranker(
                SETTINGS.reranker_model, SETTINGS.rerank_batch_size
            )
            rerank_started = time.perf_counter()
            with span("rag.retrieval.rerank", {"rag.rerank.candidate_count": len(rerank_pool)}):
                scores = active_reranker.score(
                    normalized_query, [items_by_id[chunk_id]["text"] for chunk_id, _, _ in rerank_pool]
                )
            reranking_latency = (time.perf_counter() - rerank_started) * 1000
            if len(scores) != len(rerank_pool):
                raise ValueError("Reranker returned a different number of scores than candidate documents.")
            scored = [(*row, float(score_value)) for row, score_value in zip(rerank_pool, scores)]
            # Cross-encoders can overvalue a stale passage when a contrastive question
            # quotes both the current and obsolete values. Corpus lifecycle metadata is
            # a deterministic authority: keep obsolete evidence behind active evidence
            # unless the question explicitly asks for historical/incident information.
            # Python's stable sort preserves the fused ranking when both keys tie.
            fused_rows = sorted(
                scored,
                key=lambda row: (
                    _obsolete_for_query(normalized_query, items_by_id[row[0]]),
                    -row[3],
                ),
            )
            reranker_scores = {chunk_id: (rank, score) for rank, (chunk_id, _, _, score) in enumerate(fused_rows, 1)}
            fusion_positions = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(fused, 1)}
            reranking_trace = [
                {
                    "chunk_id": chunk_id,
                    "fusion_rank": fusion_positions[chunk_id],
                    "fusion_score": fusion_score,
                    "reranker_rank": rank,
                    "reranker_score": reranker_score,
                    "obsolete_for_query": _obsolete_for_query(normalized_query, items_by_id[chunk_id]),
                    "movement": fusion_positions[chunk_id] - rank,
                }
                for rank, (chunk_id, fusion_score, _, reranker_score) in enumerate(fused_rows, 1)
            ]

        hits = []
        for rank, row in enumerate(fused_rows[:top_k], start=1):
            chunk_id, score, stages = row[:3]
            dense_info = stages.get("dense")
            lexical_info = stages.get("lexical")
            fusion_rank = next(index for index, value in enumerate(fused, 1) if value[0] == chunk_id)
            reranker_info = reranker_scores.get(chunk_id) if reranker_scores else None
            hits.append(RetrievalHit(
                item=items_by_id[chunk_id], rank=rank,
                final_score=reranker_info[1] if reranker_info else score,
                dense_rank=dense_info[0] if dense_info else None,
                dense_score=dense_info[1] if dense_info else None,
                lexical_rank=lexical_info[0] if lexical_info else None,
                lexical_score=lexical_info[1] if lexical_info else None,
                fusion_rank=fusion_rank, fusion_score=score,
                reranker_rank=reranker_info[0] if reranker_info else None,
                reranker_score=reranker_info[1] if reranker_info else None,
                stages=tuple(stages) + (("reranker",) if reranker_info else ()),
            ))
    return RetrievalResult(
        strategy, hits, candidate_count, (time.perf_counter() - started) * 1000,
        dense_latency, lexical_latency, fusion_latency, reranking_latency, reranking_trace,
        scope.organization_id if scope else None,
    )
