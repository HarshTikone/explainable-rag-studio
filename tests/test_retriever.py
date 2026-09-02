import numpy as np

from backend.retriever import reciprocal_rank_fusion, retrieve, tokenize_for_bm25


class FakeStore:
    def __init__(self):
        self.meta = {"items": [
            {"chunk_id": "broad", "text": "General troubleshooting for network errors", "source": "a", "page": 1},
            {"chunk_id": "exact", "text": "Error NX-417 requires hybrid retrieval", "source": "b", "page": 1},
            {"chunk_id": "other", "text": "Unrelated billing guidance", "source": "c", "page": 1},
        ]}

    def search(self, query_vector, top_k):
        return [(0.95, self.meta["items"][0]), (0.80, self.meta["items"][1]), (0.10, self.meta["items"][2])][:top_k]


class FakeEmbedder:
    def embed_query(self, query):
        return np.asarray([[1.0, 0.0]], dtype="float32")

    def embed_texts(self, texts):
        vectors = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
        return np.asarray(vectors[:len(texts)], dtype="float32")


class FakeReranker:
    model_name = "fake-cross-encoder"

    def __init__(self, scores):
        self.scores = scores
        self.documents = []

    def score(self, query, documents):
        self.documents = list(documents)
        return self.scores[:len(documents)]


def test_tokenizer_preserves_identifiers():
    assert tokenize_for_bm25("Fix TS-999 and api.v2") == ["fix", "ts-999", "and", "api.v2"]


def test_rrf_deduplicates_and_has_stable_ties():
    fused = reciprocal_rank_fusion((("dense", [("b", 1), ("a", .5), ("a", .4)]), ("lexical", [("a", 2)])))
    assert [item[0] for item in fused] == ["a", "b"]
    assert set(fused[0][2]) == {"dense", "lexical"}


def test_all_retrieval_strategies_return_structured_hits():
    store, embedder = FakeStore(), FakeEmbedder()
    dense = retrieve(store, embedder, "network problem", 2, "dense")
    mmr = retrieve(store, embedder, "network problem", 2, "dense_mmr")
    hybrid = retrieve(store, embedder, "NX-417", 2, "hybrid_rrf")
    assert dense.hits[0].item["chunk_id"] == "broad"
    assert "mmr" in mmr.hits[0].stages
    assert hybrid.hits[0].item["chunk_id"] == "exact"
    assert hybrid.hits[0].lexical_rank == 1


def test_hybrid_rerank_propagates_scores_and_truncates_candidates():
    store, embedder = FakeStore(), FakeEmbedder()
    reranker = FakeReranker([0.1, 0.9])
    result = retrieve(store, embedder, "NX-417", 2, "hybrid_rerank", reranker, rerank_candidates=2)
    assert len(reranker.documents) == 2
    assert len(result.reranking_trace) == 2
    assert result.hits[0].reranker_score == 0.9
    assert result.hits[0].reranker_rank == 1
    assert result.hits[0].fusion_rank == 2
    assert "reranker" in result.hits[0].stages


def test_reranking_trace_keeps_candidates_demoted_out_of_top_k():
    result = retrieve(
        FakeStore(), FakeEmbedder(), "NX-417", 2, "hybrid_rerank", FakeReranker([0.8, 0.7, 0.1]), rerank_candidates=3
    )
    assert len(result.hits) == 2
    assert len(result.reranking_trace) == 3
    assert {row["chunk_id"] for row in result.reranking_trace} == {"broad", "exact", "other"}


def test_hybrid_rerank_stable_ties_keep_fusion_order():
    store, embedder = FakeStore(), FakeEmbedder()
    fused = retrieve(store, embedder, "NX-417", 2, "hybrid_rrf")
    reranked = retrieve(
        store, embedder, "NX-417", 2, "hybrid_rerank", FakeReranker([0.5, 0.5]), rerank_candidates=2
    )
    assert [hit.item["chunk_id"] for hit in reranked.hits] == [hit.item["chunk_id"] for hit in fused.hits]


def test_rerank_candidate_pool_must_cover_top_k():
    try:
        retrieve(FakeStore(), FakeEmbedder(), "query", 3, "hybrid_rerank", FakeReranker([]), rerank_candidates=2)
        assert False, "undersized rerank pool should fail"
    except ValueError as exc:
        assert "greater than or equal" in str(exc)


def test_empty_and_invalid_queries():
    store, embedder = FakeStore(), FakeEmbedder()
    assert retrieve(store, embedder, "   ", 5, "dense").hits == []
    try:
        retrieve(store, embedder, "query", 5, "unknown")
        assert False, "unknown strategy should fail"
    except ValueError:
        pass
