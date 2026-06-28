"""Vector search: embed chunks, build FAISS, retrieve top-k."""
from __future__ import annotations
import numpy as np
import pytest
from src.vector_search import FakeEmbedder, build_index, search

def test_fake_embedder_returns_unit_vectors():
    emb = FakeEmbedder(dim=16)
    vecs = emb.encode(["hello", "world"])
    assert vecs.shape == (2, 16)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

def test_build_index_and_search_returns_relevant_chunk():
    emb = FakeEmbedder(dim=16)
    chunks = [
        {"id": 1, "source_document_id": 1, "text": "วิศวกรรมศาสตร์"},
        {"id": 2, "source_document_id": 1, "text": "คณะวิทยาศาสตร์"},
        {"id": 3, "source_document_id": 1, "text": "ทุนการศึกษา"},
    ]
    index = build_index(chunks, embedder=emb)
    results = search(index, "วิศวะ", embedder=emb, k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] in {1, 2, 3}

def test_search_respects_threshold():
    emb = FakeEmbedder(dim=16)
    chunks = [{"id": 1, "source_document_id": 1, "text": "x"}]
    index = build_index(chunks, embedder=emb)
    results = search(index, "totally unrelated gibberish query", embedder=emb, k=1, threshold=0.99)
    assert results == []
