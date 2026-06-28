"""Embeddings + FAISS index for chunk retrieval.

Two embedders:
- `BgeM3Embedder`: real model (BAAI/bge-m3). Slow first call, cached afterwards.
- `FakeEmbedder`: deterministic hash-based vectors for tests.

The FAISS index uses inner-product on L2-normalized vectors (= cosine sim).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol
import numpy as np


class Embedder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


@dataclass
class BgeM3Embedder:
    """Real BGE-M3 embedder. Loaded lazily; model cached under data/models/."""
    model_name: str = "BAAI/bge-m3"
    cache_dir: str = "data/models"

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


class FakeEmbedder:
    """Deterministic embedder: hashes text -> a fixed-dim unit vector.

    Same text -> same vector. Useful in tests so we don't load bge-m3.
    Implemented as a regular class (NOT a dataclass) so the `dim` property
    does not collide with a dataclass field of the same name.
    """
    def __init__(self, dim: int = 64):
        self._dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Token-bag-hash embedder: splits text into UTF-8 byte n-grams
        and hashes each n-gram into a signed bucket. Texts sharing substrings
        (e.g. "วิศวะ" and "วิศวกรรมศาสตร์") share many buckets and therefore
        have high cosine similarity. Output is L2-normalized.
        """
        import hashlib
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        n = 4  # byte n-gram size
        for i, t in enumerate(texts):
            data = t.encode("utf-8")
            if len(data) < n:
                data = data + b"\x00" * (n - len(data))
            for j in range(len(data) - n + 1):
                gram = data[j:j + n]
                bucket = int.from_bytes(
                    hashlib.blake2b(gram, digest_size=4).digest(), "big"
                ) % self._dim
                # +1 for first occurrence, -1 for second => signed bag
                if out[i, bucket] >= 0:
                    out[i, bucket] += 1
                else:
                    out[i, bucket] += 1  # accumulate count, signed at end
            # Sign each bucket by parity of count
            signed = np.sign(out[i]) * np.sqrt(np.abs(out[i]))
            out[i] = signed
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    @property
    def dim(self) -> int:
        return self._dim


def _build_faiss(dim: int):
    import faiss
    return faiss.IndexFlatIP(dim)


def build_index(chunks: list[dict], embedder: Embedder):
    """Build an in-memory FAISS index from chunk dicts.

    Each chunk dict must have at least: id, source_document_id, text.
    Returns: (faiss_index, list_of_chunk_dicts_in_index_order)
    """
    texts = [c["text"] for c in chunks]
    vecs = embedder.encode(texts).astype(np.float32)
    index = _build_faiss(embedder.dim)
    index.add(vecs)
    return (index, chunks)


def search(index_tuple, query: str, embedder: Embedder, k: int = 5, threshold: float = 0.5) -> list[dict]:
    """Return top-k chunks above `threshold` cosine similarity.

    Each result: {"chunk_id", "source_document_id", "text", "score"}.
    """
    index, chunks = index_tuple
    qvec = embedder.encode([query]).astype(np.float32)
    scores, ids = index.search(qvec, min(k, len(chunks)))
    out = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        if score < threshold:
            continue
        c = chunks[idx]
        out.append({
            "chunk_id": c["id"],
            "source_document_id": c["source_document_id"],
            "text": c["text"],
            "score": float(score),
        })
    return out
