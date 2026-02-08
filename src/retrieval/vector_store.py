import json
from typing import List

import numpy as np

from models import Chunk, RetrievalResult
from retrieval.embeddings import embed_query, embed_texts


class VectorStore:
    def __init__(self) -> None:
        self.vectors: List[List[float]] = []
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        embeddings = embed_texts([chunk.text for chunk in chunks])
        self.vectors.extend(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        if not self.vectors:
            return []

        query_vec = np.array(embed_query(query))
        matrix = np.array(self.vectors)
        scores = self._cosine_similarity(matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            results.append(RetrievalResult(chunk=self.chunks[idx], score=float(scores[idx])))
        return results

    def save(self, path: str) -> None:
        payload = {
            "vectors": self.vectors,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    @staticmethod
    def load(path: str) -> "VectorStore":
        with open(path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        store = VectorStore()
        store.vectors = payload.get("vectors", [])
        store.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        return store

    @staticmethod
    def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        matrix_norm = np.linalg.norm(matrix, axis=1)
        vector_norm = np.linalg.norm(vector)
        denom = (matrix_norm * vector_norm) + 1e-8
        return np.dot(matrix, vector) / denom
