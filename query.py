# query.py
import os
import json
import numpy as np
import faiss
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from mistralai import Mistral

MISTRAL_API_KEY = ""
EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-large-latest"

INDEX_PATH = "data/index_faiss.npz"
DATA_DIR = "data"


@dataclass
class ChunkMeta:
    doc_id: str
    doc_name: str
    node_id: str
    title: str
    chunk_id: int
    text: str
    score: float = 0.0


def load_index() -> Tuple[faiss.Index, List[ChunkMeta]]:
    data = np.load(INDEX_PATH, allow_pickle=True)
    vecs = data["vectors"]
    metas_raw = data["metas"]

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    metas: List[ChunkMeta] = []
    for m in metas_raw:
        metas.append(ChunkMeta(**m))

    return index, metas


def embed_query(text: str, client: Mistral) -> np.ndarray:
    res = client.embeddings.create(model=EMBED_MODEL, inputs=[text])
    v = np.array(res.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v


def compute_doc_scores(chunk_scores: List[Tuple[ChunkMeta, float]]) -> Dict[str, float]:
    """
    Implémente la formule :
        DocScore = (1 / sqrt(N+1)) * sum_n ChunkScore(n)
    où N = nb de chunks associés au doc.
    """
    doc_chunks: Dict[str, List[float]] = defaultdict(list)
    for meta, s in chunk_scores:
        doc_chunks[meta.doc_id].append(s)

    doc_scores: Dict[str, float] = {}
    for doc_id, scores in doc_chunks.items():
        N = len(scores)
        s = sum(scores)
        doc_scores[doc_id] = s / np.sqrt(N + 1.0)
    return doc_scores


def load_doc_by_id(doc_id: str) -> Dict[str, Any]:
    # Ici on suppose que doc_id = "doc_k" correspond à data/doc_k.json
    idx = int(doc_id.split("_")[1])
    path = os.path.join(DATA_DIR, f"doc_{idx}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_context(doc_ids: List[str], metas: List[ChunkMeta], top_k_chunks_per_doc: int = 5) -> List[ChunkMeta]:
    # Filtrer les chunks appartenant aux meilleurs docs et garder top-k par doc
    selected: List[ChunkMeta] = []
    for doc_id in doc_ids:
        doc_chunks = [m for m in metas if m.doc_id == doc_id]
        # tri décroissant par score (rempli avant l’appel)
        doc_chunks = sorted(doc_chunks, key=lambda m: m.score, reverse=True)
        selected.extend(doc_chunks[:top_k_chunks_per_doc])
    return selected


def answer_query(query: str, top_k: int = 20, top_docs: int = 3) -> Dict[str, Any]:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY non défini")

    client = Mistral(api_key=MISTRAL_API_KEY)

    index, metas = load_index()
    q_vec = embed_query(query, client)

    D, I = index.search(q_vec, top_k)  # D: scores, I: indices
    scores = D[0]
    idxs = I[0]

    chunk_hits: List[Tuple[ChunkMeta, float]] = []
    for score, idx in zip(scores, idxs):
        meta = metas[idx]
        meta.score = float(score)
        chunk_hits.append((meta, float(score)))

    # Calcul DocScore
    doc_scores = compute_doc_scores(chunk_hits)
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in ranked_docs[:top_docs]]

    # Contexte pour génération
    context_chunks = choose_context(top_doc_ids, [m for m, _ in chunk_hits], top_k_chunks_per_doc=5)

    context_str_parts = []
    for ch in context_chunks:
        context_str_parts.append(
            f"[{ch.doc_name} | node {ch.node_id} | chunk {ch.chunk_id} | score={ch.score:.3f}]\n{ch.text}"
        )
    context_str = "\n\n".join(context_str_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context.\n"
        "If the answer is not in the context, say you don't know.\n"
    )
    user_prompt = f"Question:\n{query}\n\nContext:\n{context_str}\n\nAnswer in French, concise but precise."

    chat = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    answer = chat.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "top_docs": [
            {"doc_id": doc_id, "score": doc_scores[doc_id]} for doc_id in top_doc_ids
        ],
        "used_chunks": [
            {
                "doc_id": ch.doc_id,
                "doc_name": ch.doc_name,
                "node_id": ch.node_id,
                "title": ch.title,
                "chunk_id": ch.chunk_id,
                "score": ch.score,
            }
            for ch in context_chunks
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-docs", type=int, default=3)
    args = parser.parse_args()

    res = answer_query(args.query, top_k=args.top_k, top_docs=args.top_docs)
    print("\n=== ANSWER ===\n")
    print(res["answer"])
    print("\n=== DOCS ===")
    for d in res["top_docs"]:
        print(d)
