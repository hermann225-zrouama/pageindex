# vector_retriever.py
import os
import numpy as np
import faiss
from dataclasses import dataclass as dc
from typing import List, Dict, Any, Tuple, Optional
from mistralai import Mistral

from retriever_base import BaseRetriever, RetrievalResult, RetrieverType


MISTRAL_API_KEY = "rUqtUW7Az9sYVdRQI3Lo2Y6QWdIrVp4b"
EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-large-latest"


@dc
class ChunkMeta:
    """Métadonnées d'un chunk (structure interne)."""
    doc_id: str
    doc_name: str
    node_id: str
    title: str
    chunk_id: int
    text: str
    score: float = 0.0


class VectorRetriever(BaseRetriever):
    """
    Retriever vectoriel avec FAISS.
    Correspond à ton implémentation actuelle.
    """
    
    def __init__(
        self,
        index_path: str = "data/index_faiss.npz",
        data_dir: str = "data",
        mistral_api_key: Optional[str] = None
    ):
        self.index_path = index_path
        self.data_dir = data_dir
        self.client = Mistral(api_key=mistral_api_key or MISTRAL_API_KEY)
        
        # Charge l'index FAISS
        self.index, self.metas = self._load_index()
    
    @property
    def name(self) -> str:
        return "VectorRetriever"
    
    @property
    def retriever_type(self) -> RetrieverType:
        return RetrieverType.VECTOR
    
    def _load_index(self) -> Tuple[faiss.Index, List[ChunkMeta]]:
        """Charge l'index FAISS."""
        data = np.load(self.index_path, allow_pickle=True)
        vecs = data["vectors"]
        metas_raw = data["metas"]
        
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        
        metas = [ChunkMeta(**m) for m in metas_raw]
        return index, metas
    
    def _embed_query(self, text: str) -> np.ndarray:
        """Embed une requête."""
        res = self.client.embeddings.create(model=EMBED_MODEL, inputs=[text])
        v = np.array(res.data[0].embedding, dtype="float32")[None, :]
        faiss.normalize_L2(v)
        return v
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Recherche vectorielle avec FAISS."""
        # Embed la requête
        q_vec = self._embed_query(query)
        
        # Recherche FAISS
        D, I = self.index.search(q_vec, top_k * 2)  # Cherche 2x plus pour filtrer
        
        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metas[idx]
            
            # Applique les filtres
            if filters:
                if filters.get("doc_id") and meta.doc_id != filters["doc_id"]:
                    continue
                if filters.get("doc_ids") and meta.doc_id not in filters["doc_ids"]:
                    continue
            
            results.append(RetrievalResult(
                content=meta.text,
                doc_id=meta.doc_id,
                doc_name=meta.doc_name,
                section=meta.title,
                node_id=meta.node_id,
                chunk_id=meta.chunk_id,
                score=float(score),
                source_type=RetrieverType.VECTOR,
                metadata={
                    "index": int(idx),
                    "embedding_model": EMBED_MODEL
                }
            ))
        
        return results[:top_k]
    
    def search_multi_query(
        self,
        queries: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Multi-query search avec fusion des résultats.
        """
        all_results: Dict[str, RetrievalResult] = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k, filters=filters)
            
            for result in results:
                key = f"{result.doc_id}_{result.node_id}_{result.chunk_id}"
                
                # Garde le meilleur score
                if key not in all_results or result.score > all_results[key].score:
                    all_results[key] = result
        
        # Trie par score
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    def generate_hyde(self, query: str) -> str:
        """Génère un document hypothétique (HyDE)."""
        prompt = f"""Réponds brièvement à cette question:

Question: {query}

Réponse (2 paragraphes max):"""
        
        response = self.client.chat.complete(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    def expand_query(self, query: str, num_variants: int = 2) -> List[str]:
        """Génère des variantes de la requête."""
        prompt = f"""Génère {num_variants} reformulations courtes:

Question: {query}

Reformulations:
1."""
        
        response = self.client.chat.complete(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        reformulations = response.choices[0].message.content.strip().split('\n')
        queries = [query]
        
        for line in reformulations:
            line = line.strip()
            if line and len(line) > 10:
                line = line.lstrip('0123456789.-) ')
                if line:
                    queries.append(line)
        
        return queries[:num_variants + 1]
    
    def close(self):
        """Rien à fermer pour FAISS local."""
        pass
