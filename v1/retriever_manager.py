# retriever_manager.py
from typing import List, Dict, Any, Optional
from collections import defaultdict

from retriever_base import BaseRetriever, RetrievalResult


class RetrieverManager:
    """
    Gestionnaire de retrievers multiples.
    Permet d'enregistrer et d'utiliser plusieurs retrievers.
    """
    
    def __init__(self):
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.default_retriever: Optional[str] = None
    
    def register(
        self, 
        retriever: BaseRetriever, 
        set_as_default: bool = False
    ):
        """
        Enregistre un retriever.
        
        Args:
            retriever: Instance du retriever
            set_as_default: Le définir comme retriever par défaut
        """
        self.retrievers[retriever.name] = retriever
        
        if set_as_default or not self.default_retriever:
            self.default_retriever = retriever.name
        
        print(f"✅ Retriever '{retriever.name}' enregistré")
    
    def get(self, name: Optional[str] = None) -> BaseRetriever:
        """Récupère un retriever par nom (ou le défaut)."""
        retriever_name = name or self.default_retriever
        
        if not retriever_name or retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' non trouvé")
        
        return self.retrievers[retriever_name]
    
    def search(
        self,
        query: str,
        retriever_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Recherche avec un retriever spécifique."""
        retriever = self.get(retriever_name)
        return retriever.search(query, top_k=top_k, filters=filters)
    
    def search_multiple(
        self,
        query: str,
        retriever_names: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fusion_strategy: str = "weighted"
    ) -> List[RetrievalResult]:
        """
        Recherche avec plusieurs retrievers et fusion des résultats.
        
        Args:
            fusion_strategy: "weighted", "reciprocal_rank", "max_score"
        """
        all_results: Dict[str, List[RetrievalResult]] = defaultdict(list)
        
        # Collecte les résultats de chaque retriever
        for retriever_name in retriever_names:
            retriever = self.get(retriever_name)
            results = retriever.search(query, top_k=top_k, filters=filters)
            
            for result in results:
                key = f"{result.doc_id}_{result.node_id}_{result.chunk_id}"
                all_results[key].append(result)
        
        # Fusion selon stratégie
        if fusion_strategy == "weighted":
            merged = self._weighted_fusion(all_results, len(retriever_names))
        elif fusion_strategy == "reciprocal_rank":
            merged = self._reciprocal_rank_fusion(all_results)
        else:  # max_score
            merged = self._max_score_fusion(all_results)
        
        return sorted(merged, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _weighted_fusion(
        self,
        results_by_key: Dict[str, List[RetrievalResult]],
        num_retrievers: int
    ) -> List[RetrievalResult]:
        """Fusion avec moyenne pondérée des scores."""
        merged = []
        
        for key, results in results_by_key.items():
            # Moyenne des scores
            avg_score = sum(r.score for r in results) / len(results)
            
            # Bonus si trouvé par plusieurs retrievers
            consensus_bonus = (len(results) / num_retrievers) * 0.2
            final_score = avg_score + consensus_bonus
            
            # Garde le premier résultat mais avec score fusionné
            result = results[0]
            result.score = final_score
            merged.append(result)
        
        return merged
    
    def _reciprocal_rank_fusion(
        self,
        results_by_key: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF)."""
        K = 60  # Constante RRF
        merged = []
        
        for key, results in results_by_key.items():
            # RRF score = sum(1 / (K + rank))
            rrf_score = sum(1.0 / (K + i + 1) for i in range(len(results)))
            
            result = results[0]
            result.score = rrf_score
            merged.append(result)
        
        return merged
    
    def _max_score_fusion(
        self,
        results_by_key: Dict[str, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Fusion en gardant le score maximum."""
        merged = []
        
        for key, results in results_by_key.items():
            # Garde le meilleur score
            best_result = max(results, key=lambda r: r.score)
            merged.append(best_result)
        
        return merged
    
    def list_retrievers(self) -> List[str]:
        """Liste les retrievers enregistrés."""
        return list(self.retrievers.keys())
    
    def close_all(self):
        """Ferme tous les retrievers."""
        for retriever in self.retrievers.values():
            retriever.close()

