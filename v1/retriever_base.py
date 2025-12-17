from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class RetrieverType(Enum):
    """Types de retrievers disponibles."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    BM25 = "bm25"


@dataclass
class RetrievalResult:
    """Résultat unifié de retrieval."""
    content: str
    doc_id: str
    doc_name: str
    section: str
    node_id: str
    chunk_id: int
    score: float
    source_type: RetrieverType
    metadata: Dict[str, Any]


class BaseRetriever(ABC):
    """Interface abstraite pour tous les retrievers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du retriever."""
        pass
    
    @property
    @abstractmethod
    def retriever_type(self) -> RetrieverType:
        """Type du retriever."""
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Recherche principale.
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats
            filters: Filtres optionnels (doc_id, date, etc.)
        
        Returns:
            Liste de RetrievalResult triés par score
        """
        pass
    
    @abstractmethod
    def close(self):
        """Ferme les connexions/ressources."""
        pass

