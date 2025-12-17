# rag_pipeline.py
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from mistralai import Mistral
from collections import defaultdict

from retriever_manager import RetrieverManager
from retriever_base import RetrievalResult


class RAGPipeline:
    """
    Pipeline RAG modulaire avec retrievers pluggables.
    """
    
    def __init__(
        self,
        retriever_manager: RetrieverManager,
        data_dir: str = "data",
        mistral_api_key: Optional[str] = None
    ):
        self.retriever_manager = retriever_manager
        self.data_dir = data_dir
        self.client = Mistral(api_key=mistral_api_key or os.environ.get("MISTRAL_API_KEY"))
    
    def load_doc_structure(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Charge la structure JSON d'un document."""
        for fname in os.listdir(self.data_dir):
            if fname.endswith("_structure.json"):
                with open(os.path.join(self.data_dir, fname), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    doc_idx = int(doc_id.split("_")[1])
                    all_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith("_structure.json")])
                    if all_files[doc_idx] == fname:
                        return doc
        return None
    
    def compute_doc_scores(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """DocScore = (1 / sqrt(N+1)) * sum_n ChunkScore(n)"""
        doc_chunks: Dict[str, List[float]] = defaultdict(list)
        
        for result in results:
            doc_chunks[result.doc_id].append(result.score)
        
        doc_scores: Dict[str, float] = {}
        for doc_id, scores in doc_chunks.items():
            N = len(scores)
            s = sum(scores)
            doc_scores[doc_id] = s / (N + 1) ** 0.5
        
        return doc_scores
    
    def get_doc_summary(
        self,
        doc_id: str,
        results: List[RetrievalResult],
        top_n: int = 8
    ) -> Dict[str, Any]:
        """Crée un résumé enrichi d'un document."""
        doc_results = [r for r in results if r.doc_id == doc_id]
        doc_results = sorted(doc_results, key=lambda x: x.score, reverse=True)[:top_n]
        
        if not doc_results:
            return {
                "doc_id": doc_id,
                "doc_name": "Unknown",
                "doc_description": None,
                "toc_preview": "",
                "top_excerpts": [],
                "avg_score": 0.0,
                "max_score": 0.0,
                "num_relevant_chunks": 0
            }
        
        doc_name = doc_results[0].doc_name
        avg_score = sum(r.score for r in doc_results) / len(doc_results)
        max_score = max(r.score for r in doc_results)
        num_relevant_chunks = len([r for r in results if r.doc_id == doc_id])
        
        # Charge la structure
        doc_structure = self.load_doc_structure(doc_id)
        
        doc_description = None
        if doc_structure and "doc_description" in doc_structure:
            doc_description = doc_structure["doc_description"]
        
        toc_preview = ""
        if doc_structure and "structure" in doc_structure:
            toc_items = []
            
            def extract_titles(nodes, depth=0, max_depth=4):
                if depth > max_depth:
                    return
                for node in nodes:
                    title = node.get("title", "")
                    if title:
                        toc_items.append("  " * depth + "- " + title)
                    if node.get("nodes"):
                        extract_titles(node.get("nodes", []), depth + 1, max_depth)
            
            extract_titles(doc_structure["structure"])
            toc_preview = "\n".join(toc_items[:35])
        
        top_excerpts = []
        for result in doc_results:
            excerpt = result.content[:500] + "..." if len(result.content) > 500 else result.content
            top_excerpts.append({
                "title": result.section,
                "node_id": result.node_id,
                "score": result.score,
                "excerpt": excerpt
            })
        
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_description": doc_description,
            "avg_score": avg_score,
            "max_score": max_score,
            "num_relevant_chunks": num_relevant_chunks,
            "toc_preview": toc_preview,
            "top_excerpts": top_excerpts
        }
    
    def select_documents(
        self,
        query: str,
        doc_summaries: List[Dict[str, Any]],
        max_primary: int = 1,
        max_fallback: int = 1
    ) -> Tuple[List[str], List[str], str]:
        """Sélection LLM des documents pertinents."""
        docs_info = []
        for i, doc_sum in enumerate(doc_summaries):
            doc_info = f"""
{'='*60}
Document {i+1}: {doc_sum['doc_name']}
{'='*60}
ID: {doc_sum['doc_id']}
"""
            
            if doc_sum.get('doc_description'):
                doc_info += f"\n📝 DESCRIPTION:\n{doc_sum['doc_description']}\n"
            
            if doc_sum.get('toc_preview'):
                doc_info += f"\n📑 TABLE DES MATIÈRES:\n{doc_sum['toc_preview']}\n"
            
            doc_info += f"\n🔍 EXTRAITS:\n"
            for j, excerpt in enumerate(doc_sum['top_excerpts'][:5], 1):
                doc_info += f"\n{j}. [{excerpt['title']}]\n{excerpt['excerpt']}\n"
            
            docs_info.append(doc_info)
        
        docs_text = "\n".join(docs_info)
        
        system_prompt = f"""Tu es un expert en sélection de documents.

Identifie:
1. **PRIMARY**: Les documents les plus pertinents (max {max_primary})
2. **FALLBACK**: Documents de contexte général (max {max_fallback}, optionnel)

Règles:
- PRIMARY = documents qui traitent DIRECTEMENT du sujet
- FALLBACK = contexte général uniquement si nécessaire
- Privilégie la qualité sur la quantité"""
        
        user_prompt = f"""Question: {query}

Documents:
{docs_text}

JSON:
{{
    "keywords": ["mot-clé1", "mot-clé2"],
    "primary_docs": ["doc_id_X"],
    "fallback_docs": [],
    "reasoning": "Justification courte"
}}"""
        
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            primary = result.get("primary_docs", [])[:max_primary]
            fallback = result.get("fallback_docs", [])[:max_fallback]
            reasoning = result.get("reasoning", "")
            keywords = result.get("keywords", [])
            
            print(f"\n🔑 Mots-clés: {', '.join(keywords)}")
            print(f"💭 Raisonnement: {reasoning}")
            
            return primary, fallback, reasoning
        except Exception as e:
            print(f"⚠️ Erreur parsing: {e}")
            primary = [doc_summaries[0]["doc_id"]] if doc_summaries else []
            return primary, [], "Fallback: erreur parsing"
    
    def smart_truncate(self, text: str, max_chars: int = 500) -> str:
        """Tronque intelligemment un texte."""
        if len(text) <= max_chars:
            return text
        
        start_chars = int(max_chars * 0.7)
        end_chars = int(max_chars * 0.3)
        
        return f"{text[:start_chars]}\n\n[...]\n\n{text[-end_chars:]}"
    
    def answer(
        self,
        query: str,
        retriever_name: Optional[str] = None,
        top_k_initial: int = 80,
        top_k_expanded: int = 120,
        max_docs_to_consider: int = 6,
        max_primary_docs: int = 1,
        max_fallback_docs: int = 0,
        chunks_per_primary: int = 4,
        chunks_per_fallback: int = 0,
        max_chunk_length: int = 500,
        use_hyde: bool = True,
        use_query_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Pipeline complet de RAG.
        """
        print("="*60)
        print("🔍 PHASE 1: Sélection documents")
        print("="*60 + "\n")
        
        # Récupère le retriever
        retriever = self.retriever_manager.get(retriever_name)
        print(f"📦 Utilisation: {retriever.name}")
        
        # Recherche initiale
        initial_results = retriever.search(query, top_k=top_k_initial)
        print(f"✅ {len(initial_results)} chunks trouvés")
        
        # Calcule scores documents
        doc_scores = self.compute_doc_scores(initial_results)
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_docs_to_consider]
        candidate_doc_ids = [doc_id for doc_id, _ in ranked_docs]
        
        print(f"\n🤖 Analyse LLM de {len(candidate_doc_ids)} candidats...")
        
        # Crée les résumés
        doc_summaries = []
        for doc_id in candidate_doc_ids:
            summary = self.get_doc_summary(doc_id, initial_results, top_n=8)
            doc_summaries.append(summary)
            print(f"  📄 {summary['doc_name']}: {summary['avg_score']:.3f}")
        
        # Sélection LLM
        primary_ids, fallback_ids, reasoning = self.select_documents(
            query, doc_summaries,
            max_primary=max_primary_docs,
            max_fallback=max_fallback_docs
        )
        
        if not primary_ids:
            print("⚠️ Aucun primary doc, fallback sur premier candidat")
            primary_ids = [candidate_doc_ids[0]]
        
        primary_names = [next((s['doc_name'] for s in doc_summaries if s['doc_id'] == d), d) for d in primary_ids]
        fallback_names = [next((s['doc_name'] for s in doc_summaries if s['doc_id'] == d), d) for d in fallback_ids]
        
        print(f"\n✅ PRIMARY ({len(primary_ids)}): {primary_names}")
        if fallback_ids:
            print(f"🔄 FALLBACK ({len(fallback_ids)}): {fallback_names}")
        
        # Phase 2: Retrieval approfondi
        print("\n" + "="*60)
        print("🔍 PHASE 2: Retrieval approfondi")
        print("="*60 + "\n")
        
        queries = [query]
        
        # HyDE
        if use_hyde and hasattr(retriever, 'generate_hyde'):
            hyde = retriever.generate_hyde(query)
            queries.append(hyde)
            print(f"✅ HyDE généré")
        
        # Query expansion
        if use_query_expansion and hasattr(retriever, 'expand_query'):
            expanded = retriever.expand_query(query, num_variants=2)
            queries.extend(expanded[1:])
            print(f"✅ {len(queries)} variantes totales")
        
        # Multi-query search avec filtres
        all_docs = primary_ids + fallback_ids
        
        if hasattr(retriever, 'search_multi_query'):
            final_results = retriever.search_multi_query(
                queries,
                top_k=top_k_expanded,
                filters={"doc_ids": all_docs}
            )
        else:
            # Fallback: recherche manuelle
            all_results_dict = {}
            for q in queries:
                results = retriever.search(q, top_k=top_k_expanded, filters={"doc_ids": all_docs})
                for r in results:
                    key = f"{r.doc_id}_{r.node_id}_{r.chunk_id}"
                    if key not in all_results_dict or r.score > all_results_dict[key].score:
                        all_results_dict[key] = r
            final_results = sorted(all_results_dict.values(), key=lambda x: x.score, reverse=True)
        
        print(f"✅ {len(final_results)} chunks récupérés")
        
        # Phase 3: Construction contexte
        print("\n" + "="*60)
        print("📄 PHASE 3: Construction contexte")
        print("="*60 + "\n")
        
        final_chunks = []
        seen_nodes: Set[Tuple[str, str]] = set()
        
        # PRIMARY docs
        for doc_id in primary_ids:
            doc_results = [r for r in final_results if r.doc_id == doc_id]
            doc_results = sorted(doc_results, key=lambda r: r.score, reverse=True)
            
            doc_name = doc_results[0].doc_name if doc_results else "Unknown"
            print(f"\n📄 PRIMARY: {doc_name}")
            
            for result in doc_results[:chunks_per_primary]:
                node_key = (result.doc_id, result.node_id)
                if node_key in seen_nodes:
                    continue
                seen_nodes.add(node_key)
                
                # Tronque si nécessaire
                result.content = self.smart_truncate(result.content, max_chunk_length)
                
                final_chunks.append(result)
                print(f"  ✅ {result.section} (~{len(result.content)} chars, score: {result.score:.3f})")
        
        # FALLBACK docs
        for doc_id in fallback_ids:
            doc_results = [r for r in final_results if r.doc_id == doc_id]
            doc_results = sorted(doc_results, key=lambda r: r.score, reverse=True)
            
            doc_name = doc_results[0].doc_name if doc_results else "Unknown"
            print(f"\n🔄 FALLBACK: {doc_name}")
            
            for result in doc_results[:chunks_per_fallback]:
                node_key = (result.doc_id, result.node_id)
                if node_key in seen_nodes:
                    continue
                seen_nodes.add(node_key)
                
                result.content = self.smart_truncate(result.content, max_chunk_length)
                
                final_chunks.append(result)
                print(f"  ✅ {result.section} (~{len(result.content)} chars, score: {result.score:.3f})")
        
        total_context_chars = sum(len(chunk.content) for chunk in final_chunks)
        print(f"\n✅ Total: {len(final_chunks)} chunks, ~{total_context_chars} caractères")
        
        # Construction du contexte
        context_parts = []
        for chunk in final_chunks:
            context_parts.append(f"[{chunk.doc_name} | {chunk.section}]\n{chunk.content}")
        
        context_str = "\n\n" + "─"*40 + "\n\n".join(context_parts)
        
        # Phase 4: Génération
        print("\n" + "="*60)
        print("💬 PHASE 4: Génération réponse")
        print("="*60 + "\n")
        
        system_prompt = """Tu es un assistant expert pour un centre de relation client.

STYLE DE RÉPONSE:

**Structure:**
- Commence par reformuler brièvement: "D'après le contexte fourni..."
- Organise clairement (titres ###, listes)
- Termine par: **Source(s):** [Document | Section]

**Mise en forme:**
- **Gras** pour infos critiques
- Listes numérotées pour étapes
- Listes à puces pour options

**Ton:** Précis, concis, actionnable"""
        
        user_prompt = f"""Question: {query}

Contexte:
{context_str}

Réponds en français de manière structurée."""
        
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content
        
        return {
            "query": query,
            "query_variants": queries,
            "answer": answer,
            "selection_reasoning": reasoning,
            "primary_docs": [
                {
                    "doc_id": doc_id,
                    "doc_name": next((s["doc_name"] for s in doc_summaries if s["doc_id"] == doc_id), "Unknown"),
                    "semantic_score": doc_scores.get(doc_id, 0.0)
                }
                for doc_id in primary_ids
            ],
            "fallback_docs": [
                {
                    "doc_id": doc_id,
                    "doc_name": next((s["doc_name"] for s in doc_summaries if s["doc_id"] == doc_id), "Unknown"),
                    "semantic_score": doc_scores.get(doc_id, 0.0)
                }
                for doc_id in fallback_ids
            ],
            "num_chunks": len(final_chunks),
            "total_context_chars": total_context_chars,
            "retriever_used": retriever.name
        }
