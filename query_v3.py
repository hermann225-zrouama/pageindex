# query_v6.py - Hybrid avec contexte MINIMAL et réponses structurées
import os
import json
import numpy as np
import faiss
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set
from mistralai import Mistral

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "rUqtUW7Az9sYVdRQI3Lo2Y6QWdIrVp4b")
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
    source: str = "primary"


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


def generate_hyde_document(query: str, client: Mistral) -> str:
    """HyDE pour améliorer la recherche."""
    prompt = f"""Réponds brièvement à cette question:

Question: {query}

Réponse (2 paragraphes max):"""

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


def expand_query(query: str, client: Mistral, num_variants: int = 2) -> List[str]:
    """Génère des variantes de question."""
    prompt = f"""Génère {num_variants} reformulations courtes:

    Question: {query}

    Reformulations:
    1."""

    response = client.chat.complete(
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


def load_doc_structure(doc_id: str) -> Dict[str, Any]:
    """Charge la structure d'un document."""
    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_structure.json"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                doc = json.load(f)
                doc_idx = int(doc_id.split("_")[1])
                all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_structure.json")])
                if all_files[doc_idx] == fname:
                    return doc
    return None
                                  

def compute_doc_scores(chunk_scores: List[Tuple[ChunkMeta, float]]) -> Dict[str, float]:
    """DocScore = (1 / sqrt(N+1)) * sum_n ChunkScore(n)"""
    doc_chunks: Dict[str, List[float]] = defaultdict(list)
    for meta, s in chunk_scores:
        doc_chunks[meta.doc_id].append(s)

    doc_scores: Dict[str, float] = {}
    for doc_id, scores in doc_chunks.items():
        N = len(scores)
        s = sum(scores)
        doc_scores[doc_id] = s / np.sqrt(N + 1.0)
    return doc_scores


def get_doc_summary(doc_id: str, chunk_hits: List[Tuple[ChunkMeta, float]], top_n: int = 8) -> Dict[str, Any]:
    """Résumé enrichi pour sélection."""
    doc_chunks = [(m, s) for m, s in chunk_hits if m.doc_id == doc_id]
    doc_chunks = sorted(doc_chunks, key=lambda x: x[1], reverse=True)[:top_n]
    
    if not doc_chunks:
        return {
            "doc_id": doc_id,
            "doc_name": "Unknown",
            "doc_description": None,
            "toc_preview": "",
            "top_chunks": [],
            "avg_score": 0.0,
            "max_score": 0.0,
            "num_relevant_chunks": 0
        }
    
    doc_name = doc_chunks[0][0].doc_name
    avg_score = sum(s for _, s in doc_chunks) / len(doc_chunks)
    max_score = max(s for _, s in doc_chunks)
    num_relevant_chunks = len([m for m, s in chunk_hits if m.doc_id == doc_id])
    
    doc_structure = load_doc_structure(doc_id)
    
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
    for meta, score in doc_chunks:
        excerpt = meta.text[:500] + "..." if len(meta.text) > 500 else meta.text
        top_excerpts.append({
            "title": meta.title,
            "node_id": meta.node_id,
            "score": score,
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


def llm_select_documents(
    query: str, 
    doc_summaries: List[Dict[str, Any]], 
    client: Mistral, 
    max_docs: int = 1
) -> Tuple[List[str], List[str], str]:
    """
    Sélection LLM stricte + suggestions de fallback.
    """
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
    
    system_prompt = """Tu es un expert en sélection de documents.

    Identifie:
    1. **PRIMARY**: LE document le plus spécifique et pertinent (1 seul)
    2. **FALLBACK**: 1 document de contexte général (optionnel, seulement si vraiment utile)

    Règles:
    - PRIMARY = document spécialisé qui traite DIRECTEMENT du sujet
    - FALLBACK = contexte général/procédure de base (seulement si nécessaire)
    - Si un seul doc suffit, ne mets pas de FALLBACK"""

    user_prompt = f"""Question: {query}

        Documents:
        {docs_text}

        JSON:
        {{
            "keywords": ["mot-clé1", "mot-clé2"],
            "primary_docs": ["doc_id_X"],
            "fallback_docs": ["doc_id_Y"],
            "reasoning": "Justification courte"
        }}"""

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        primary = result.get("primary_docs", [])[:1]
        fallback = result.get("fallback_docs", [])[:1]
        reasoning = result.get("reasoning", "")
        keywords = result.get("keywords", [])
        
        print(f"\n Primary docs: {primary}")
        print(f"\n🔑 Mots-clés: {', '.join(keywords)}")
        print(f"💭 Raisonnement: {reasoning}")
        
        return primary, fallback, reasoning
    except Exception as e:
        print(f"⚠️  Parsing error: {e}")
        primary = [doc_summaries[0]["doc_id"]] if doc_summaries else []
        return primary, [], "Fallback: erreur parsing"


def smart_truncate_text(text: str, max_chars: int = 500) -> str:
    """Tronque intelligemment: garde début + fin."""
    if len(text) <= max_chars:
        return text
    
    start_chars = int(max_chars * 0.7)
    end_chars = int(max_chars * 0.3)
    
    start = text[:start_chars]
    end = text[-end_chars:]
    
    return f"{start}\n\n[...]\n\n{end}"


def answer_query_v6(
    query: str, 
    top_k_initial: int = 80,
    top_k_expanded: int = 120,
    max_docs_to_consider: int = 6,
    primary_docs: int = 1,
    chunks_per_primary: int = 4,
    chunks_per_fallback: int = 3,
    max_chunk_length: int = 500,
    use_hyde: bool = True,
    use_query_expansion: bool = True
) -> Dict[str, Any]:
    """
    Version 6: Contexte MINIMAL avec réponses structurées adaptatives.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY non défini")

    client = Mistral(api_key=MISTRAL_API_KEY)

    print("="*60)
    print("🔍 PHASE 1: Sélection documents")
    print("="*60 + "\n")
    
    # Recherche initiale avec question brute
    index, metas = load_index()
    q_vec = embed_query(query, client)
    D, I = index.search(q_vec, top_k_initial)
    
    initial_chunk_hits: List[Tuple[ChunkMeta, float]] = []
    for score, idx in zip(D[0], I[0]):
        meta = metas[idx]
        meta.score = float(score)
        initial_chunk_hits.append((meta, float(score)))
    
    print(f"✅ {len(initial_chunk_hits)} chunks trouvés")
    
    doc_scores = compute_doc_scores(initial_chunk_hits)
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_docs_to_consider]
    candidate_doc_ids = [doc_id for doc_id, _ in ranked_docs]
    
    print(f"\n🤖 Analyse LLM de {len(candidate_doc_ids)} candidats...")
    
    doc_summaries = []
    for doc_id in candidate_doc_ids:
        summary = get_doc_summary(doc_id, initial_chunk_hits, top_n=8)
        doc_summaries.append(summary)
        print(f"  📄 {summary['doc_name']}: {summary['avg_score']:.3f}")
    
    # Sélection avec classification primary/fallback
    primary_ids, fallback_ids, reasoning = llm_select_documents(
        query, doc_summaries, client, max_docs=primary_docs
    )
    
    if not primary_ids:
        print("⚠️  Aucun primary doc, fallback sur premier candidat")
        primary_ids = [candidate_doc_ids[0]]
    
    primary_names = [next((s['doc_name'] for s in doc_summaries if s['doc_id'] == d), d) for d in primary_ids]
    fallback_names = [next((s['doc_name'] for s in doc_summaries if s['doc_id'] == d), d) for d in fallback_ids]
    
    print(f"\n✅ PRIMARY: {primary_names}")
    if fallback_ids:
        print(f"🔄 FALLBACK: {fallback_names}")
    
    print("\n" + "="*60)
    print("🔍 PHASE 2: Retrieval approfondi")
    print("="*60 + "\n")
    
    # Générer variantes de requêtes
    queries = [query]
    
    if use_hyde:
        hyde = generate_hyde_document(query, client)
        queries.append(hyde)
        print(f"✅ HyDE généré")
    
    if use_query_expansion:
        expanded = expand_query(query, client, num_variants=2)
        queries.extend(expanded[1:])
        print(f"✅ {len(queries)} variantes totales")
    
    # Multi-query search
    all_chunk_hits: Dict[int, Tuple[ChunkMeta, float]] = {}
    all_docs = primary_ids + fallback_ids
    
    for q in queries:
        q_vec = embed_query(q, client)
        D, I = index.search(q_vec, top_k_expanded)
        
        for score, idx in zip(D[0], I[0]):
            meta = metas[idx]
            
            if meta.doc_id not in all_docs:
                continue
            
            if meta.doc_id in primary_ids:
                meta.source = "primary"
            else:
                meta.source = "fallback"
            
            if idx not in all_chunk_hits or score > all_chunk_hits[idx][1]:
                meta.score = float(score)
                all_chunk_hits[idx] = (meta, float(score))
    
    chunk_hits = list(all_chunk_hits.values())
    chunk_hits = sorted(chunk_hits, key=lambda x: x[1], reverse=True)
    
    print(f"✅ {len(chunk_hits)} chunks récupérés")
    
    print("\n" + "="*60)
    print("📄 PHASE 3: Construction contexte MINIMAL")
    print("="*60 + "\n")
    
    # Sélection STRICTE des chunks
    final_chunks = []
    seen_nodes: Set[Tuple[str, str]] = set()
    
    # PRIMARY docs
    for doc_id in primary_ids:
        doc_chunks = [m for m, _ in chunk_hits if m.doc_id == doc_id]
        doc_chunks = sorted(doc_chunks, key=lambda m: m.score, reverse=True)
        
        doc_name = doc_chunks[0].doc_name if doc_chunks else "Unknown"
        print(f"\n📄 PRIMARY: {doc_name}")
        
        for chunk in doc_chunks[:chunks_per_primary]:
            node_key = (chunk.doc_id, chunk.node_id)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            
            chunk.text = smart_truncate_text(chunk.text, max_chunk_length)
            
            final_chunks.append(chunk)
            print(f"  ✅ {chunk.title} (~{len(chunk.text)} chars, score: {chunk.score:.3f})")
    
    # FALLBACK docs
    for doc_id in fallback_ids:
        doc_chunks = [m for m, _ in chunk_hits if m.doc_id == doc_id]
        doc_chunks = sorted(doc_chunks, key=lambda m: m.score, reverse=True)
        
        doc_name = doc_chunks[0].doc_name if doc_chunks else "Unknown"
        print(f"\n🔄 FALLBACK: {doc_name}")
        
        for chunk in doc_chunks[:chunks_per_fallback]:
            node_key = (chunk.doc_id, chunk.node_id)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            
            chunk.text = smart_truncate_text(chunk.text, max_chunk_length)
            
            final_chunks.append(chunk)
            print(f"  ✅ {chunk.title} (~{len(chunk.text)} chars, score: {chunk.score:.3f})")
    
    total_context_chars = sum(len(ch.text) for ch in final_chunks)
    print(f"\n✅ Total: {len(final_chunks)} chunks, ~{total_context_chars} caractères")
    
    # Construction contexte
    context_parts = []
    for ch in final_chunks:
        context_parts.append(f"[{ch.doc_name} | {ch.title}]\n{ch.text}")
    
    context_str = "\n\n" + "─"*40 + "\n\n".join(context_parts)
    
    print("\n" + "="*60)
    print("💬 PHASE 4: Génération réponse structurée")
    print("="*60 + "\n")
    
    system_prompt = """Tu es un assistant expert pour un centre de relation client.

STYLE DE RÉPONSE:

**Structure:**
- Commence par reformuler brièvement la situation: "D'après le contexte fourni..." ou "Selon la documentation..."
- Organise ta réponse clairement (titres ### si plusieurs parties, listes si étapes)
- Termine par les sources: **Source(s) :** [Document | Section](contexte)

**Mise en forme:**
- **Gras** pour les informations critiques (actions, dates, montants, documents)
- Listes numérotées (1. 2. 3.) pour les étapes/procédures
- Listes à puces (- ) pour options/conditions/éléments

**Ton:**
- Précis et actionnable
- Concis (pas de blabla inutile)
- Professionnel mais accessible
- Base-toi UNIQUEMENT sur le contexte fourni

La structure s'adapte naturellement à la question:
- Question simple → réponse courte en paragraphe
- Procédure → liste numérotée
- Plusieurs aspects → titres de sections
- Conditions → liste à puces avec "Si..."

Exemple naturel:
"D'après le contexte fourni, vous devez **faire X** dans les **délais Y**. La procédure est la suivante:
1. **Action A**
2. **Action B**

**Source(s) :**
- [Document | Section](contexte)."
"""

    user_prompt = f"""Question: {query}

Contexte:
{context_str}

Réponds en français de manière structurée et adaptée à la question."""

    chat = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=600
    )

    answer = chat.choices[0].message.content

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
        "total_context_chars": total_context_chars
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query v6 - Contexte minimal, réponses structurées")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chunks-per-primary", type=int, default=4, help="Chunks du doc principal (3-5 recommandé)")
    parser.add_argument("--chunks-per-fallback", type=int, default=0, help="Chunks du doc fallback (2-3 recommandé)")
    parser.add_argument("--max-chunk-length", type=int, default=500, help="Longueur max par chunk (400-600)")
    parser.add_argument("--no-hyde", action="store_true")
    parser.add_argument("--no-expansion", action="store_true")
    args = parser.parse_args()

    if args.query:
        res = answer_query_v6(
            args.query,
            chunks_per_primary=args.chunks_per_primary,
            chunks_per_fallback=args.chunks_per_fallback,
            max_chunk_length=args.max_chunk_length,
            use_hyde=not args.no_hyde,
            use_query_expansion=not args.no_expansion
        )
    
    print("\n" + "="*60)
    print("🎯 RÉPONSE")
    print("="*60)
    print(res["answer"])
    
    print("\n" + "="*60)
    print("📊 STATISTIQUES")
    print("="*60)
    print(f"  • Chunks utilisés: {res['num_chunks']}")
    print(f"  • Contexte total: ~{res['total_context_chars']} caractères")
    print(f"  • Variantes question: {len(res['query_variants'])}")
    
    print("\n" + "="*60)
    print("💭 RAISONNEMENT")
    print("="*60)
    print(res["selection_reasoning"])
    
    print("\n" + "="*60)
    print("📚 DOCUMENTS")
    print("="*60)
    print("\nPRIMARY:")
    for doc in res["primary_docs"]:
        print(f"  • {doc['doc_name']}")
    
    if res["fallback_docs"]:
        print("\nFALLBACK:")
        for doc in res["fallback_docs"]:
            print(f"  • {doc['doc_name']}")
