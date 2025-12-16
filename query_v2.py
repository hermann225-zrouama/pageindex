# query_v2.py (improved)
import os
import json
import numpy as np
import faiss
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from mistralai import Mistral

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
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


def load_doc_structure(doc_id: str) -> Dict[str, Any]:
    """Charge la structure complète d'un document."""
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
    """
    Calcule le DocScore classique.
    DocScore = (1 / sqrt(N+1)) * sum_n ChunkScore(n)
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


def get_doc_summary(doc_id: str, chunk_hits: List[Tuple[ChunkMeta, float]], top_n: int = 5) -> Dict[str, Any]:
    """
    Génère un résumé enrichi d'un document.
    ✅ Plus d'extraits (5 au lieu de 3)
    ✅ TOC complète (3 niveaux)
    """
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
    
    # ✅ TOC plus détaillée (3 niveaux au lieu de 2)
    toc_preview = ""
    if doc_structure and "structure" in doc_structure:
        toc_items = []
        
        def extract_titles(nodes, depth=0, max_depth=3):  # ✅ 3 niveaux
            if depth > max_depth:
                return
            for node in nodes[:8]:  # ✅ 8 items au lieu de 5
                title = node.get("title", "")
                if title:
                    toc_items.append("  " * depth + "- " + title)
                if node.get("nodes"):
                    extract_titles(node.get("nodes", []), depth + 1, max_depth)
        
        extract_titles(doc_structure["structure"])
        toc_preview = "\n".join(toc_items[:20])  # ✅ 20 lignes au lieu de 10
    
    # ✅ Plus d'extraits (5 au lieu de 2)
    top_excerpts = []
    for meta, score in doc_chunks:
        # Extraits plus longs (300 chars au lieu de 200)
        excerpt = meta.text[:300] + "..." if len(meta.text) > 300 else meta.text
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
        "max_score": max_score,  # ✅ Ajouté
        "num_relevant_chunks": num_relevant_chunks,  # ✅ Ajouté
        "toc_preview": toc_preview,
        "top_excerpts": top_excerpts
    }


def llm_select_documents(
    query: str, 
    doc_summaries: List[Dict[str, Any]], 
    client: Mistral, 
    max_docs: int = 3
) -> Tuple[List[str], str]:
    """
    ✅ Prompt amélioré avec plus de contexte et instructions claires.
    """
    docs_info = []
    for i, doc_sum in enumerate(doc_summaries):
        doc_info = f"""
{'='*60}
Document {i+1}: {doc_sum['doc_name']}
{'='*60}
ID: {doc_sum['doc_id']}

📊 Scores sémantiques:
  - Score moyen: {doc_sum['avg_score']:.3f}
  - Meilleur chunk: {doc_sum['max_score']:.3f}
  - Nombre de chunks pertinents: {doc_sum['num_relevant_chunks']}
"""
        
        if doc_sum.get('doc_description'):
            doc_info += f"""
📝 Description du document:
{doc_sum['doc_description']}
"""
        
        if doc_sum.get('toc_preview'):
            doc_info += f"""
📑 Table des matières:
{doc_sum['toc_preview']}
"""
        
        doc_info += f"""
🔍 Top {len(doc_sum['top_excerpts'])} extraits les plus pertinents:
"""
        for j, excerpt in enumerate(doc_sum['top_excerpts'], 1):
            doc_info += f"""
  Extrait {j} - [{excerpt['title']}] (score: {excerpt['score']:.3f})
  {excerpt['excerpt']}
"""
        
        docs_info.append(doc_info)
    
    docs_text = "\n".join(docs_info)
    
    system_prompt = """Tu es un expert en recherche documentaire et sélection de sources pertinentes.

Ta mission: identifier les documents les PLUS PERTINENTS pour répondre précisément à la question posée.

Critères de sélection (par ordre d'importance):
1. **Contenu direct**: Le document traite-t-il DIRECTEMENT du sujet demandé?
2. **Description**: La description du document mentionne-t-elle les concepts clés de la question?
3. **Table des matières**: Les titres de sections correspondent-ils aux informations recherchées?
4. **Extraits pertinents**: Les passages trouvés contiennent-ils des réponses concrètes?
5. **Scores sémantiques**: Les scores indiquent la similarité lexicale, mais ne suffisent pas seuls.

⚠️ ATTENTION:
- Un score sémantique élevé ne signifie pas forcément pertinence (peut être juste du vocabulaire similaire)
- Privilégie les documents qui répondent DIRECTEMENT à la question
- Évite les documents trop généraux ou hors-sujet même s'ils ont un bon score"""

    user_prompt = f"""Question posée: {query}

Documents candidats:
{docs_text}

Ta tâche:
1. Analyse chaque document en profondeur
2. Identifie les {max_docs} documents qui permettent de répondre LE MIEUX à la question
3. Vérifie que les extraits contiennent des informations CONCRÈTES en lien avec la question

Retourne un JSON avec:
{{
    "selected_docs": ["doc_id_1", "doc_id_2", ...],
    "reasoning": "Explication DÉTAILLÉE: pourquoi ces documents et pas les autres? Qu'ont-ils de spécifique par rapport à la question?"
}}

Réponds UNIQUEMENT avec le JSON."""

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    result_text = response.choices[0].message.content
    
    try:
        result = json.loads(result_text)
        selected_doc_ids = result.get("selected_docs", [])
        reasoning = result.get("reasoning", "")
        return selected_doc_ids, reasoning
    except json.JSONDecodeError:
        print(f"⚠️  LLM response parsing failed, falling back to top docs by score")
        return [doc_sum["doc_id"] for doc_sum in doc_summaries[:max_docs]], "Fallback to semantic scores"


def choose_context_from_selected_docs(
    selected_doc_ids: List[str], 
    chunk_hits: List[Tuple[ChunkMeta, float]], 
    top_k_per_doc: int = 5
) -> List[ChunkMeta]:
    """Récupère les meilleurs chunks sémantiques pour les documents sélectionnés."""
    selected_chunks: List[ChunkMeta] = []
    
    for doc_id in selected_doc_ids:
        doc_chunks = [m for m, _ in chunk_hits if m.doc_id == doc_id]
        doc_chunks = sorted(doc_chunks, key=lambda m: m.score, reverse=True)
        selected_chunks.extend(doc_chunks[:top_k_per_doc])
    
    return selected_chunks


def answer_query_v2(
    query: str, 
    top_k: int = 100,  # ✅ Augmenté de 50 à 100
    max_docs_to_consider: int = 8,  # ✅ Augmenté de 10 à 8 (qualité > quantité)
    top_docs: int = 3
) -> Dict[str, Any]:
    """
    Version 2 améliorée: Sélection LLM avec plus de contexte.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY non défini")

    client = Mistral(api_key=MISTRAL_API_KEY)

    print("🔍 Step 1: Semantic search for relevant chunks...")
    index, metas = load_index()
    q_vec = embed_query(query, client)

    D, I = index.search(q_vec, top_k)
    scores = D[0]
    idxs = I[0]

    chunk_hits: List[Tuple[ChunkMeta, float]] = []
    for score, idx in zip(scores, idxs):
        meta = metas[idx]
        meta.score = float(score)
        chunk_hits.append((meta, float(score)))

    print(f"✅ Found {len(chunk_hits)} relevant chunks")

    doc_ids = list(set(m.doc_id for m, _ in chunk_hits))
    print(f"📚 Spanning {len(doc_ids)} documents")

    doc_scores = compute_doc_scores(chunk_hits)
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_docs_to_consider]
    candidate_doc_ids = [doc_id for doc_id, _ in ranked_docs]

    print(f"\n🤖 Step 2: LLM document selection (analyzing {len(candidate_doc_ids)} candidates)...")
    
    doc_summaries = []
    for doc_id in candidate_doc_ids:
        summary = get_doc_summary(doc_id, chunk_hits, top_n=5)  # ✅ 5 extraits
        doc_summaries.append(summary)
        desc = "✅ with description" if summary.get("doc_description") else "⚠️  no description"
        print(f"  📄 {summary['doc_name']}: score={summary['avg_score']:.3f}, {summary['num_relevant_chunks']} chunks {desc}")

    selected_doc_ids, reasoning = llm_select_documents(query, doc_summaries, client, max_docs=top_docs)
    
    print(f"\n🎯 Selection reasoning:\n{reasoning}\n")
    print(f"✅ Selected {len(selected_doc_ids)} documents: {selected_doc_ids}")

    print(f"\n📄 Step 3: Retrieving semantic chunks from selected documents...")
    context_chunks = choose_context_from_selected_docs(selected_doc_ids, chunk_hits, top_k_per_doc=5)
    
    print(f"✅ Retrieved {len(context_chunks)} chunks for context")

    context_str_parts = []
    for ch in context_chunks:
        context_str_parts.append(
            f"[{ch.doc_name} | {ch.title} | node {ch.node_id} | score={ch.score:.3f}]\n{ch.text}"
        )
    context_str = "\n\n".join(context_str_parts)

    print(f"\n💬 Step 4: Generating answer...")
    system_prompt = (
        "Tu es un assistant expert qui répond aux questions en te basant UNIQUEMENT sur le contexte fourni.\n"
        "Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.\n"
        "Cite les sources (noms de documents et sections) quand tu réponds."
    )
    user_prompt = f"Question:\n{query}\n\nContexte:\n{context_str}\n\nRéponds en français de manière concise mais précise."

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
        "selection_reasoning": reasoning,  # ✅ Ajouté
        "selected_docs": [
            {
                "doc_id": doc_id,
                "doc_name": next((s["doc_name"] for s in doc_summaries if s["doc_id"] == doc_id), "Unknown"),
                "has_description": bool(next((s.get("doc_description") for s in doc_summaries if s["doc_id"] == doc_id), None)),
                "semantic_score": doc_scores.get(doc_id, 0.0),
                "num_chunks": next((s["num_relevant_chunks"] for s in doc_summaries if s["doc_id"] == doc_id), 0)
            }
            for doc_id in selected_doc_ids
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

    parser = argparse.ArgumentParser(description="Query with improved LLM document selection")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-docs-consider", type=int, default=8)
    parser.add_argument("--top-docs", type=int, default=3)
    args = parser.parse_args()

    res = answer_query_v2(
        args.query, 
        top_k=args.top_k, 
        max_docs_to_consider=args.max_docs_consider,
        top_docs=args.top_docs
    )
    
    print("\n" + "="*60)
    print("🎯 RÉPONSE")
    print("="*60)
    print(res["answer"])
    
    print("\n" + "="*60)
    print("💭 RAISONNEMENT DE SÉLECTION")
    print("="*60)
    print(res["selection_reasoning"])
    
    print("\n" + "="*60)
    print("📚 DOCUMENTS SÉLECTIONNÉS")
    print("="*60)
    for doc in res["selected_docs"]:
        desc = "✅" if doc['has_description'] else "⚠️"
        print(f"  {desc} {doc['doc_name']}")
        print(f"      ID: {doc['doc_id']}, Score: {doc['semantic_score']:.3f}, Chunks: {doc['num_chunks']}")
    
    print("\n" + "="*60)
    print("📄 CHUNKS UTILISÉS")
    print("="*60)
    for chunk in res["used_chunks"]:
        print(f"  • [{chunk['doc_name']}] {chunk['title']} (score: {chunk['score']:.3f})")
