# index_docs.py
import os
import json
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from mistralai import Mistral

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
EMBED_MODEL = "mistral-embed"

DATA_DIR = "data"
INDEX_PATH = "data/index_faiss.npz"


@dataclass
class ChunkMeta:
    doc_id: str
    doc_name: str
    node_id: str
    title: str
    chunk_id: int
    text: str
    score: float = 0.0


def load_docs(data_dir: str) -> List[Dict[str, Any]]:
    """Charge tous les JSON de documents."""
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                    if "structure" in doc:
                        docs.append(doc)
                        print(f"✅ Loaded: {fname}")
                    else:
                        print(f"⚠️  Skipped {fname} (no 'structure' key)")
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {fname}")
    return docs


def estimate_tokens(text: str) -> int:
    """Estimation grossière du nb de tokens (1 token ≈ 4 chars en moyenne)."""
    return len(text) // 4


def smart_chunk_long_node(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """
    Découpe un nœud long en chunks sémantiques (par paragraphes/phrases).
    
    Args:
        text: Texte à découper
        max_tokens: Taille max d'un chunk en tokens
        overlap: Overlap entre chunks (en mots) pour conserver le contexte
    
    Returns:
        Liste de chunks
    """
    # Si le texte est assez court, le garder entier
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    
    # D'abord essayer de découper par paragraphes
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_tokens = estimate_tokens(para)
        
        # Si un seul paragraphe est trop long, le découper par phrases
        if para_tokens > max_tokens:
            # Sauver le chunk courant si non vide
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Découper le paragraphe long par phrases
            sentences = para.replace('!', '.').replace('?', '.').split('.')
            
            sent_chunk = []
            sent_size = 0
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                sent_tokens = estimate_tokens(sent)
                
                if sent_size + sent_tokens > max_tokens and sent_chunk:
                    chunks.append('. '.join(sent_chunk) + '.')
                    # Overlap : garder la dernière phrase
                    sent_chunk = [sent_chunk[-1]] if overlap > 0 else []
                    sent_size = estimate_tokens(sent_chunk[0]) if sent_chunk else 0
                
                sent_chunk.append(sent)
                sent_size += sent_tokens
            
            if sent_chunk:
                chunks.append('. '.join(sent_chunk) + '.')
        
        # Ajouter le paragraphe au chunk courant
        elif current_size + para_tokens > max_tokens:
            # Sauver le chunk courant
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            # Commencer un nouveau chunk avec overlap
            if overlap > 0 and current_chunk:
                current_chunk = [current_chunk[-1], para]
                current_size = estimate_tokens(current_chunk[0]) + para_tokens
            else:
                current_chunk = [para]
                current_size = para_tokens
        else:
            current_chunk.append(para)
            current_size += para_tokens
    
    # Ajouter le dernier chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]


def chunk_by_node(node: Dict[str, Any], max_tokens: int = 512) -> List[str]:
    """
    Crée des chunks à partir d'un nœud PageIndex.
    
    Stratégie:
    - Si le nœud est court (<= max_tokens), 1 chunk = 1 nœud entier
    - Si le nœud est long, découpage intelligent par paragraphes/phrases
    
    Args:
        node: Nœud de la structure PageIndex
        max_tokens: Taille max d'un chunk
    
    Returns:
        Liste de chunks de texte
    """
    text = node.get("text", "") or ""
    
    if not text.strip():
        return []
    
    # Découpage intelligent si trop long
    return smart_chunk_long_node(text, max_tokens=max_tokens, overlap=50)


def embed_texts(texts: List[str], client: Mistral, batch_size: int = 64) -> np.ndarray:
    """Génère les embeddings par batch."""
    vecs = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Embedding batch {batch_num}/{total_batches}...")
        
        res = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        batch_vecs = [d.embedding for d in res.data]
        vecs.extend(batch_vecs)
    
    return np.array(vecs, dtype="float32")


def build_index(max_tokens_per_chunk: int = 512):
    """
    Construit l'index FAISS avec chunking par nœud PageIndex.
    
    Args:
        max_tokens_per_chunk: Taille max d'un chunk (si nœud trop long, découpage intelligent)
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY non défini")

    client = Mistral(api_key=MISTRAL_API_KEY)
    
    print(f"📂 Loading documents from {DATA_DIR}...")
    docs = load_docs(DATA_DIR)
    print(f"📄 Found {len(docs)} documents\n")
    
    if not docs:
        raise RuntimeError(f"Aucun document trouvé dans {DATA_DIR}/")

    all_chunks: List[ChunkMeta] = []
    all_texts: List[str] = []

    for doc_idx, doc in enumerate(docs):
        doc_id = f"doc_{doc_idx}"
        doc_name = doc.get("doc_name", doc_id)
        structure = doc["structure"]
        
        print(f"📖 Processing: {doc_name}")
        
        # Aplatir la structure pour obtenir tous les nœuds
        flat_nodes = []
        
        def rec(node):
            flat_nodes.append(node)
            for c in node.get("nodes", []) or []:
                rec(c)
        
        for root in structure:
            rec(root)
        
        print(f"   → {len(flat_nodes)} nodes in structure")
        
        node_count = 0
        chunk_count = 0
        
        # ✅ Chunking par nœud (pas par mots arbitraires)
        for node in flat_nodes:
            node_id = node.get("node_id", "")
            title = node.get("title", "Untitled")
            
            # Créer des chunks à partir du nœud
            node_chunks = chunk_by_node(node, max_tokens=max_tokens_per_chunk)
            
            if not node_chunks:
                continue
            
            node_count += 1
            
            # Créer une metadata pour chaque chunk
            for chunk_idx, chunk_text in enumerate(node_chunks):
                meta = ChunkMeta(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    node_id=node_id,
                    title=title,
                    chunk_id=chunk_idx,
                    text=chunk_text,
                )
                all_chunks.append(meta)
                all_texts.append(chunk_text)
                chunk_count += 1
        
        print(f"   → {node_count} nodes with text")
        print(f"   → {chunk_count} chunks created")
        print()

    if not all_texts:
        raise RuntimeError("Aucun texte à indexer")

    print(f"🔢 Total: {len(all_texts)} chunks from {len(docs)} documents")
    print(f"📊 Average: {len(all_texts) / len(docs):.1f} chunks per document\n")

    print(f"🔢 Embedding {len(all_texts)} chunks with {EMBED_MODEL}...")
    vecs = embed_texts(all_texts, client)
    dim = vecs.shape[1]
    print(f"✅ Embeddings generated (dimension: {dim})\n")

    # Normalisation L2 pour cosine similarity via dot product
    print("🔧 Normalizing vectors for cosine similarity...")
    faiss.normalize_L2(vecs)

    # Index FAISS simple (Inner Product = cosine avec vecteurs normalisés)
    print("🏗️  Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"✅ Index built with {index.ntotal} vectors\n")

    # Sauvegarde
    print(f"💾 Saving index to {INDEX_PATH}...")
    np.savez_compressed(
        INDEX_PATH,
        vectors=vecs,
        metas=np.array([meta.__dict__ for meta in all_chunks], dtype=object),
    )
    
    print(f"\n{'='*60}")
    print(f"✅ INDEX SUCCESSFULLY CREATED")
    print(f"{'='*60}")
    print(f"Path: {INDEX_PATH}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Dimension: {dim}")
    print(f"Model: {EMBED_MODEL}")
    print(f"Chunking: By PageIndex nodes (max {max_tokens_per_chunk} tokens)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build FAISS index from PageIndex documents')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens per chunk (default: 512)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help=f'Data directory (default: {DATA_DIR})')
    
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir
    
    build_index(max_tokens_per_chunk=args.max_tokens)
