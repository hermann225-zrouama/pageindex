# main.py
import argparse
import json
from datetime import datetime
from pathlib import Path
from vector_retriever import VectorRetriever
from retriever_manager import RetrieverManager
from rag_pipeline import RAGPipeline


def process_single_query(pipeline, query, args):
    """Traite une seule requête et retourne le résultat"""
    result = pipeline.answer(
        query=query,
        chunks_per_primary=args.chunks_per_primary,
        chunks_per_fallback=args.chunks_per_fallback,
        max_chunk_length=args.max_chunk_length,
        use_hyde=args.use_hyde,
        use_query_expansion=not args.no_expansion
    )
    return result


def display_single_result(result):
    """Affiche le résultat d'une seule requête"""
    print("\n" + "="*60)
    print("🎯 RÉPONSE")
    print("="*60)
    print(result["answer"])
    
    print("\n" + "="*60)
    print("📊 STATISTIQUES")
    print("="*60)
    print(f"  • Retriever: {result['retriever_used']}")
    print(f"  • Chunks: {result['num_chunks']}")
    print(f"  • Contexte: ~{result['total_context_chars']} chars")
    print(f"  • Variantes: {len(result['query_variants'])}")
    
    print("\n" + "="*60)
    print("📚 DOCUMENTS")
    print("="*60)
    print("\nPRIMARY:")
    for doc in result["primary_docs"]:
        print(f"  • {doc['doc_name']}")
    
    if result["fallback_docs"]:
        print("\nFALLBACK:")
        for doc in result["fallback_docs"]:
            print(f"  • {doc['doc_name']}")


def format_result_to_markdown(query_idx, query, result):
    """Formate un résultat en markdown"""
    md_content = f"## Question {query_idx + 1}\n\n"
    md_content += f"**Query:** {query}\n\n"
    md_content += f"### Réponse\n\n{result['answer']}\n\n"
    md_content += f"### Statistiques\n\n"
    md_content += f"- **Retriever:** {result['retriever_used']}\n"
    md_content += f"- **Chunks:** {result['num_chunks']}\n"
    md_content += f"- **Contexte:** ~{result['total_context_chars']} chars\n"
    md_content += f"- **Variantes:** {len(result['query_variants'])}\n\n"
    
    md_content += f"### Documents Utilisés\n\n"
    md_content += "**PRIMARY:**\n\n"
    for doc in result["primary_docs"]:
        md_content += f"- {doc['doc_name']}\n"
    
    if result["fallback_docs"]:
        md_content += "\n**FALLBACK:**\n\n"
        for doc in result["fallback_docs"]:
            md_content += f"- {doc['doc_name']}\n"
    
    md_content += "\n---\n\n"
    return md_content


def process_questionnaire(pipeline, questionnaire_path, args):
    """Traite un fichier questionnaire JSON et génère answers.md"""
    # Charger le fichier JSON
    with open(questionnaire_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    
    if not queries:
        print("❌ Aucune requête trouvée dans le fichier JSON")
        return
    
    print(f"📋 {len(queries)} requêtes à traiter...\n")
    
    # Préparer le contenu Markdown
    md_content = f"# Résultats du Questionnaire\n\n"
    md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Nombre de questions:** {len(queries)}\n\n"
    md_content += "---\n\n"
    
    # Traiter chaque requête
    for idx, query in enumerate(queries):
        print(f"[{idx + 1}/{len(queries)}] Traitement: {query[:80]}...")
        
        try:
            result = process_single_query(pipeline, query, args)
            md_content += format_result_to_markdown(idx, query, result)
            print(f"✅ Question {idx + 1} traitée avec succès")
        except Exception as e:
            print(f"❌ Erreur pour la question {idx + 1}: {str(e)}")
            md_content += f"## Question {idx + 1}\n\n"
            md_content += f"**Query:** {query}\n\n"
            md_content += f"### Erreur\n\n{str(e)}\n\n---\n\n"
    
    # Écrire le fichier answers.md
    output_path = Path("answers.md")
    output_path.write_text(md_content, encoding='utf-8')
    
    print(f"\n✅ Fichier généré: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Modulaire")
    
    # Mode questionnaire ou requête unique
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Requête unique à traiter")
    group.add_argument("--questionnaire", type=str, help="Chemin vers le fichier JSON de questions")
    
    # Paramètres du pipeline
    parser.add_argument("--chunks-per-primary", type=int, default=4)
    parser.add_argument("--chunks-per-fallback", type=int, default=0)
    parser.add_argument("--max-chunk-length", type=int, default=500)
    parser.add_argument("--use-hyde", action="store_true", default=False)
    parser.add_argument("--no-expansion", action="store_true")
    
    args = parser.parse_args()
    
    # 1. Crée le retriever manager
    manager = RetrieverManager()
    
    # 2. Enregistre le VectorRetriever
    vector_retriever = VectorRetriever(
        index_path="data/index_faiss.npz",
        data_dir="data"
    )
    manager.register(vector_retriever, set_as_default=True)
    
    # 3. Crée le pipeline RAG
    pipeline = RAGPipeline(
        manager, 
        data_dir="data", 
        mistral_api_key="rUqtUW7Az9sYVdRQI3Lo2Y6QWdIrVp4b"
    )
    
    # 4. Execute selon le mode
    if args.questionnaire:
        process_questionnaire(pipeline, args.questionnaire, args)
    else:
        result = process_single_query(pipeline, args.query, args)
        display_single_result(result)


if __name__ == "__main__":
    main()
