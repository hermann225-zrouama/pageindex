import argparse
import os
import json
import asyncio
from pageindex.page_index_md import md_to_tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default='mistral/mistral-large-latest', 
                       help='Text model to use (default: mistral/mistral-large-latest)')
    
    parser.add_argument('--vision-model', type=str, default='mistral/pixtral-large-latest',
                       help='Vision model for PDF extraction (default: mistral/pixtral-large-latest)')
    parser.add_argument('--use-vision', type=str, default='yes',
                       choices=['yes', 'no'],
                       help='Use Vision LLM for PDF extraction (default: yes)')
    parser.add_argument('--vision-zoom', type=float, default=2.0,
                       help='Zoom factor for Vision LLM (1.0=72dpi, 2.0=144dpi, 3.0=216dpi)')

    parser.add_argument('--toc-check-pages', type=int, default=20, 
                       help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=50,
                       help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=100000,
                       help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default='yes',
                       help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='no',
                       help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default='no',
                       help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default='yes',
                       help='Whether to add text to the node')
                      
    parser.add_argument('--if-thinning', type=str, default='no',
                       help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                       help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                       help='Token threshold for generating summaries')
    
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validation
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    # ============================================================================
    # PDF PROCESSING
    # ============================================================================
    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        print(f"Processing PDF: {args.pdf_path}")
        print(f"Text Model: {args.model}")
        print(f"Vision Model: {args.vision_model}")
        print(f"Use Vision: {args.use_vision}")
        
        # Définir la fonction async principale
        async def process_pdf():
            from pageindex.page_index import extract_document_structure
            
            print("\n🚀 Starting PDF extraction...")
            
            tree_structure = await extract_document_structure(
                pdf_path=args.pdf_path,
                model=args.model,
                vision_model=args.vision_model,
                use_vision=(args.use_vision.lower() == 'yes'),
                toc_check_page_num=args.toc_check_pages,
                max_page_num_each_node=args.max_pages_per_node,
                max_token_num_each_node=args.max_tokens_per_node,
                logger=None
            )
            
            print('✅ Parsing done!')
            
            # Construire le résultat final
            pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            result = {
                'doc_name': pdf_name,
                'structure': tree_structure
            }
            
            # Génération de résumés
            if args.if_add_node_summary.lower() == 'yes':
                print("📝 Generating summaries...")
                from pageindex.utils import structure_to_list, generate_node_summary, count_tokens
                
                all_nodes = structure_to_list(result['structure'])
                
                for node in all_nodes:
                    text = node.get('text', '')
                    if text and len(text.strip()) > 0:
                        num_tokens = count_tokens(text, model=args.model)
                        
                        if num_tokens < args.summary_token_threshold:
                            if not node.get('nodes'):
                                node['summary'] = text
                            else:
                                node['prefix_summary'] = text
                        else:
                            summary = await generate_node_summary(node, model=args.model)
                            if not node.get('nodes'):
                                node['summary'] = summary
                            else:
                                node['prefix_summary'] = summary
                
                print(f"✅ Generated summaries for {len(all_nodes)} nodes")
            
            # Ajouter description du document
            if args.if_add_doc_description.lower() == 'yes':
                print("📄 Generating document description...")
                from pageindex.utils import generate_doc_description, create_clean_structure_for_description
                
                clean_structure = create_clean_structure_for_description(result['structure'])
                doc_description = generate_doc_description(clean_structure, model=args.model)
                result['doc_description'] = doc_description
            
            return result
        
        # Exécuter la fonction async
        result = asyncio.run(process_pdf())
        
        # Save results
        print('💾 Saving to file...')
        output_dir = args.output_dir
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f'\n✅ Tree structure saved to: {output_file}')
        
        # Print statistics
        from pageindex.utils import structure_to_list
        all_nodes = structure_to_list(result['structure'])
        print(f"📊 Total nodes extracted: {len(all_nodes)}")
        
        # ✅ CORRECTION: Retirer max_depth
        print("\n📑 Table of Contents Preview:")
        from pageindex.utils import print_toc
        try:
            print_toc(result['structure'])
        except Exception as e:
            print(f"Could not print TOC: {e}")
            # Afficher manuellement
            for i, node in enumerate(result['structure'][:5]):
                title = node.get('title', 'Untitled')
                node_id = node.get('node_id', 'N/A')
                print(f"  {node_id}. {title}")
                if i == 4 and len(result['structure']) > 5:
                    print(f"  ... and {len(result['structure']) - 5} more")
    
    # ============================================================================
    # MARKDOWN PROCESSING
    # ============================================================================
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        print(f"Processing markdown: {args.md_path}")
        print(f"Model: {args.model}")
        
        print("\n🚀 Starting markdown extraction...")
        result = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=args.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=args.model,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text,
            if_add_node_id=args.if_add_node_id
        ))
        
        print('✅ Parsing done!')
        print('💾 Saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = args.output_dir
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f'\n✅ Tree structure saved to: {output_file}')
        
        # Print statistics
        if isinstance(result, dict) and 'structure' in result:
            structure = result['structure']
        else:
            structure = result
        
        from pageindex.utils import structure_to_list
        all_nodes = structure_to_list(structure)
        print(f"📊 Total nodes extracted: {len(all_nodes)}")
        
        # ✅ CORRECTION: Retirer max_depth
        print("\n📑 Table of Contents Preview:")
        from pageindex.utils import print_toc
        try:
            print_toc(structure)
        except Exception as e:
            print(f"Could not print TOC: {e}")
            # Afficher manuellement
            for i, node in enumerate(structure[:5] if isinstance(structure, list) else []):
                title = node.get('title', 'Untitled')
                node_id = node.get('node_id', 'N/A')
                print(f"  {node_id}. {title}")
                if i == 4 and len(structure) > 5:
                    print(f"  ... and {len(structure) - 5} more")
