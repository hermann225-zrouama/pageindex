import argparse
import os
import json
import asyncio
from pathlib import Path
from pageindex.page_index_md import md_to_tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF or Markdown documents and generate structure')
    
    # Mode batch ou fichier unique
    parser.add_argument('--raw-dir', type=str, default='./raw',
                       help='Directory containing PDFs to process (default: ./raw)')
    parser.add_argument('--pdf_path', type=str, help='Path to a single PDF file (overrides raw-dir)')
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
    
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for results (default: ./data)')
    
    args = parser.parse_args()
    
    # ============================================================================
    # BATCH PDF PROCESSING (from raw directory)
    # ============================================================================
    
    async def process_single_pdf(pdf_path: str, output_dir: str):
        """Process a single PDF file."""
        from pageindex.page_index import extract_document_structure
        from pageindex.utils import structure_to_list, generate_node_summary, count_tokens
        from pageindex.utils import generate_doc_description, create_clean_structure_for_description
        
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'='*60}")
        print(f"📄 Processing: {pdf_name}")
        print(f"{'='*60}")
        
        try:
            tree_structure = await extract_document_structure(
                pdf_path=pdf_path,
                model=args.model,
                vision_model=args.vision_model,
                use_vision=(args.use_vision.lower() == 'yes'),
                toc_check_page_num=args.toc_check_pages,
                max_page_num_each_node=args.max_pages_per_node,
                max_token_num_each_node=args.max_tokens_per_node,
                logger=None
            )
            
            print('✅ Parsing done!')
            
            result = {
                'doc_name': pdf_name,
                'structure': tree_structure
            }
            
            # Génération de résumés
            if args.if_add_node_summary.lower() == 'yes':
                print("📝 Generating summaries...")
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
                clean_structure = create_clean_structure_for_description(result['structure'])
                doc_description = generate_doc_description(clean_structure, model=args.model)
                result['doc_description'] = doc_description
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            output = {
                "doc_name": pdf_name,
                "structure": result["structure"]
            }
            if "doc_description" in result:
                output["doc_description"] = result["doc_description"]
            
            output_file = os.path.join(output_dir, f"{pdf_name}_structure.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f'✅ Saved to: {output_file}')
            
            # Statistics
            all_nodes = structure_to_list(result['structure'])
            print(f"📊 Total nodes: {len(all_nodes)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {pdf_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    # Déterminer le mode d'opération
    if args.md_path:
        # ============================================================================
        # MARKDOWN PROCESSING
        # ============================================================================
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
        
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "doc_name": md_name,
            "structure": result["structure"]
        }
        if "doc_description" in result:
            output["doc_description"] = result["doc_description"]
        
        output_file = os.path.join(output_dir, f"{md_name}_structure.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f'\n✅ Tree structure saved to: {output_file}')
        
    elif args.pdf_path:
        # ============================================================================
        # SINGLE PDF PROCESSING
        # ============================================================================
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
        
        print(f"Text Model: {args.model}")
        print(f"Vision Model: {args.vision_model}")
        print(f"Use Vision: {args.use_vision}")
        
        asyncio.run(process_single_pdf(args.pdf_path, args.output_dir))
        
    else:
        # ============================================================================
        # BATCH PDF PROCESSING (default mode)
        # ============================================================================
        raw_dir = args.raw_dir
        
        if not os.path.isdir(raw_dir):
            print(f"❌ Raw directory not found: {raw_dir}")
            print(f"💡 Creating directory: {raw_dir}")
            os.makedirs(raw_dir, exist_ok=True)
            print(f"📝 Place your PDF files in {raw_dir}/ and run again")
            exit(0)
        
        # Find all PDFs
        pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {raw_dir}/")
            print(f"📝 Place your PDF files in {raw_dir}/ and run again")
            exit(0)
        
        print(f"\n{'='*60}")
        print(f"🚀 BATCH PROCESSING MODE")
        print(f"{'='*60}")
        print(f"📂 Input directory: {raw_dir}")
        print(f"📂 Output directory: {args.output_dir}")
        print(f"📄 Found {len(pdf_files)} PDF files")
        print(f"🤖 Text Model: {args.model}")
        print(f"👁️  Vision Model: {args.vision_model}")
        print(f"🔍 Use Vision: {args.use_vision}")
        print(f"{'='*60}\n")
        
        async def process_all():
            results = []
            for i, pdf_file in enumerate(pdf_files, 1):
                pdf_path = os.path.join(raw_dir, pdf_file)
                print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file}")
                success = await process_single_pdf(pdf_path, args.output_dir)
                results.append((pdf_file, success))
            
            return results
        
        results = asyncio.run(process_all())
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count
        
        print(f"✅ Successfully processed: {success_count}/{len(results)}")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count}")
            print("\nFailed files:")
            for name, success in results:
                if not success:
                    print(f"  - {name}")
        
        print(f"\n📁 Output files saved in: {args.output_dir}/")
        print(f"{'='*60}\n")
