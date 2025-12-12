import litellm
from litellm import completion, acompletion
import logging
import os
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import yaml
from pathlib import Path
from types import SimpleNamespace as config
from typing import Optional, List, Dict, Union, Tuple
import base64
from PIL import Image
import re

# Configuration des clés API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Configuration LiteLLM
litellm.drop_params = True
litellm.set_verbose = False

# ============================================================================
# TOKEN COUNTING - Multi-provider support (INCHANGÉ)
# ============================================================================

def count_tokens(text: str, model: str = None) -> int:
    """Count tokens for any model using the appropriate tokenizer."""
    if not text:
        return 0
    
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception as e:
        logging.warning(f"LiteLLM token counting failed for {model}: {e}")
        
        try:
            if "gpt" in model.lower() or "openai" in model.lower():
                import tiktoken
                enc = tiktoken.encoding_for_model(model.replace("openai/", ""))
                return len(enc.encode(text))
            
            elif "mistral" in model.lower():
                try:
                    from mistral_common.protocol.instruct.messages import UserMessage
                    from mistral_common.protocol.instruct.request import ChatCompletionRequest
                    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                    
                    tokenizer = MistralTokenizer.v3()
                    tokenized = tokenizer.encode_chat_completion(
                        ChatCompletionRequest(messages=[UserMessage(content=text)])
                    )
                    return len(tokenized.tokens)
                except ImportError:
                    return len(text) // 4
            
            elif "claude" in model.lower() or "anthropic" in model.lower():
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    return client.count_tokens(text)
                except:
                    return len(text) // 4
            
            else:
                return len(text) // 4
                
        except Exception as fallback_error:
            logging.error(f"All token counting methods failed: {fallback_error}")
            return len(text) // 4


# ============================================================================
# PDF PAGE TO IMAGE - Vision support
# ============================================================================

def pdf_page_to_pixmap(
    pdf_path: Union[str, BytesIO], 
    page_num: int, 
    zoom: float = 2.0,
    colorspace: str = "rgb"
) -> pymupdf.Pixmap:
    """
    Convert a PDF page to a PyMuPDF Pixmap [web:25].
    
    Args:
        pdf_path: Path to PDF file or BytesIO object
        page_num: Page number (0-indexed)
        zoom: Zoom factor for resolution (default 2.0 for 144 DPI)
        colorspace: "rgb" or "gray"
    
    Returns:
        PyMuPDF Pixmap object
    """
    try:
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError(f"Invalid pdf_path: {pdf_path}")
        
        if page_num >= doc.page_count or page_num < 0:
            raise ValueError(f"Page {page_num} out of range (0-{doc.page_count-1})")
        
        page = doc[page_num]
        
        # Create transformation matrix for zoom
        mat = pymupdf.Matrix(zoom, zoom)
        
        # Set colorspace
        cs = pymupdf.csRGB if colorspace == "rgb" else pymupdf.csGRAY
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, colorspace=cs)
        
        doc.close()
        return pix
        
    except Exception as e:
        logging.error(f"Error converting PDF page {page_num} to pixmap: {e}")
        raise


def pixmap_to_base64(pix: pymupdf.Pixmap, image_format: str = "png") -> str:
    """
    Convert PyMuPDF Pixmap to base64 string [web:38].
    
    Args:
        pix: PyMuPDF Pixmap object
        image_format: Output format ("png", "jpeg", "jpg")
    
    Returns:
        Base64 encoded string
    """
    try:
        # Get image data as bytes
        if image_format.lower() in ["jpg", "jpeg"]:
            img_data = pix.pil_tobytes(format="JPEG", optimize=True, dpi=(72, 72))
        else:
            img_data = pix.pil_tobytes(format="PNG")
        
        # Encode to base64
        base64_str = base64.b64encode(img_data).decode('utf-8')
        return base64_str
        
    except Exception as e:
        logging.error(f"Error converting pixmap to base64: {e}")
        raise


def pdf_page_to_base64(
    pdf_path: Union[str, BytesIO], 
    page_num: int, 
    zoom: float = 2.0,
    image_format: str = "png"
) -> str:
    """
    Convert a PDF page directly to base64 string [web:25][web:38].
    
    Args:
        pdf_path: Path to PDF file or BytesIO object
        page_num: Page number (0-indexed)
        zoom: Zoom factor for resolution (default 2.0 for 144 DPI)
        image_format: Output format ("png", "jpeg")
    
    Returns:
        Base64 encoded string
    """
    pix = pdf_page_to_pixmap(pdf_path, page_num, zoom)
    base64_str = pixmap_to_base64(pix, image_format)
    return base64_str


def save_pdf_page_as_image(
    pdf_path: Union[str, BytesIO], 
    page_num: int, 
    output_path: str,
    zoom: float = 2.0,
    image_format: str = "png"
):
    """
    Save a PDF page as an image file [web:25].
    
    Args:
        pdf_path: Path to PDF file or BytesIO object
        page_num: Page number (0-indexed)
        output_path: Output file path
        zoom: Zoom factor for resolution
        image_format: Output format ("png", "jpeg")
    """
    pix = pdf_page_to_pixmap(pdf_path, page_num, zoom)
    pix.save(output_path)
    logging.info(f"Saved page {page_num} to {output_path}")


# ============================================================================
# VISION LLM API CALLS - Multi-provider support
# ============================================================================

def Vision_LLM_API(
    model: str,
    prompt: str,
    image_input: Union[str, List[str]],
    api_key: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    temperature: float = 0,
    max_retries: int = 10,
    image_detail: str = "high"
) -> str:
    """
    Call any Vision LLM with image(s) using LiteLLM [web:20][web:22][web:30].
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "mistral/pixtral-large-latest", "anthropic/claude-3-5-sonnet")
        prompt: Text prompt
        image_input: Single base64 string, URL, or list of them
        api_key: Optional API key
        chat_history: Optional conversation history
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        image_detail: "high", "low", or "auto" (OpenAI specific)
    
    Returns:
        Response content as string
    """
    for i in range(max_retries):
        try:
            # Prepare images list
            if isinstance(image_input, str):
                images = [image_input]
            else:
                images = image_input
            
            # Build content array with text and images
            content = [{"type": "text", "text": prompt}]
            
            for img in images:
                # Detect if it's a URL or base64
                if img.startswith("http://") or img.startswith("https://"):
                    # URL format
                    image_dict = {
                        "type": "image_url",
                        "image_url": {"url": img, "detail": image_detail}
                    }
                else:
                    # Base64 format - add data URI prefix if not present
                    if not img.startswith("data:"):
                        img = f"data:image/png;base64,{img}"
                    
                    image_dict = {
                        "type": "image_url",
                        "image_url": {"url": img, "detail": image_detail}
                    }
                
                content.append(image_dict)
            
            # Prepare messages
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": content})
            else:
                messages = [{"role": "user", "content": content}]
            
            # Prepare kwargs
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if api_key:
                kwargs["api_key"] = api_key
            
            response = completion(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print('************* Retrying Vision LLM *************')
            logging.error(f"Vision LLM Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error(f'Max retries reached for vision prompt: {prompt[:100]}...')
                return "Error"


async def Vision_LLM_API_async(
    model: str,
    prompt: str,
    image_input: Union[str, List[str]],
    api_key: Optional[str] = None,
    temperature: float = 0,
    max_retries: int = 10,
    image_detail: str = "high"
) -> str:
    """
    Async call to any Vision LLM with image(s) [web:20][web:22][web:30].
    
    Args:
        model: Model identifier
        prompt: Text prompt
        image_input: Single base64 string, URL, or list of them
        api_key: Optional API key
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        image_detail: "high", "low", or "auto"
    
    Returns:
        Response content as string
    """
    for i in range(max_retries):
        try:
            # Prepare images
            if isinstance(image_input, str):
                images = [image_input]
            else:
                images = image_input
            
            # Build content
            content = [{"type": "text", "text": prompt}]
            
            for img in images:
                if img.startswith("http://") or img.startswith("https://"):
                    image_dict = {
                        "type": "image_url",
                        "image_url": {"url": img, "detail": image_detail}
                    }
                else:
                    if not img.startswith("data:"):
                        img = f"data:image/png;base64,{img}"
                    
                    image_dict = {
                        "type": "image_url",
                        "image_url": {"url": img, "detail": image_detail}
                    }
                
                content.append(image_dict)
            
            messages = [{"role": "user", "content": content}]
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if api_key:
                kwargs["api_key"] = api_key
            
            response = await acompletion(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print('************* Retrying Vision LLM Async *************')
            logging.error(f"Vision LLM Async Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error(f'Max retries reached for async vision prompt: {prompt[:100]}...')
                return "Error"


# ============================================================================
# PDF PAGE EXTRACTION WITH VISION - High-level functions
# ============================================================================

def extract_text_from_pdf_page_with_vision(
    pdf_path: Union[str, BytesIO],
    page_num: int,
    model: str = "gpt-4o",
    prompt: Optional[str] = None,
    zoom: float = 2.0,
    api_key: Optional[str] = None
) -> str:
    """
    Extract text/information from a PDF page using Vision LLM [web:20][web:30].
    
    Args:
        pdf_path: Path to PDF file or BytesIO object
        page_num: Page number (0-indexed)
        model: Vision model to use
        prompt: Custom prompt (default: extract all text)
        zoom: Zoom factor for image quality
        api_key: Optional API key
    
    Returns:
        Extracted text or information
    """
    if prompt is None:
        prompt = """Extract all text from this image, maintaining the original structure and formatting. 
        Include all headings, paragraphs, lists, tables, and any other text content."""
    
    # Convert page to base64
    base64_image = pdf_page_to_base64(pdf_path, page_num, zoom=zoom, image_format="png")
    
    # Call vision LLM
    response = Vision_LLM_API(
        model=model,
        prompt=prompt,
        image_input=base64_image,
        api_key=api_key
    )
    
    return response


def LLM_API_with_finish_reason(model, prompt, chat_history=None, temperature=0.0):
    """
    Appel LLM qui retourne aussi le finish_reason (pour détecter les réponses tronquées).
    
    Returns:
        tuple: (response_text, finish_reason)
        - finish_reason peut être: 'finished', 'length', 'stop', etc.
    """
    try:
        messages = []
        
        # Ajouter l'historique si fourni
        if chat_history:
            messages.extend(chat_history)
        
        # Ajouter le prompt actuel
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Appel LiteLLM
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        # Extraire le contenu et le finish_reason
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # Normaliser finish_reason
        # LiteLLM peut retourner: 'stop', 'length', 'tool_calls', 'content_filter'
        # On normalise en 'finished' ou autre
        if finish_reason == 'stop':
            finish_reason = 'finished'
        
        return content, finish_reason
        
    except Exception as e:
        print(f"Error in LLM_API_with_finish_reason: {e}")
        raise


async def LLM_API_with_finish_reason_async(model, prompt, chat_history=None, temperature=0.0):
    """
    Version async de LLM_API_with_finish_reason.
    
    Returns:
        tuple: (response_text, finish_reason)
    """
    try:
        messages = []
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Appel async LiteLLM
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == 'stop':
            finish_reason = 'finished'
        
        return content, finish_reason
        
    except Exception as e:
        print(f"Error in LLM_API_with_finish_reason_async: {e}")
        raise



async def extract_text_from_pdf_page_with_vision_async(
    pdf_path: Union[str, BytesIO],
    page_num: int,
    model: str = "gpt-4o",
    prompt: Optional[str] = None,
    zoom: float = 2.0,
    api_key: Optional[str] = None
) -> str:
    """
    Async version of PDF page extraction with Vision LLM [web:20][web:30].
    """
    if prompt is None:
        prompt = """Extract all text from this image, maintaining the original structure and formatting. 
        Include all headings, paragraphs, lists, tables, and any other text content."""
    
    base64_image = pdf_page_to_base64(pdf_path, page_num, zoom=zoom, image_format="png")
    
    response = await Vision_LLM_API_async(
        model=model,
        prompt=prompt,
        image_input=base64_image,
        api_key=api_key
    )
    
    return response


def extract_text_from_pdf_pages_with_vision(
    pdf_path: Union[str, BytesIO],
    start_page: int,
    end_page: int,
    model: str = "gpt-4o",
    prompt: Optional[str] = None,
    zoom: float = 2.0,
    api_key: Optional[str] = None,
    batch_size: int = 1
) -> str:
    """
    Extract text from multiple PDF pages using Vision LLM.
    
    Args:
        pdf_path: Path to PDF file or BytesIO object
        start_page: Start page (1-indexed)
        end_page: End page (1-indexed, inclusive)
        model: Vision model to use
        prompt: Custom prompt
        zoom: Zoom factor
        api_key: Optional API key
        batch_size: Number of images per request (if model supports multiple images)
    
    Returns:
        Combined extracted text
    """
    if prompt is None:
        prompt = """Extract all text from this image/these images, maintaining the original structure and formatting. 
        Include all headings, paragraphs, lists, tables, and any other text content."""
    
    all_text = []
    
    # Process in batches
    for page_batch_start in range(start_page - 1, end_page, batch_size):
        page_batch_end = min(page_batch_start + batch_size, end_page)
        
        # Convert pages to base64
        base64_images = []
        for page_num in range(page_batch_start, page_batch_end):
            base64_image = pdf_page_to_base64(pdf_path, page_num, zoom=zoom, image_format="png")
            base64_images.append(base64_image)
        
        # Call vision LLM
        if len(base64_images) == 1:
            response = Vision_LLM_API(
                model=model,
                prompt=prompt,
                image_input=base64_images[0],
                api_key=api_key
            )
        else:
            # Multiple images in one request
            response = Vision_LLM_API(
                model=model,
                prompt=prompt,
                image_input=base64_images,
                api_key=api_key
            )
        
        all_text.append(response)
    
    return "\n\n".join(all_text)


async def extract_text_from_pdf_pages_with_vision_async(
    pdf_path: Union[str, BytesIO],
    start_page: int,
    end_page: int,
    model: str = "gpt-4o",
    prompt: Optional[str] = None,
    zoom: float = 2.0,
    api_key: Optional[str] = None
) -> str:
    """
    Async batch extraction of text from multiple PDF pages using Vision LLM.
    """
    if prompt is None:
        prompt = """Extract all text from this image, maintaining the original structure and formatting. 
        Include all headings, paragraphs, lists, tables, and any other text content."""
    
    # Create tasks for all pages
    tasks = []
    for page_num in range(start_page - 1, end_page):
        task = extract_text_from_pdf_page_with_vision_async(
            pdf_path=pdf_path,
            page_num=page_num,
            model=model,
            prompt=prompt,
            zoom=zoom,
            api_key=api_key
        )
        tasks.append(task)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return "\n\n".join(results)


# ============================================================================
# STANDARD LLM API CALLS - Multi-provider support (POUR COMPATIBILITÉ)
# ============================================================================

def LLM_API(model: str, prompt: str, api_key: Optional[str] = None, 
            chat_history: Optional[List[Dict]] = None, temperature: float = 0,
            max_retries: int = 10) -> str:
    """Call any LLM using LiteLLM."""
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            
            response = completion(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error(f'Max retries reached for prompt: {prompt[:100]}...')
                return "Error"


async def LLM_API_async(model: str, prompt: str, api_key: Optional[str] = None,
                        temperature: float = 0, max_retries: int = 10) -> str:
    """Async call to any LLM using LiteLLM."""
    messages = [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature}
            if api_key:
                kwargs["api_key"] = api_key
            
            response = await acompletion(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error(f'Max retries reached for prompt: {prompt[:100]}...')
                return "Error"


# Backward compatibility wrappers
def ChatGPT_API(model, prompt, api_key=OPENAI_API_KEY, chat_history=None):
    if "/" not in model:
        model = f"openai/{model}" if not model.startswith("gpt") else model
    return LLM_API(model, prompt, api_key, chat_history)


async def ChatGPT_API_async(model, prompt, api_key=OPENAI_API_KEY):
    if "/" not in model:
        model = f"openai/{model}" if not model.startswith("gpt") else model
    return await LLM_API_async(model, prompt, api_key)


# ============================================================================
# TOUTES LES AUTRES FONCTIONS UTILITAIRES (get_json_content, extract_json, etc.)
# [Copiez toutes les fonctions utilitaires de votre code original ici]
# ============================================================================


# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================

def get_json_content(response: str) -> str:
    """
    Extract JSON content from markdown code blocks.
    """
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content


def extract_json(content: str) -> dict:
    """
    Extract and parse JSON from string content.
    """
    try:
        # First, try to extract JSON enclosed within ``````
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}


# ============================================================================
# TREE/NODE STRUCTURE UTILITIES
# ============================================================================

def write_node_id(data, node_id: int = 0) -> int:
    """
    Recursively assign node IDs to a tree structure.
    """
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


def get_nodes(structure) -> list:
    """
    Get all nodes from a tree structure (flattened).
    """
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    return []


def structure_to_list(structure) -> list:
    """
    Convert tree structure to flat list of nodes.
    """
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def get_leaf_nodes(structure) -> list:
    """
    Get only the leaf nodes (nodes without children).
    """
    if isinstance(structure, dict):
        if not structure.get('nodes'):
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []


def is_leaf_node(data, node_id: str) -> bool:
    """
    Check if a node with given node_id is a leaf node.
    """
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False


def get_last_node(structure):
    """
    Get the last node from a structure.
    """
    if isinstance(structure, list):
        return structure[-1]
    return structure


def list_to_tree(data: list) -> list:
    """
    Convert a flat list with 'structure' field to a tree hierarchy.
    """
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]


# ============================================================================
# PDF EXTRACTION UTILITIES (Text-based)
# ============================================================================

def extract_text_from_pdf(pdf_path: Union[str, BytesIO]) -> str:
    """
    Extract all text from a PDF using PyPDF2.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_pdf_title(pdf_path: Union[str, BytesIO]) -> str:
    """
    Get PDF title from metadata.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title


def get_text_of_pages(pdf_path: Union[str, BytesIO], start_page: int, end_page: int, tag: bool = True) -> str:
    """
    Get text from a range of pages with optional page tags.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text: str) -> int:
    """
    Extract the first page number from tagged text.
    """
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text: str) -> int:
    """
    Extract the last page number from tagged text.
    """
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename: str, replacement: str = '-') -> str:
    """
    Sanitize filename for Linux/Unix systems.
    """
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)


def get_pdf_name(pdf_path: Union[str, BytesIO]) -> str:
    """
    Extract PDF name from path or metadata.
    """
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


def get_text_of_pdf_pages(pdf_pages: list, start_page: int, end_page: int) -> str:
    """
    Get text from pre-processed pdf_pages list.
    """
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages: list, start_page: int, end_page: int) -> str:
    """
    Get text from pre-processed pdf_pages list with page labels.
    """
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text


def get_number_of_pages(pdf_path: Union[str, BytesIO]) -> int:
    """
    Get total number of pages in PDF.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num


def add_preface_if_needed(data: list) -> list:
    """
    Add a preface node if the first page is not page 1.
    """
    if not isinstance(data, list) or not data:
        return data

    if data[0].get('physical_index') is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


# ============================================================================
# STRUCTURE POST-PROCESSING UTILITIES
# ============================================================================

def post_processing(structure: list, end_physical_index: int):
    """
    Post-process structure to add start_index and end_index, then convert to tree.
    """
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index'] - 1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    else:
        # Remove appear_start
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure


def clean_structure_post(data):
    """
    Remove page_number, start_index, end_index from structure.
    """
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data


def remove_fields(data, fields: list = ['text']):
    """
    Remove specified fields from data structure.
    """
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def remove_structure_text(data):
    """
    Remove 'text' field from structure (alias for remove_fields).
    """
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def convert_physical_index_to_int(data):
    """
    Convert physical_index from string tags to integers.
    """
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data: list) -> list:
    """
    Convert 'page' field from string to int.
    """
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def reorder_dict(data: dict, key_order: list) -> dict:
    """
    Reorder dictionary keys according to specified order.
    """
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order: list = None):
    """
    Format structure by reordering keys and cleaning empty nodes.
    """
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


# ============================================================================
# NODE TEXT MANAGEMENT
# ============================================================================

def add_node_text(node, pdf_pages: list):
    """
    Add text content to nodes from pre-processed pdf_pages.
    """
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages: list):
    """
    Add text content with page labels to nodes.
    """
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


# ============================================================================
# LLM-BASED PROCESSING FUNCTIONS
# ============================================================================

async def generate_node_summary(node: dict, model: str = None) -> str:
    """
    Generate a description/summary for a node using LLM.
    """
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}
    
    Directly return the description, do not include any other text.
    """
    response = await LLM_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model: str = None):
    """
    Generate summaries for all nodes in structure.
    """
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model: str = None) -> str:
    """
    Generate a one-sentence description for the entire document.
    """
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = LLM_API(model, prompt)
    return response


# ============================================================================
# DISPLAY/DEBUG UTILITIES
# ============================================================================

def print_toc(tree: list, indent: int = 0):
    """
    Print table of contents in a readable format.
    """
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)


def print_json(data, max_len: int = 40, indent: int = 2):
    """
    Print JSON with truncated long strings for readability.
    """
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def check_token_limit(structure, model: str = 'gpt-4o', limit: int = 110000):
    """
    Check if any node exceeds token limit.
    """
    list_nodes = structure_to_list(structure)
    for node in list_nodes:
        num_tokens = count_tokens(node.get('text', ''), model=model)
        if num_tokens > limit:
            print(f"Node ID: {node.get('node_id')} has {num_tokens} tokens")
            print("Start Index:", node.get('start_index'))
            print("End Index:", node.get('end_index'))
            print("Title:", node.get('title'))
            print("\n")


# ============================================================================
# JSON LOGGER CLASS
# ============================================================================

class JsonLogger:
    """
    Logger that saves all log entries to a JSON file.
    """
    def __init__(self, file_path: Union[str, BytesIO]):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level: str, message, **kwargs):
        """Log a message at specified level."""
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        """Log info level message."""
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        """Log error level message."""
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        """Log debug level message."""
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        """Log exception."""
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self) -> str:
        """Get full filepath for log file."""
        return os.path.join("logs", self.filename)


# ============================================================================
# CONFIG LOADER CLASS
# ============================================================================

class ConfigLoader:
    """
    Load configuration from YAML with validation and merging.
    """
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path) -> dict:
        """Load YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict: dict):
        """Validate that user config keys exist in default config."""
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        
        Args:
            user_opt: User configuration as dict, SimpleNamespace, or None
        
        Returns:
            SimpleNamespace config object
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)


# ============================================================================
# UPDATED get_page_tokens with Vision support
# ============================================================================

def get_page_tokens(
    pdf_path: Union[str, BytesIO], 
    model: str = "gpt-4o", 
    pdf_parser: str = "PyPDF2",
    use_vision: bool = False,
    zoom: float = 2.0
) -> list:
    """
    Get page text and token counts.
    
    Args:
        pdf_path: Path to PDF or BytesIO
        model: Model name for tokenization
        pdf_parser: "PyPDF2" or "PyMuPDF"
        use_vision: If True, use Vision LLM for extraction
        zoom: Zoom factor if using vision extraction
    
    Returns:
        List of tuples (page_text, token_count)
    """
    if use_vision:
        # Use Vision LLM extraction
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError(f"Invalid pdf_path: {pdf_path}")
        
        page_list = []
        for page_num in range(doc.page_count):
            # Extract using Vision LLM
            page_text = extract_text_from_pdf_page_with_vision(
                pdf_path=pdf_path,
                page_num=page_num,
                model=model,
                zoom=zoom
            )
            token_length = count_tokens(page_text, model=model)
            page_list.append((page_text, token_length))
        
        doc.close()
        return page_list
    
    else:
        # Use traditional text extraction
        if pdf_parser == "PyPDF2":
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            page_list = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                token_length = count_tokens(page_text, model=model)
                page_list.append((page_text, token_length))
            return page_list
        
        elif pdf_parser == "PyMuPDF":
            if isinstance(pdf_path, BytesIO):
                pdf_stream = pdf_path
                doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
            elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
                doc = pymupdf.open(pdf_path)
            page_list = []
            for page in doc:
                page_text = page.get_text()
                token_length = count_tokens(page_text, model=model)
                page_list.append((page_text, token_length))
            return page_list
        else:
            raise ValueError(f"Unsupported PDF parser: {pdf_parser}")
