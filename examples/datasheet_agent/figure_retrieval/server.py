#!/usr/bin/env python3


import argparse
import json
import time
from pathlib import Path
from typing import Any

import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import base64
import os
import requests
from dotenv import load_dotenv
import traceback
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")


from typing import List, Optional, Dict, Any

VISION_SYSTEM_PROMPT = """
You are a helpful assistant that extracts information from images and converts them to Markdown.
"""


def _render_pdf_page_to_data_url(
    pdf_path: str,
    page_number: int,
    dpi: int = 180
) -> List[str]:
    """
    Render a single page (page_number, 1-based) to a PNG data URL.
    Returns a list with a single data URL, or empty if error.
    """
    if not pdf_path or not Path(pdf_path).exists() or fitz is None:
        return []

    try:
        doc = fitz.open(pdf_path)
        page_idx = int(page_number) - 1
        if page_idx < 0 or page_idx >= doc.page_count:
            doc.close()
            return []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        doc.close()
        return [data_url]
    except Exception:
        return []


def call_openrouter_vision_markdown(
    image_data_urls: List[str],
    query_hint: Optional[str],
    model: str = "google/gemini-2.5-pro",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> str:
    """
    Sends multiple base64 PNG data URLs to a vision-capable model and asks for Markdown extraction.
    Many OpenRouter multimodal models accept the OpenAI-style 'image_url' content.
    """
    if not image_data_urls:
        return ""

    messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT}]
    user_content: List[Dict[str, Any]] = []
    if query_hint:
        user_content.append({"type": "text", "text": f"Extract information for the following query: {query_hint}"})
        user_content.append({"type": "text", "text": "Please extract only the relevant parts but keep tables complete and describe the images in detail and include the name of the image in the description."})
    else:
        user_content.append({"type": "text", "text": "Extract the following PDF pages to Markdown (include tables)."})
    # Add images
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    # OpenRouter accepts OpenAI-format messages where content can be a list of parts
    messages.append({"role": "user", "content": user_content})

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]




class LocalDatasheetFigureRetriever:

    def __init__(self, data_dir: str): 
        self.data_dir = Path(data_dir) 
        self.encoder = SentenceTransformer("intfloat/e5-base-v2")
    
    def search(self, query: str, datasheet_id: str, k: int = 2) -> list[dict[str, Any]]:
        """
        Search the dense index for the given query and datasheet ID.
        """
        figuredense_index_file = self.data_dir / f"{datasheet_id}" / "figure_index.faiss"
        figure_metadata_file = self.data_dir / f"{datasheet_id}" / "figure_index.metadata.json"
        if not figuredense_index_file.exists() or not figure_metadata_file.exists():
            raise FileNotFoundError(f"Figure dense index or metadata file not found for datasheet {datasheet_id}")
        figure_dense_index = faiss.read_index(str(figuredense_index_file))
        with open(figure_metadata_file, "r") as f:
            figure_metadata = json.load(f)

        query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = figure_dense_index.search(query_vector, k)
        figures = [{"figure_description": figure_metadata[idx]["figure_description"], "page_number": figure_metadata[idx]["page_number"], "score": float(score)} for score, idx in zip(scores[0], indices[0], strict=False) if idx < len(figure_metadata)]

        image_urls = self.get_figure_data_urls(figures, datasheet_id)
        response_from_vision_llm = call_openrouter_vision_markdown(image_urls, query)
        return {'response': response_from_vision_llm}

    def get_figure_data_urls(self, figures: List[dict[str, Any]], datasheet_id: str) -> List[str]:
        """
        Get the data URLs for the given figures.
        """
        data_urls = []
        for figure in figures:
            data_url = _render_pdf_page_to_data_url(self.data_dir / datasheet_id / "datasheet.pdf", figure['page_number'], dpi=180) 
            if data_url:
                data_urls.append(data_url[0])
        return data_urls


app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        if not data or "query" not in data or "datasheet_id" not in data:
            return jsonify({"error": "Missing 'query' or 'datasheet_id' in request"}), 400

        query = data["query"]
        datasheet_id = data["datasheet_id"]
        print(f"Query: {query}, Datasheet ID: {datasheet_id}")
        response = retriever.search(query=query, datasheet_id=datasheet_id)
        print(f"Response: {response}")
        return jsonify({"query": query, "datasheet_id": datasheet_id, "method": "vision", "response": response['response']})
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="Datasheet agent retrieval server")
    parser.add_argument("--data_dir", default="/lustre/scratch/users/abhishek.maiti/rag/data/extracted_data_v2_with_PDFs", help="Directory containing datasheet data")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_time = time.time()
    # Initialize retriever
    global retriever
    try:
        retriever = LocalDatasheetFigureRetriever(args.data_dir)
        print(f"Datasheet agent retrieval server initialized")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Start server
    print(f"Took {time.time() - start_time} seconds to start the server")
    print(f"Starting datasheet agent retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()