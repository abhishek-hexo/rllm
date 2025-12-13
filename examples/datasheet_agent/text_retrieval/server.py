#!/usr/bin/env python3

"""
Datasheet text-chunk retrieval server.

Dense retrieval using E5 embeddings + FAISS over per-datasheet indices.

Each datasheet folder is expected to contain:
  - text_index.faiss
  - text_index.metadata.json   (array; element i corresponds to vector i)

Usage:
    python server.py --data_dir ./datasheet_data/prebuilt_indices --port 8000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


class LocalDatasheetTextRetriever:
    """Dense-only retrieval over per-datasheet FAISS indices."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.encoder = SentenceTransformer("intfloat/e5-base-v2")
        # Simple in-memory cache: datasheet_id -> (faiss.Index, metadata_list)
        self._cache: dict[str, tuple[Any, list[dict[str, Any]]]] = {}

    def _load_for_datasheet(self, datasheet_id: str) -> tuple[Any, list[dict[str, Any]]]:
        if datasheet_id in self._cache:
            return self._cache[datasheet_id]

        ds_dir = self.data_dir / str(datasheet_id)
        index_path = ds_dir / "text_index.faiss"
        meta_path = ds_dir / "text_index.metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing text index for datasheet_id={datasheet_id}. "
                f"Expected {index_path} and {meta_path}"
            )

        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            raise ValueError(f"Invalid metadata format in {meta_path} (expected JSON array)")

        self._cache[datasheet_id] = (index, metadata)
        return index, metadata

    def search(self, query: str, datasheet_id: str, k: int = 10) -> list[dict[str, Any]]:
        index, metadata = self._load_for_datasheet(datasheet_id)

        query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = index.search(query_vector, k)

        results = [{"text": metadata[idx]["text"], "score": float(score)} for score, idx in zip(scores[0], indices[0], strict=False) if idx < len(metadata)]

        return {'results': results}


# Flask app
app = Flask(__name__)
retriever: Optional[LocalDatasheetTextRetriever] = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "index_type": "datasheet_text_dense",
            "data_dir": str(retriever.data_dir) if retriever else None,
        }
    )


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        if not data or "query" not in data or "datasheet_id" not in data:
            return jsonify({"error": "Missing 'query' or 'datasheet_id' in request"}), 400

        query = data["query"]
        datasheet_id = data["datasheet_id"]

        results = retriever.search(query=query, datasheet_id=datasheet_id)
        return jsonify(
            {
                "query": query,
                "datasheet_id": datasheet_id,
                "method": "dense_text",
                "results": results['results'],
                "num_results": len(results['results']),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Datasheet text retrieval server")
    parser.add_argument(
        "--data_dir",
        default="/lustre/scratch/users/abhishek.maiti/rag/data/extracted_data_v2_with_PDFs",
        help="Directory containing per-datasheet folders with text_index.faiss + text_index.metadata.json",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_time = time.time()
    global retriever
    try:
        retriever = LocalDatasheetTextRetriever(args.data_dir)
        print("Datasheet text retrieval server initialized")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    print(f"Took {time.time() - start_time} seconds to start the server")
    print(f"Starting datasheet text retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()