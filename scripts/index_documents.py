"""
Script: Index documents into Pinecone vector store.

Usage:
    python -m scripts.index_documents [--data-dir data/]
"""

from __future__ import annotations

import argparse

from config.logging_config import setup_logging, get_logger
from config.settings import get_settings
from retrieval.indexer import DocumentIndexer


def main():
    parser = argparse.ArgumentParser(description="Index medical documents into Pinecone")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing PDF files")
    args = parser.parse_args()

    setup_logging(level="INFO", json_output=False)
    logger = get_logger("index_script")

    settings = get_settings()

    logger.info(f"Starting document indexing from '{args.data_dir}'")

    indexer = DocumentIndexer(data_dir=args.data_dir)
    num_chunks = indexer.run()

    logger.info(f"Indexing complete: {num_chunks} chunks indexed into '{settings.pinecone_index_name}'")


if __name__ == "__main__":
    main()
