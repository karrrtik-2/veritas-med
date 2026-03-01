"""
Document indexing entrypoint — uses the new retrieval pipeline.

Usage:
    python store_index.py
"""

from config.logging_config import setup_logging, get_logger
from config.settings import get_settings
from retrieval.indexer import DocumentIndexer


def main() -> None:
    setup_logging(level="INFO", json_output=False)
    logger = get_logger("store_index")
    settings = get_settings()

    logger.info(f"Indexing documents from 'data/' into '{settings.pinecone_index_name}'")
    indexer = DocumentIndexer(data_dir="data")
    num_chunks = indexer.run()
    logger.info(f"Done — {num_chunks} chunks indexed.")


if __name__ == "__main__":
    main()