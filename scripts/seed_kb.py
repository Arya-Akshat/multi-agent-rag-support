#!/usr/bin/env python3
"""
scripts/seed_kb.py — CLI utility to seed the Knowledge Base.

This script wipes the existing ChromaDB collection, loads the JSON
articles from the data directory, chunks them, and embeds them.
Used by the `make seed` command.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_logging.logger import get_logger
from retrieval.ingest import ingest_articles, load_articles_from_disk
from retrieval.vectorstore import reset_collection

logger = get_logger("seed_kb")


def main():
    parser = argparse.ArgumentParser(description="Seed the CloudDash Knowledge Base")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="knowledge_base/data/articles",
        help="Path to the directory containing JSON KB articles"
    )
    args = parser.parse_args()

    logger.info("Starting Knowledge Base seeding process...")

    # Step 1: Wipe existing DB to ensure a clean state
    logger.info("Resetting ChromaDB collection...")
    reset_collection()

    # Step 2: Load raw articles
    articles = load_articles_from_disk(args.data_dir)
    if not articles:
        logger.error(f"No articles found in {args.data_dir}. Seeding aborted.")
        sys.exit(1)

    # Step 3: Run the ingestion pipeline
    chunks_inserted = ingest_articles(articles)

    if chunks_inserted > 0:
        logger.info(f"✅ Seeding complete! {chunks_inserted} chunks inserted.")
        sys.exit(0)
    else:
        logger.error("❌ Seeding failed. No chunks were inserted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
