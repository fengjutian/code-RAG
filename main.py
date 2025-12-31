import logging
import os
import warnings

from coderag.config import WATCHED_DIR
from coderag.embeddings import generate_embeddings
from coderag.index import add_to_index, clear_index, save_index
from coderag.monitor import should_ignore_path, start_monitoring

# Configure comprehensive logging in the entrypoint only
handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    # Enable file logging only if environment allows it
    if os.getenv("CODERAG_ENABLE_FILE_LOGS", "1") == "1":
        handlers.append(logging.FileHandler("coderag.log", encoding="utf-8"))
except Exception:
    # Ignore file handler failures (e.g., read-only FS)
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
    force=True,
)

logger = logging.getLogger(__name__)

# Suppress transformers warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)


def full_reindex() -> int:
    """Perform a full reindex of the entire codebase.

    Returns:
        Number of files successfully processed
    """
    logger.info("Starting full reindexing of the codebase...")

    if not os.path.exists(WATCHED_DIR):
        logger.error(f"Watched directory does not exist: {WATCHED_DIR}")
        return 0

    files_processed = 0
    files_failed = 0

    try:
        for root, _, files in os.walk(WATCHED_DIR):
            if should_ignore_path(root):
                logger.debug(f"Ignoring directory: {root}")
                continue

            for file in files:
                filepath = os.path.join(root, file)
                if should_ignore_path(filepath):
                    logger.debug(f"Ignoring file: {filepath}")
                    continue

                if file.endswith(".py"):
                    logger.debug(f"Processing file: {filepath}")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            full_content = f.read()

                        if not full_content.strip():
                            logger.debug(f"Skipping empty file: {filepath}")
                            continue

                        embeddings = generate_embeddings(full_content)
                        if embeddings is not None:
                            add_to_index(embeddings, full_content, file, filepath)
                            files_processed += 1
                        else:
                            logger.warning(
                                f"Failed to generate embeddings for {filepath}"
                            )
                            files_failed += 1

                    except (IOError, UnicodeDecodeError) as e:
                        logger.error(f"Error reading file {filepath}: {str(e)}")
                        files_failed += 1
                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing file {filepath}: {str(e)}"
                        )
                        files_failed += 1

        save_index()
        logger.info(
            f"Full reindexing completed. {files_processed} files processed, "
            f"{files_failed} files failed"
        )
        return files_processed

    except Exception as e:
        logger.error(f"Critical error during reindexing: {str(e)}")
        return files_processed


def main() -> None:
    """Main entry point for the CodeRAG indexing system."""
    try:
        logger.info("Starting CodeRAG indexing system")

        # Completely clear the FAISS index and metadata
        logger.info("Clearing existing index...")
        clear_index()

        # Perform a full reindex of the codebase
        logger.info("Starting full reindex...")
        processed_files = full_reindex()

        if processed_files == 0:
            logger.warning("No files were processed during indexing")
        else:
            logger.info("Indexing complete. Starting file monitoring...")
            # Start monitoring the directory for changes
            start_monitoring()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully")
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
