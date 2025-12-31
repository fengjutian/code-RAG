import logging
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from coderag.config import IGNORE_PATHS, WATCHED_DIR
from coderag.embeddings import generate_embeddings
from coderag.index import add_to_index, save_index

logger = logging.getLogger(__name__)


def should_ignore_path(path: str) -> bool:
    """Check if the given path should be ignored based on the IGNORE_PATHS list.

    Args:
        path: File or directory path to check

    Returns:
        True if path should be ignored, False otherwise
    """
    try:
        for ignore_path in IGNORE_PATHS:
            if path.startswith(ignore_path):
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking ignore path for {path}: {str(e)}")
        return True  # Err on the side of caution


class CodeChangeHandler(FileSystemEventHandler):
    """Handle file system events for code changes."""

    def on_modified(self, event):
        """Handle file modification events."""
        try:
            if event.is_directory or should_ignore_path(event.src_path):
                return

            if event.src_path.endswith(".py"):
                logger.info(f"Detected change in file: {event.src_path}")

                # Read file content with error handling
                try:
                    with open(event.src_path, "r", encoding="utf-8") as f:
                        full_content = f.read()
                except (IOError, UnicodeDecodeError) as e:
                    logger.error(f"Error reading file {event.src_path}: {str(e)}")
                    return

                # Generate embeddings
                embeddings = generate_embeddings(full_content)
                if embeddings is not None and embeddings.size > 0:
                    filename = os.path.basename(event.src_path)
                    try:
                        add_to_index(embeddings, full_content, filename, event.src_path)
                        save_index()
                        logger.info(f"Updated FAISS index for file: {event.src_path}")
                    except Exception as e:
                        logger.error(
                            f"Error updating index for {event.src_path}: {str(e)}"
                        )
                else:
                    logger.warning(
                        f"Failed to generate embeddings for {event.src_path}"
                    )

        except Exception as e:
            logger.error(f"Unexpected error handling file event: {str(e)}")


def start_monitoring() -> None:
    """Start monitoring the directory for file changes."""
    try:
        if not os.path.exists(WATCHED_DIR):
            logger.error(f"Watched directory does not exist: {WATCHED_DIR}")
            return

        event_handler = CodeChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, path=WATCHED_DIR, recursive=True)
        observer.start()
        logger.info(f"Started monitoring {WATCHED_DIR} for changes...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping file monitoring...")
            observer.stop()
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")
            observer.stop()
            raise
        finally:
            observer.join()
            logger.info("File monitoring stopped")

    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise
