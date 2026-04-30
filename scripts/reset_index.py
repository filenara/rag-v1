import argparse
import gc
import logging
import shutil
from pathlib import Path
from typing import Iterable

import chromadb

from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def safe_remove_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()
        logger.info("Removed file: %s", path)
    else:
        logger.info("File already missing: %s", path)


def safe_remove_dir(path: Path) -> None:
    if not path.exists():
        logger.info("Directory already missing: %s", path)
        return

    if not path.is_dir():
        raise RuntimeError(f"Expected directory but found non-directory: {path}")

    resolved = path.resolve()

    if str(resolved) in {"/", str(Path.home().resolve())}:
        raise RuntimeError(f"Refusing to delete unsafe path: {resolved}")

    shutil.rmtree(resolved)
    logger.info("Removed directory: %s", resolved)


def delete_chroma_collection(
    persist_path: Path,
    collection_name: str,
) -> None:
    if not persist_path.exists():
        logger.info("Chroma path does not exist yet: %s", persist_path)
        return

    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        client.delete_collection(collection_name)
        logger.info("Deleted Chroma collection: %s", collection_name)
    except Exception as exc:
        logger.warning(
            "Could not delete Chroma collection '%s': %s",
            collection_name,
            exc,
        )
    finally:
        try:
            del client
        except UnboundLocalError:
            pass
        gc.collect()


def remove_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            safe_remove_dir(path)
        elif path.is_file():
            safe_remove_file(path)
        else:
            logger.info("Path already missing: %s", path)


def reset_index(
    keep_vision_cache: bool = False,
    keep_assets: bool = False,
) -> None:
    cfg = load_config()

    vector_cfg = cfg.get("vector_db", {})
    persist_path = Path(vector_cfg.get("persist_path", "./chroma_db"))
    collection_name = vector_cfg.get("collection_name", "doc_store")
    bm25_cache_path = Path(
        vector_cfg.get("bm25_cache_path", "data/bm25_cache.pkl")
    )

    vision_cache_path = Path("data/vision_cache.json")
    checkpoint_path = Path("data/ingest_checkpoint.json")
    assets_path = Path("data/assets")

    logger.info("Resetting RAG index artifacts.")
    logger.info("Chroma persist path: %s", persist_path)
    logger.info("Chroma collection: %s", collection_name)

    delete_chroma_collection(
        persist_path=persist_path,
        collection_name=collection_name,
    )

    paths_to_remove = [
        persist_path,
        bm25_cache_path,
        checkpoint_path,
    ]

    if not keep_vision_cache:
        paths_to_remove.append(vision_cache_path)

    if not keep_assets:
        paths_to_remove.append(assets_path)

    remove_paths(paths_to_remove)

    assets_path.mkdir(parents=True, exist_ok=True)

    logger.info("Reset completed.")
    logger.info("PDF files were not removed.")
    logger.info("Run ingestion again before starting the API backend.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset ChromaDB, BM25, checkpoint, and generated assets."
    )

    parser.add_argument(
        "--keep-vision-cache",
        action="store_true",
        help="Do not delete data/vision_cache.json.",
    )

    parser.add_argument(
        "--keep-assets",
        action="store_true",
        help="Do not delete data/assets.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    reset_index(
        keep_vision_cache=args.keep_vision_cache,
        keep_assets=args.keep_assets,
    )