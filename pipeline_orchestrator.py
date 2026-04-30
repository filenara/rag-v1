import os
import gc
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set

import torch
from tqdm import tqdm
from PIL import Image

from src.utils import load_prompts, load_config
from document_parser import DocumentParser
from vision_processor import VisionProcessor
from semantic_splitter import STE100SemanticSplitter
from vector_indexer import VectorIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_image_hash(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()

def get_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256_hash.update(block)

    return sha256_hash.hexdigest()


def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_ingestion_signature(
    document_hash: str,
    collection_name: str,
    embedding_model: str,
    embedding_dimension: int,
    chunker: str,
    max_chunk_length: int,
) -> str:
    signature_payload = {
        "document_hash": document_hash,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "embedding_dimension": int(embedding_dimension),
        "chunker": chunker,
        "max_chunk_length": int(max_chunk_length),
    }

    serialized_payload = json.dumps(
        signature_payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

class VisionCacheManager:
    def __init__(self, filepath: str = "data/vision_cache.json"):
        self.filepath = filepath
        self.cache: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Cache okuma hatasi: %s", e)
                return {}
        return {}

    def save(self) -> None:
        try:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            tmp_filepath = f"{self.filepath}.tmp"
            with open(tmp_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            logger.error("Cache kaydetme hatasi: %s", e, exc_info=True)

    def get(self, img_hash: str) -> str:
        return self.cache.get(img_hash, "")

    def set(self, img_hash: str, caption: str) -> None:
        self.cache[img_hash] = caption


class CheckpointManager:
    def __init__(self, filepath: str = "data/ingest_checkpoint.json"):
        self.filepath = filepath
        self.processed: Set[str] = self._load()

    def _load(self) -> Set[str]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if isinstance(data, list):
                    return set(str(item) for item in data)

                return set()
            except Exception:
                return set()

        return set()

    def _make_key(self, filename: str, ingestion_signature: str) -> str:
        return f"{filename}::{ingestion_signature}"

    def save(self) -> None:
        try:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            tmp_filepath = f"{self.filepath}.tmp"
            with open(tmp_filepath, "w", encoding="utf-8") as file:
                json.dump(
                    sorted(self.processed),
                    file,
                    ensure_ascii=False,
                    indent=4,
                )

            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            logger.error("Checkpoint kaydetme hatasi: %s", e, exc_info=True)

    def mark_as_done(self, filename: str, ingestion_signature: str) -> None:
        checkpoint_key = self._make_key(filename, ingestion_signature)
        self.processed.add(checkpoint_key)
        self.save()

    def is_processed(self, filename: str, ingestion_signature: str) -> bool:
        checkpoint_key = self._make_key(filename, ingestion_signature)
        return checkpoint_key in self.processed

    def remove_by_filename(self, filename: str) -> None:
        prefix = f"{filename}::"
        self.processed = {
            item for item in self.processed
            if not item.startswith(prefix)
        }
        self.save()
    def __init__(self, filepath: str = "data/ingest_checkpoint.json"):
        self.filepath = filepath
        self.processed: Set[str] = self._load()

    def _load(self) -> Set[str]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if isinstance(data, list):
                    return set(str(item) for item in data)

                return set()
            except Exception:
                return set()

        return set()

    def _make_key(self, filename: str, document_hash: str) -> str:
        return f"{filename}::{document_hash}"

    def mark_as_done(self, filename: str, document_hash: str) -> None:
        checkpoint_key = self._make_key(filename, document_hash)
        self.processed.add(checkpoint_key)

        try:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            tmp_filepath = f"{self.filepath}.tmp"
            with open(tmp_filepath, "w", encoding="utf-8") as file:
                json.dump(
                    sorted(self.processed),
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            logger.error("Checkpoint kaydetme hatasi: %s", e, exc_info=True)

    def is_processed(self, filename: str, document_hash: str) -> bool:
        checkpoint_key = self._make_key(filename, document_hash)
        return checkpoint_key in self.processed
    
class ManifestManager:
    def __init__(self, filepath: str = "data/ingest_manifest.json"):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    data.setdefault("documents", {})
                    return data
            except Exception as e:
                logger.warning("Manifest okuma hatasi: %s", e)

        return {"documents": {}}

    def save(self) -> None:
        try:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            tmp_filepath = f"{self.filepath}.tmp"
            with open(tmp_filepath, "w", encoding="utf-8") as file:
                json.dump(
                    self.data,
                    file,
                    ensure_ascii=False,
                    indent=4,
                )

            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            logger.error("Manifest kaydetme hatasi: %s", e, exc_info=True)

    def get_document(self, filename: str) -> Dict:
        documents = self.data.get("documents", {})
        document_data = documents.get(filename, {})

        if isinstance(document_data, dict):
            return document_data

        return {}

    def update_document(
        self,
        filename: str,
        source_path: str,
        document_hash: str,
        ingestion_signature: str,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        chunker: str,
        max_chunk_length: int,
        chunk_count: int,
    ) -> None:
        self.data.setdefault("documents", {})
        self.data["documents"][filename] = {
            "filename": filename,
            "source_path": source_path,
            "document_hash": document_hash,
            "ingestion_signature": ingestion_signature,
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "embedding_dimension": int(embedding_dimension),
            "chunker": chunker,
            "max_chunk_length": int(max_chunk_length),
            "chunk_count": int(chunk_count),
            "ingested_at": get_utc_timestamp(),
        }
        self.save()

class PipelineOrchestrator:
    def __init__(self):
        self.cfg = load_config()
        self.collection_name = self.cfg.get("vector_db", {}).get("collection_name", "doc_store")
        self.bm25_path = self.cfg.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
        self.assets_dir = os.path.join("data", "assets")
        self.model_cfg = self.cfg.get("models", {})
        self.embedding_model_name = self.model_cfg.get(
            "embedding_model_name",
            "BAAI/bge-m3",
        )
        self.embedding_dimension = self.model_cfg.get("embedding_dimension", 1024)
        self.chunker_name = "docling_hierarchical"
        
        self.ingestion_cfg = self.cfg.get("ingestion", {})
        self.batch_size_limit = self.ingestion_cfg.get("batch_size", 32)
        self.max_chunk_length = self.ingestion_cfg.get("max_chunk_length", 1500)
        
        self.max_image_size = self.ingestion_cfg.get("max_image_size", 1024)
        self.clear_every_n_images = self.ingestion_cfg.get("clear_every_n_images", 5)

        self.prompts = load_prompts()
        self.caption_prompt = self.prompts.get("caption_prompt", "Describe this technical image accurately.")

        self.checkpoint = CheckpointManager()
        self.manifest = ManifestManager()
        self.vision_cache = VisionCacheManager()
        self.parser = DocumentParser(assets_dir=self.assets_dir)
        self.vision = VisionProcessor()
        self.splitter = STE100SemanticSplitter(max_chunk_length=self.max_chunk_length)
        self.indexer = VectorIndexer(self.collection_name, self.bm25_path)

    def run_pipeline(self, pdf_paths: List[str]) -> None:
        logger.info("Ingestion baslatiliyor. Koleksiyon: %s", self.collection_name)
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)

            if not os.path.exists(pdf_path):
                continue

            document_hash = get_file_hash(pdf_path)
            ingestion_signature = build_ingestion_signature(
                document_hash=document_hash,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model_name,
                embedding_dimension=int(self.embedding_dimension),
                chunker=self.chunker_name,
                max_chunk_length=int(self.max_chunk_length),
            )

            existing_manifest = self.manifest.get_document(filename)
            old_signature = existing_manifest.get("ingestion_signature", "")

            if old_signature == ingestion_signature:
                if self.checkpoint.is_processed(filename, ingestion_signature):
                    logger.info(
                        "Dokuman ayni ingestion signature ile daha once islenmis: %s",
                        filename,
                    )
                    continue

            if old_signature and old_signature != ingestion_signature:
                logger.info(
                    "Dokuman ingestion signature degisti. Eski chunklar siliniyor: %s",
                    filename,
                )
                self.indexer.delete_by_source(filename)
                self.checkpoint.remove_by_filename(filename)

            try:
                dl_doc = self.parser.parse_document(pdf_path)
            except Exception as e:
                logger.error("Dosya acilamadi %s: %s", filename, e, exc_info=True)
                continue

            logger.info("Gorsel analizler yapiliyor: %s", filename)
            images_info = []
            
            pictures = [item for item, level in dl_doc.iterate_items() if type(item).__name__ == "PictureItem"]
            image_process_counter = 0
            
            for pic in tqdm(pictures, desc="VLM Islemleri (Resimler)"):
                try:
                    img = pic.get_image(dl_doc)
                    if not img:
                        continue

                    if img.width < 80 or img.height < 80:
                        del img
                        continue
                        
                    img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                        
                    page_no = pic.prov[0].page_no if hasattr(pic, "prov") and pic.prov else 0
                    
                    img_hash = get_image_hash(img)
                    save_path = os.path.join(self.assets_dir, f"vis_{img_hash}.png")
                    if not os.path.exists(save_path):
                        img.save(save_path)
                        
                    cached_caption = self.vision_cache.get(img_hash)
                    if cached_caption:
                        caption = cached_caption
                    else:
                        safe_text = "A Technical Image."
                        formatted_prompt = self.caption_prompt.replace("{page_text}", safe_text)
                        
                        caption_list = self.vision.generate_captions([img], formatted_prompt)
                        caption = caption_list[0] if caption_list else "Gorsel analiz edilemedi."
                        
                        self.vision_cache.set(img_hash, caption)
                        self.vision_cache.save()
                    
                    anchor_text = ""
                    if hasattr(pic, "captions") and pic.captions:
                        cap_texts = []
                        for cap in pic.captions:
                            if hasattr(cap, "text") and cap.text:
                                cap_texts.append(cap.text.strip())
                        anchor_text = " ".join(cap_texts)
                        
                    images_info.append({
                        "page_no": page_no,
                        "anchor_text": anchor_text,
                        "summary": f"\n[VISUAL DETECTED]:\n- Analysis: {caption}\n",
                        "image_path": save_path,
                        "injected": False
                    })
                    
                    del img
                    
                    image_process_counter += 1
                    if image_process_counter % self.clear_every_n_images == 0:
                        self._clear_memory()
                    
                except Exception as e:
                    logger.warning("Gorsel islenirken hata: %s", e, exc_info=True)

            self._clear_memory()

            logger.info("Metinler bolunuyor ve indeksleniyor: %s", filename)
            chunks_data = self.splitter.extract_semantic_chunks(dl_doc, filename)
            
            for chunk_dict in chunks_data:
                meta = chunk_dict.get("metadata", {})
                meta["has_visual"] = "False"
                meta["image_path"] = ""
            
            for chunk_dict in chunks_data:
                chunk_text = chunk_dict.get("text", "")
                meta = chunk_dict.get("metadata", {})
                page_no = meta.get("page", 0)
                
                chunk_image_paths = []
                chunk_summaries = []
                
                for img_info in images_info:
                    if img_info["page_no"] == page_no and not img_info["injected"]:
                        if img_info["anchor_text"] and img_info["anchor_text"] in chunk_text:
                            chunk_summaries.append(img_info["summary"])
                            chunk_image_paths.append(img_info["image_path"])
                            img_info["injected"] = True
                            
                if chunk_summaries:
                    chunk_dict["text"] = chunk_text + "".join(chunk_summaries)
                    existing_paths = meta.get("image_path", "")
                    all_paths = existing_paths.split(",") if existing_paths else []
                    all_paths.extend(chunk_image_paths)
                    meta["image_path"] = ",".join(filter(None, all_paths))
                    meta["has_visual"] = "True"

            for img_info in images_info:
                if not img_info["injected"]:
                    for chunk_dict in chunks_data:
                        meta = chunk_dict.get("metadata", {})
                        if meta.get("page", 0) == img_info["page_no"]:
                            chunk_dict["text"] = chunk_dict.get("text", "") + img_info["summary"]
                            existing_paths = meta.get("image_path", "")
                            all_paths = existing_paths.split(",") if existing_paths else []
                            all_paths.append(img_info["image_path"])
                            meta["image_path"] = ",".join(filter(None, all_paths))
                            meta["has_visual"] = "True"
                            img_info["injected"] = True
                            break

            batch_chunks = []
            batch_metadatas = []

            for chunk_index, chunk_dict in enumerate(chunks_data):
                chunk_text = chunk_dict.get("text", "")
                meta = chunk_dict.get("metadata", {})
                page_no = meta.get("page", 0)

                final_chunk = f"--- SOURCE: {filename} | PAGE {page_no} ---\n{chunk_text}"
                chunk_hash = get_text_hash(
                    f"{ingestion_signature}\n{chunk_index}\n{final_chunk}"
                )

                meta["document_hash"] = document_hash
                meta["ingestion_signature"] = ingestion_signature
                meta["chunk_hash"] = chunk_hash
                meta["chunk_index"] = int(chunk_index)
                meta["embedding_model"] = self.embedding_model_name
                meta["embedding_dimension"] = int(self.embedding_dimension)
                meta["collection_name"] = self.collection_name
                meta["chunker"] = self.chunker_name
                meta["max_chunk_length"] = int(self.max_chunk_length)

                batch_chunks.append(final_chunk)
                batch_metadatas.append(meta)

                if len(batch_chunks) >= self.batch_size_limit:
                    self.indexer.save_batch(batch_chunks, batch_metadatas)
                    batch_chunks, batch_metadatas = [], []
                    self._clear_memory()

            if batch_chunks:
                self.indexer.save_batch(batch_chunks, batch_metadatas)
            
            self.manifest.update_document(
                filename=filename,
                source_path=pdf_path,
                document_hash=document_hash,
                ingestion_signature=ingestion_signature,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model_name,
                embedding_dimension=int(self.embedding_dimension),
                chunker=self.chunker_name,
                max_chunk_length=int(self.max_chunk_length),
                chunk_count=len(chunks_data),
            )

            self.checkpoint.mark_as_done(filename, ingestion_signature)
            self._clear_memory()

        self.indexer.build_and_save_bm25()
        logger.info("Tum islemler tamamlandi.")

    def _clear_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    data_folder = "data"
    if os.path.exists(data_folder):
        target_files = [
            os.path.join(data_folder, f) 
            for f in os.listdir(data_folder) 
            if f.lower().endswith(".pdf")
        ]
        if target_files:
            orchestrator = PipelineOrchestrator()
            orchestrator.run_pipeline(target_files)
        else:
            print("Klasorde PDF bulunamadi.")