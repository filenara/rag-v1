import os
import gc
import json
import hashlib
import logging
import torch
from tqdm import tqdm
from typing import List, Set, Dict
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
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def mark_as_done(self, filename: str) -> None:
        self.processed.add(filename)
        try:
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            tmp_filepath = f"{self.filepath}.tmp"
            with open(tmp_filepath, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed), f, ensure_ascii=False, indent=4)
            os.replace(tmp_filepath, self.filepath)
        except Exception as e:
            logger.error("Checkpoint kaydetme hatasi: %s", e, exc_info=True)

    def is_processed(self, filename: str) -> bool:
        return filename in self.processed


class PipelineOrchestrator:
    def __init__(self):
        self.cfg = load_config()
        self.collection_name = self.cfg.get("vector_db", {}).get("collection_name", "doc_store")
        self.bm25_path = self.cfg.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
        self.assets_dir = os.path.join("data", "assets")
        
        self.ingestion_cfg = self.cfg.get("ingestion", {})
        self.batch_size_limit = self.ingestion_cfg.get("batch_size", 32)
        self.max_chunk_length = self.ingestion_cfg.get("max_chunk_length", 1500)
        
        self.max_image_size = self.ingestion_cfg.get("max_image_size", 1024)
        self.clear_every_n_images = self.ingestion_cfg.get("clear_every_n_images", 5)

        self.prompts = load_prompts()
        self.caption_prompt = self.prompts.get("caption_prompt", "Describe this technical image accurately.")

        self.checkpoint = CheckpointManager()
        self.vision_cache = VisionCacheManager()
        self.parser = DocumentParser(assets_dir=self.assets_dir)
        self.vision = VisionProcessor()
        self.splitter = STE100SemanticSplitter(max_chunk_length=self.max_chunk_length)
        self.indexer = VectorIndexer(self.collection_name, self.bm25_path)

    def run_pipeline(self, pdf_paths: List[str]) -> None:
        logger.info("Ingestion baslatiliyor. Koleksiyon: %s", self.collection_name)
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if self.checkpoint.is_processed(filename) or not os.path.exists(pdf_path):
                continue

            try:
                dl_doc = self.parser.parse_document(pdf_path)
            except Exception as e:
                logger.error("Dosya acilamadi %s: %s", filename, e, exc_info=True)
                continue

            logger.info("Gorsel analizler yapiliyor: %s", filename)
            visual_summaries_by_page = {}
            image_paths_by_page = {}
            
            pictures = [item for item, level in dl_doc.iterate_items() if item.label == "picture"]
            image_process_counter = 0
            
            for pic in tqdm(pictures, desc="VLM Islemleri (Resimler)"):
                try:
                    img = pic.get_image(dl_doc)
                    if not img:
                        continue
                        
                    img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                        
                    page_no = pic.prov[0].page_no if hasattr(pic, "prov") and pic.prov else 0
                    
                    img_hash = get_image_hash(img)
                    save_path = os.path.join(self.assets_dir, f"vis_{img_hash}.png")
                    if not os.path.exists(save_path):
                        img.save(save_path)
                        
                    if page_no not in image_paths_by_page:
                        image_paths_by_page[page_no] = []
                        visual_summaries_by_page[page_no] = "\n[VISUAL DETECTED]:\n"
                    
                    if save_path not in image_paths_by_page[page_no]:
                        image_paths_by_page[page_no].append(save_path)
                        
                    cached_caption = self.vision_cache.get(img_hash)
                    if cached_caption:
                        caption = cached_caption
                    else:
                        safe_text = "Docling tarafindan tespit edilen teknik gorsel."
                        formatted_prompt = self.caption_prompt.replace("{page_text}", safe_text)
                        
                        caption_list = self.vision.generate_captions([img], formatted_prompt)
                        caption = caption_list[0] if caption_list else "Gorsel analiz edilemedi."
                        
                        self.vision_cache.set(img_hash, caption)
                        self.vision_cache.save()
                        
                    visual_summaries_by_page[page_no] += f"- Analysis: {caption}\n"
                    del img
                    
                    image_process_counter += 1
                    if image_process_counter % self.clear_every_n_images == 0:
                        self._clear_memory()
                    
                except Exception as e:
                    logger.warning("Gorsel islenirken hata: %s", e, exc_info=True)

            self._clear_memory()

            logger.info("Metinler bolunuyor ve indeksleniyor: %s", filename)
            chunks_data = self.splitter.extract_semantic_chunks(dl_doc, filename)
            
            batch_chunks = []
            batch_metadatas = []
            
            for chunk_dict in chunks_data:
                chunk_text = chunk_dict.get("text", "")
                meta = chunk_dict.get("metadata", {})
                page_no = meta.get("page", 0)
                
                has_visual = page_no in visual_summaries_by_page
                v_summary = visual_summaries_by_page.get(page_no, "")
                i_paths = image_paths_by_page.get(page_no, [])
                
                final_chunk = f"--- SOURCE: {filename} | PAGE {page_no} ---\n"
                if has_visual:
                    final_chunk += v_summary
                final_chunk += chunk_text
                
                meta["has_visual"] = str(has_visual)
                meta["image_path"] = ",".join(i_paths) if i_paths else ""
                
                batch_chunks.append(final_chunk)
                batch_metadatas.append(meta)

                if len(batch_chunks) >= self.batch_size_limit:
                    self.indexer.save_batch(batch_chunks, batch_metadatas)
                    batch_chunks, batch_metadatas = [], []
                    self._clear_memory()

            if batch_chunks:
                self.indexer.save_batch(batch_chunks, batch_metadatas)
            
            self.checkpoint.mark_as_done(filename)
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