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
                logger.warning(f"Cache okuma hatasi: {e}")
                return {}
        return {}

    def save(self) -> None:
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Cache kaydetme hatasi: {e}")

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
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed), f)
        except Exception as e:
            logger.error(f"Checkpoint kaydetme hatasi: {e}")

    def is_processed(self, filename: str) -> bool:
        return filename in self.processed


class PipelineOrchestrator:
    def __init__(self):
        self.cfg = load_config()
        self.collection_name = self.cfg.get("vector_db", {}).get("collection_name", "doc_store")
        self.bm25_path = self.cfg.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
        self.assets_dir = os.path.join("data", "assets")
        self.batch_size_limit = self.cfg.get("ingestion", {}).get("batch_size", 32)

        self.prompts = load_prompts()
        self.caption_prompt = self.prompts.get("caption_prompt", "Describe this technical image accurately.")

        self.checkpoint = CheckpointManager()
        self.vision_cache = VisionCacheManager()
        self.parser = DocumentParser(assets_dir=self.assets_dir)
        self.vision = VisionProcessor()
        self.splitter = STE100SemanticSplitter()
        self.indexer = VectorIndexer(self.collection_name, self.bm25_path)

    def run_pipeline(self, pdf_paths: List[str]) -> None:
        logger.info(f"Ingestion baslatiliyor. Koleksiyon: {self.collection_name}")
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if self.checkpoint.is_processed(filename) or not os.path.exists(pdf_path):
                continue

            try:
                doc = self.parser.open_document(pdf_path)
            except Exception as e:
                logger.error(f"Dosya acilamadi {filename}: {e}", exc_info=True)
                continue

            batch_chunks = []
            batch_metadatas = []
            pages_data = []

            for page_num, page in enumerate(tqdm(doc, desc=f"Okunuyor: {filename}")):
                text = self.parser.extract_text(page)
                page_dict = self.parser.extract_text_dict(page)
                visual_assets = self.parser.extract_visuals(page, filename, page_num + 1)
                
                pages_data.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "page_dict": page_dict,
                    "visual_assets": visual_assets,
                    "visual_summary": "",
                    "has_visual": len(visual_assets) > 0,
                    "image_path": visual_assets[0].path if len(visual_assets) > 0 else ""
                })
                
            doc.close()

            logger.info(f"Gorsel analizler yapiliyor: {filename}")
            for data in tqdm(pages_data, desc="VLM Islemleri"):
                if data["has_visual"]:
                    safe_text = data["text"][:2000] if data["text"] else "Metin bulunamadi."
                    formatted_prompt = self.caption_prompt.replace("{page_text}", safe_text)
                    
                    final_captions = []
                    images_to_process = []
                    hashes_to_process = []
                    
                    for asset in data["visual_assets"]:
                        img_hash = get_image_hash(asset.image)
                        cached_caption = self.vision_cache.get(img_hash)
                        
                        if cached_caption:
                            final_captions.append(cached_caption)
                        else:
                            images_to_process.append(asset.image)
                            hashes_to_process.append(img_hash)
                    
                    if images_to_process:
                        new_captions = self.vision.generate_captions(images_to_process, formatted_prompt)
                        for h, cap in zip(hashes_to_process, new_captions):
                            self.vision_cache.set(h, cap)
                            final_captions.append(cap)
                        self.vision_cache.save()
                        
                    summary = "\n[VISUAL/TABLE DETECTED]:\n"
                    for idx, caption in enumerate(final_captions):
                        summary += f"- Analysis {idx+1}: {caption}\n"
                    data["visual_summary"] = summary

            logger.info(f"Metinler bolunuyor ve indeksleniyor: {filename}")
            self.splitter.reset_buffer()
            
            total_pages = len(pages_data)
            
            for idx, data in enumerate(pages_data):
                # 2. ve 3. DEGISIKLIK: Sadece metni gonder ve son sayfa bilgisini ilet
                is_last = (idx == total_pages - 1)
                page_chunks = self.splitter.extract_semantic_chunks(data["text"], is_last_page=is_last)

                for chunk_text in page_chunks:
                    final_chunk = f"--- SOURCE: {filename} | PAGE {data['page_num']} ---\n"
                    if data["has_visual"]:
                        final_chunk += data["visual_summary"]
                    final_chunk += f"{chunk_text}"
                    
                    batch_chunks.append(final_chunk)
                    batch_metadatas.append({
                        "source": filename, 
                        "page": data["page_num"], 
                        "has_visual": str(data["has_visual"]),
                        "image_path": data["image_path"] 
                    })

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