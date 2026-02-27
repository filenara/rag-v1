import os
import gc
import json
import logging
import torch
from tqdm import tqdm
from typing import List, Set

from src.utils import load_prompts, load_config
from document_parser import DocumentParser
from vision_processor import VisionProcessor
from semantic_splitter import STE100SemanticSplitter
from vector_indexer import VectorIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, filepath: str = "ingest_checkpoint.json"):
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

    def mark_as_done(self, filename: str):
        self.processed.add(filename)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed), f)

    def is_processed(self, filename: str) -> bool:
        return filename in self.processed

class PipelineOrchestrator:
    def __init__(self):
        self.cfg = load_config()
        self.collection_name = self.cfg.get("vector_db", {}).get("collection_name", "doc_v2_asset_store")
        self.bm25_path = self.cfg.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
        self.assets_dir = os.path.join("data", "assets")
        self.batch_size_limit = 32

        self.prompts = load_prompts()
        self.caption_prompt = self.prompts.get("caption_prompt", "Describe this technical image accurately.")

        self.checkpoint = CheckpointManager()
        self.parser = DocumentParser(assets_dir=self.assets_dir)
        self.vision = VisionProcessor()
        self.splitter = STE100SemanticSplitter()
        self.indexer = VectorIndexer(self.collection_name, self.bm25_path)

    def run_pipeline(self, pdf_paths: List[str]):
        logger.info(f"Ingestion baslatiliyor. Koleksiyon: {self.collection_name}")
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if self.checkpoint.is_processed(filename) or not os.path.exists(pdf_path):
                continue

            try:
                doc = self.parser.open_document(pdf_path)
            except Exception as e:
                logger.error(f"Dosya acilamadi {filename}: {e}")
                continue

            batch_chunks = []
            batch_metadatas = []

            for page_num, page in enumerate(tqdm(doc, desc=f"Isleniyor: {filename}")):
                text = self.parser.extract_text(page)
                page_dict = self.parser.extract_text_dict(page)
                visual_assets = self.parser.extract_visuals(page, filename, page_num + 1)
                
                visual_summary = ""
                has_visual = len(visual_assets) > 0
                image_path = visual_assets[0].path if has_visual else ""

                if has_visual:
                    safe_text = text[:2000] if text else "Metin bulunamadi."
                    formatted_prompt = self.caption_prompt.replace("{page_text}", safe_text)
                    pil_images = [asset.image for asset in visual_assets]
                    
                    captions = self.vision.generate_captions(pil_images, formatted_prompt)
                    
                    visual_summary += "\n[VISUAL/TABLE DETECTED]:\n"
                    for idx, caption in enumerate(captions):
                        visual_summary += f"- Analysis {idx+1}: {caption}\n"

                page_chunks = self.splitter.extract_semantic_chunks(page_dict)

                for chunk_text in page_chunks:
                    final_chunk = f"--- SOURCE: {filename} | PAGE {page_num + 1} ---\n"
                    if has_visual:
                        final_chunk += visual_summary
                    final_chunk += f"{chunk_text}"
                    
                    batch_chunks.append(final_chunk)
                    batch_metadatas.append({
                        "source": filename, 
                        "page": page_num + 1, 
                        "has_visual": str(has_visual),
                        "image_path": image_path 
                    })

                if len(batch_chunks) >= self.batch_size_limit:
                    self.indexer.save_batch(batch_chunks, batch_metadatas)
                    batch_chunks, batch_metadatas = [], []
                    self._clear_memory()

            doc.close()
            if batch_chunks:
                self.indexer.save_batch(batch_chunks, batch_metadatas)
            
            self.checkpoint.mark_as_done(filename)
            self._clear_memory()

        self.indexer.build_and_save_bm25()
        logger.info("Tum islemler tamamlandi.")

    def _clear_memory(self):
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