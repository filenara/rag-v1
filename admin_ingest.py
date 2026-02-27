import os
import io
import gc
import json
import logging
import fitz
import torch
import uuid
import pickle
from rank_bm25 import BM25Okapi
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.utils import load_prompts, load_config

class CFG:
    MIN_WIDTH = 80
    MIN_HEIGHT = 80
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 32
    CAPTION_MAX_TOKENS = 512
    IMAGE_DPI_SCALE = 2.0  
    
    CHECKPOINT_FILE = "ingest_checkpoint.json"
    LOG_FILE = "ingest_status.log"
    ASSETS_DIR = os.path.join("data", "assets") 

os.makedirs(CFG.ASSETS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CFG.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class STE100SemanticSplitter:
    def __init__(self, normal_text_size_limit=11.0):
        self.normal_text_size_limit = normal_text_size_limit

    def extract_semantic_chunks(self, page):
        semantic_chunks = []
        current_heading = "Genel Baglam"
        current_content = ""

        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if b['type'] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        font_size = span["size"]
                        font_name = span["font"].lower()

                        is_heading = font_size > self.normal_text_size_limit or "bold" in font_name

                        if is_heading:
                            if current_content.strip() and len(current_content) > 20:
                                chunk_text = f"[{current_heading}]\n{current_content.strip()}"
                                semantic_chunks.append(chunk_text)
                            
                            current_heading = text
                            current_content = ""
                        else:
                            current_content += text + " "
                
                current_content += "\n\n"

        if current_content.strip() and len(current_content) > 20:
            chunk_text = f"[{current_heading}]\n{current_content.strip()}"
            semantic_chunks.append(chunk_text)

        return semantic_chunks

class CheckpointManager:
    def __init__(self, filepath=CFG.CHECKPOINT_FILE):
        self.filepath = filepath
        self.processed = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def mark_as_done(self, filename):
        self.processed.add(filename)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed), f)

    def is_processed(self, filename):
        return filename in self.processed

class CompositeVisualMiner:
    def __init__(self, min_width=CFG.MIN_WIDTH, min_height=CFG.MIN_HEIGHT, vector_spam_limit=600):
        self.min_width = min_width
        self.min_height = min_height
        self.vector_spam_limit = vector_spam_limit 

    def _boxes_intersect_or_close(self, box1, box2, margin=150): 
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)

    def _merge_boxes(self, boxes):
        if not boxes:
            return []
        merged = []
        while boxes:
            current = boxes.pop(0)
            has_overlap = True
            while has_overlap:
                has_overlap = False
                rest = []
                for other in boxes:
                    if self._boxes_intersect_or_close(current, other):
                        current = (
                            min(current[0], other[0]),
                            min(current[1], other[1]),
                            max(current[2], other[2]),
                            max(current[3], other[3])
                        )
                        has_overlap = True
                    else:
                        rest.append(other)
                boxes = rest
            merged.append(current)
        return merged

    def extract_visual_crops(self, page):
        all_visual_rects = []

        images = page.get_images(full=True)
        for img in images:
            rect = page.get_image_bbox(img)
            if rect.width > 50 and rect.height > 50:
                all_visual_rects.append(list(rect))
                
        paths = page.get_drawings()
        is_complex_table = len(paths) > self.vector_spam_limit
        if not is_complex_table:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))

        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if len(block[4]) < 200:
                all_visual_rects.append([block[0], block[1], block[2], block[3]])

        if not all_visual_rects:
            return []

        merged_rects = self._merge_boxes(all_visual_rects)
        
        final_crops = []
        for rect in merged_rects:
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            if w < self.min_width or h < self.min_height:
                continue
            
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                mat = fitz.Matrix(CFG.IMAGE_DPI_SCALE, CFG.IMAGE_DPI_SCALE)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                img_data = pix.tobytes("png")
                final_crops.append(Image.open(io.BytesIO(img_data)))
            except Exception: 
                pass
                
        return final_crops

def get_dynamic_batch_size():
    """Sistemin anlik VRAM durumunu kontrol ederek guvenli batch boyutunu hesaplar."""
    if not torch.cuda.is_available():
        return 1
    
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        
        if free_gb > 16.0:
            return 4
        elif free_gb > 10.0:
            return 3
        elif free_gb > 6.0:
            return 2
        else:
            return 1
    except Exception as e:
        logger.warning(f"VRAM kontrolu yapilamadi, batch_size 1 olarak ayarlandi. Hata: {e}")
        return 1

def generate_caption_safe(pil_image, model, processor, prompt_text):
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt_text}]}]
    
    try:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=CFG.CAPTION_MAX_TOKENS)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        description = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        return description
    except Exception as e:
        logger.error(f"Caption Hatasi: {e}")
        return "[Visual Description Failed]"

def save_page_asset(page, filename, page_num):
    try:
        mat = fitz.Matrix(CFG.IMAGE_DPI_SCALE, CFG.IMAGE_DPI_SCALE)
        pix = page.get_pixmap(matrix=mat)
        
        safe_filename = os.path.splitext(filename)[0].replace(" ", "_")
        asset_name = f"{safe_filename}_p{page_num}_{uuid.uuid4().hex[:6]}.png"
        save_path = os.path.join(CFG.ASSETS_DIR, asset_name)
        
        pix.save(save_path)
        return save_path
    except Exception as e:
        logger.error(f"Asset Kayit Hatasi ({filename}): {e}")
        return ""
    
def generate_caption_batch_safe(pil_images, model, processor, prompt_texts):
    messages_list = []
    for img, txt in zip(pil_images, prompt_texts):
        messages_list.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}
        ])
    
    try:
        text_inputs = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        inputs = processor(
            text=text_inputs, 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=CFG.CAPTION_MAX_TOKENS)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        descriptions = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return descriptions
    except Exception as e:
        logger.error(f"Toplu Caption Hatasi: {e}")
        raise e

def save_batch(col, chunks, metadatas, embedder):
    if not chunks: 
        return
    try:
        embeddings = embedder.encode(
            chunks, 
            show_progress_bar=False, 
            batch_size=32, 
            normalize_embeddings=True
        ).tolist()
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    except Exception as e:
        logger.error(f"Veritabani Yazma Hatasi: {e}")

def main_ingest(pdf_paths, collection_name="doc_default"):
    logger.info(f"Ingestion Baslatiliyor. Koleksiyon: {collection_name}")
    
    checkpoint = CheckpointManager()
    db_manager = DatabaseManager()
    col = db_manager.get_collection(collection_name)
    miner = CompositeVisualMiner()

    semantic_splitter = STE100SemanticSplitter(normal_text_size_limit=11.0)
    
    llm_manager = LLMManager()
    model, processor = llm_manager.load_vision_model()
    embedder = llm_manager.load_embedder()

    prompts_data = load_prompts()
    caption_prompt = prompts_data.get("caption_prompt", "Describe this technical image accurately.")
    
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        
        if checkpoint.is_processed(filename):
            logger.info(f"ATLANDI: {filename}")
            continue
            
        if not os.path.exists(pdf_path):
            continue

        try:
            doc = fitz.open(pdf_path)
        except Exception:
            continue

        logger.info(f"Isleniyor: {filename} ({doc.page_count} sayfa)")
        
        batch_chunks = []
        batch_metadatas = []
        
        for i, page in enumerate(tqdm(doc, desc=f"Sayfalar ({filename})")):
            try:
                text = page.get_text() or ""
                
                detected_images = miner.extract_visual_crops(page)
                visual_summary = ""
                asset_path = ""
                has_visual = False
                
                if detected_images:
                    has_visual = True
                    asset_path = save_page_asset(page, filename, i+1)
                    
                    visual_summary += "\n[VISUAL/TABLE DETECTED]:\n"
                    
                    safe_text = text[:2000] if text else "Metin bulunamadi."
                    formatted_prompt = caption_prompt.replace("{page_text}", safe_text)
                    
                    batch_size = get_dynamic_batch_size()
                    image_prompts = [formatted_prompt] * len(detected_images)
                    all_captions = []
                    
                    for b_idx in range(0, len(detected_images), batch_size):
                        img_batch = detected_images[b_idx:b_idx+batch_size]
                        prompt_batch = image_prompts[b_idx:b_idx+batch_size]
                        
                        try:
                            batch_captions = generate_caption_batch_safe(
                                img_batch, model, processor, prompt_batch
                            )
                            all_captions.extend(batch_captions)
                        except Exception as batch_err:
                            logger.warning(
                                f"Toplu islem basarisiz, tekli isleme geciliyor: {batch_err}"
                            )
                            for img, prp in zip(img_batch, prompt_batch):
                                try:
                                    cap = generate_caption_safe(img, model, processor, prp)
                                    all_captions.append(cap)
                                except Exception as single_err:
                                    logger.error(f"Tekli gorsel analiz hatasi: {single_err}")
                                    all_captions.append("[Visual Description Failed]")
                    
                    for idx, caption in enumerate(all_captions):
                        visual_summary += f"- Analysis {idx+1}: {caption}\n"
                
                page_chunks = semantic_splitter.extract_semantic_chunks(page)

                for chunk_text in page_chunks:
                    final_chunk = f"--- SOURCE: {filename} | PAGE {i+1} ---\n"
                    if has_visual:
                         final_chunk += visual_summary
                    final_chunk += f"{chunk_text}"
                    
                    batch_chunks.append(final_chunk)
                    
                    batch_metadatas.append({
                        "source": filename, 
                        "page": i+1, 
                        "has_visual": str(has_visual),
                        "image_path": asset_path 
                    })

                if len(batch_chunks) >= CFG.BATCH_SIZE:
                    save_batch(col, batch_chunks, batch_metadatas, embedder)
                    batch_chunks, batch_metadatas = [], []
                    gc.collect()

            except Exception as e:
                logger.error(f"Sayfa Hatasi: {e}")
        
        doc.close()
        if batch_chunks:
            save_batch(col, batch_chunks, batch_metadatas, embedder)
        
        checkpoint.mark_as_done(filename)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("BM25 Indeksi olusturuluyor ve diske kaydediliyor...")
    all_docs_data = col.get()
    
    if all_docs_data and all_docs_data['documents']:
        tokenized_corpus = [doc.lower().split(" ") for doc in all_docs_data['documents']]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_cache = {
            "bm25": bm25,
            "ids": all_docs_data["ids"],
            "documents": all_docs_data["documents"],
            "metadatas": all_docs_data["metadatas"]
        }
        
        cfg_data = load_config()
        cache_path = cfg_data.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, "wb") as f:
            pickle.dump(bm25_cache, f)
        logger.info(f"BM25 Cache basariyla kaydedildi: {cache_path}")
    else:
        logger.warning("Veritabani bos, BM25 indeksi olusturulamadi.")

    logger.info("Tum islemler tamamlandi.")

if __name__ == "__main__":
    cfg_data = load_config()
    target_collection = cfg_data.get("vector_db", {}).get("collection_name", "doc_v2_asset_store")
    data_folder = "data" 
    
    if os.path.exists(data_folder):
        pdf_files = [
            os.path.join(data_folder, f) 
            for f in os.listdir(data_folder) 
            if f.lower().endswith(".pdf")
        ]
        
        if pdf_files:
            main_ingest(pdf_files, target_collection)
        else:
            logger.warning(f"'{data_folder}' klasorunde islenecek PDF dosyasi bulunamadi.")
    else:
        logger.error(f"Hata: '{data_folder}' klasoru mevcut degil.")