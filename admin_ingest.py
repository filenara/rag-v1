import os
import io
import gc
import json
import logging
import fitz  # PyMuPDF
import torch
import uuid
from PIL import Image
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qwen_vl_utils import process_vision_info

# --- KENDİ MODÜLLERİMİZ ---
from src.database import DatabaseManager
from src.llm_manager import LLMManager

# --- 1. AYARLAR VE LOGLAMA ---
class CFG:
    MIN_WIDTH = 80
    MIN_HEIGHT = 80
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 10
    CAPTION_MAX_TOKENS = 512
    # Checkpoint ve Log dosyaları
    CHECKPOINT_FILE = "ingest_checkpoint.json"
    LOG_FILE = "ingest_status.log"

# Loglama Kurulumu
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CFG.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. YARDIMCI SINIFLAR (CHECKPOINT & MINER) ---

class CheckpointManager:
    """Hangi dosyaların işlendiğini takip eder."""
    def __init__(self, filepath=CFG.CHECKPOINT_FILE):
        self.filepath = filepath
        self.processed = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    def mark_as_done(self, filename):
        self.processed.add(filename)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed), f)

    def is_processed(self, filename):
        return filename in self.processed

class CompositeVisualMiner:
    """
    Görselleri akıllıca ayıklar ve dinamik zoom uygular.
    """
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
        if not boxes: return []
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
        
        # Resimler
        images = page.get_images(full=True)
        for img in images:
            rect = page.get_image_bbox(img)
            if (rect.width > 50 and rect.height > 50):
                all_visual_rects.append(list(rect))
                
        # Çizimler / Tablolar
        paths = page.get_drawings()
        is_complex_table = len(paths) > self.vector_spam_limit
        if not is_complex_table:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))

        # Metin Blokları (Görsel gibi davrananlar)
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if len(block[4]) < 200: 
                all_visual_rects.append([block[0], block[1], block[2], block[3]])

        if not all_visual_rects: return []

        merged_rects = self._merge_boxes(all_visual_rects)
        final_crops = []
        
        for rect in merged_rects:
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            
            if w < self.min_width or h < self.min_height: continue
            
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                
                # --- YENİ: DİNAMİK ZOOM HESAPLAMA ---
                # Hedef: En kısa kenar en az 1000px olsun (Okunabilirlik için)
                target_min_pixel = 1000
                min_side = min(w, h)
                
                if min_side > 0:
                    calculated_zoom = target_min_pixel / min_side
                else:
                    calculated_zoom = 2.0
                
                # Sınırlandırma (Clamp): En az 1.0x, En çok 4.0x
                zoom_factor = max(1.0, min(calculated_zoom, 4.0))
                
                # Matrix oluştur ve kırp
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                
                img_data = pix.tobytes("png")
                final_crops.append(Image.open(io.BytesIO(img_data)))
            except: 
                pass
        return final_crops

# --- 3. GÜVENLİ CAPTION VE OOM YÖNETİMİ ---

def generate_caption_safe(pil_image, model, processor, attempt=1):
    """
    OOM (Out of Memory) hatasına karşı dirençli caption üreticisi.
    """
    prompt = """
    Role: Senior Technical Data Analyst & OCR Specialist.
    Task: Analyze this image for a search engine index. First, CLASSIFY the image type, then extract data accordingly.

    1. **IMAGE CLASSIFICATION:** Start by stating the type: [Graph], [Table], [Technical Drawing], or [Photograph].

    2. **IF GRAPH / CHART:**
       - **Axes:** Read the X and Y axis labels and units.
       - **Data:** Describe the trend (e.g., "Linear increase from 0 to 100"). Extract key data points (peaks, valleys, intersections).
       - **Legend:** List what each line/color represents.

    3. **IF TABLE:**
       - **Structure:** Transcribe the content into a strict Markdown table format.
       - **Headers:** Preserve column headers exactly.

    4. **IF TECHNICAL DRAWING / SCHEMATIC:**
       - **Components:** List all identifiable parts (e.g., "Resistor R1", "Valve V2").
       - **Connections:** Describe key connections (e.g., "Power supply connects to Main Board via J1").
       - **Labels:** OCR all part numbers, pinouts, and annotations.

    5. **IF PHOTOGRAPH:**
       - **Subject:** Describe the equipment/part shown.
       - **Condition:** Note any visible damage (burns, cracks, corrosion) or status (LED on/off).
       - **Text:** Read all labels, serial numbers, and warning stickers verbatim.

    Constraint: Output ONLY factual data. Do not use conversational filler.
    """
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image}, 
            {"type": "text", "text": prompt}
        ]}
    ]
    
    try:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=CFG.CAPTION_MAX_TOKENS)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        description = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        del inputs, generated_ids, image_inputs, video_inputs
        return description

    except torch.cuda.OutOfMemoryError:
        logger.warning(f" GPU Bellek Hatası (OOM) - Deneme {attempt}")
        torch.cuda.empty_cache()
        gc.collect()
        
        if attempt < 3:
            # Görseli %50 Küçült ve Tekrar Dene
            logger.info(" Görsel yeniden boyutlandırılıyor (%50) ve tekrar deneniyor...")
            w, h = pil_image.size
            resized_image = pil_image.resize((int(w * 0.5), int(h * 0.5)))
            return generate_caption_safe(resized_image, model, processor, attempt=attempt + 1)
        else:
            logger.error(" OOM: Görsel çok büyük, küçültme işe yaramadı. Atlanıyor.")
            return "[ERROR: Image too large for GPU memory]"
            
    except Exception as e:
        logger.error(f" Beklenmeyen Hata: {e}")
        return f"[ERROR: {str(e)}]"

# --- 4. DATA PIPELINE ---

def save_batch(col, chunks, metadatas, embedder):
    if not chunks: return
    try:
        embeddings = embedder.encode(chunks, show_progress_bar=False, batch_size=4).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    except Exception as e:
        logger.error(f"Veritabanı Yazma Hatası: {e}")

def main_ingest(pdf_paths, collection_name="doc_default"):
    logger.info(f" Ingestion Başlatılıyor. Hedef Koleksiyon: {collection_name}")
    
    # 1. Hazırlık
    checkpoint = CheckpointManager()
    db_manager = DatabaseManager()
    col = db_manager.get_collection(collection_name)
    miner = CompositeVisualMiner()
    
    # 2. Modelleri Yükle (LLMManager üzerinden)
    llm_manager = LLMManager()
    model, processor = llm_manager.load_vision_model()
    embedder = llm_manager.load_embedder()
    
    # 3. Dosya Döngüsü
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        
        if checkpoint.is_processed(filename):
            logger.info(f" ATLANDI (Zaten İşlendi): {filename}")
            continue
            
        if not os.path.exists(pdf_path):
            logger.error(f" Dosya Bulunamadı: {pdf_path}")
            continue

        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0: raise ValueError("Sayfa sayısı 0")
        except Exception as e:
            logger.error(f" BOZUK DOSYA: {filename} - {e}")
            continue

        logger.info(f" İşleniyor: {filename} ({doc.page_count} sayfa)")
        
        batch_chunks = []
        batch_metadatas = []
        
        for i, page in enumerate(tqdm(doc, desc=f"Sayfalar ({filename})")):
            try:
                # Metin
                text = page.get_text() or ""
                
                # Görsel (Dinamik Zoom ile)
                detected_images = miner.extract_visual_crops(page)
                visual_summary = ""
                has_visual = False
                
                if detected_images:
                    visual_summary += "\n[VISUAL CONTENT DETECTED]:\n"
                    for idx, img in enumerate(detected_images):
                        caption = generate_caption_safe(img, model, processor)
                        visual_summary += f"- Image {idx+1}: {caption}\n"
                        has_visual = True

                # Birleştirme
                full_content = f"--- SOURCE: {filename} | PAGE {i+1} ---\n"
                if has_visual: full_content += f"{visual_summary}\n"
                full_content += f"[TEXT]:\n{text}"

                # Chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP)
                chunks = splitter.split_text(full_content)

                for chunk in chunks:
                    batch_chunks.append(chunk)
                    batch_metadatas.append({
                        "source": filename, 
                        "page": i+1, 
                        "has_visual": str(has_visual)
                    })

                # Batch Save
                if len(batch_chunks) >= CFG.BATCH_SIZE:
                    save_batch(col, batch_chunks, batch_metadatas, embedder)
                    batch_chunks, batch_metadatas = [], []
                    gc.collect()

            except Exception as e:
                logger.error(f"Sayfa Hatası ({filename} - Sayfa {i+1}): {e}")
        
        doc.close()
        
        if batch_chunks:
            save_batch(col, batch_chunks, batch_metadatas, embedder)
        
        checkpoint.mark_as_done(filename)
        logger.info(f" TAMAMLANDI: {filename}")
        
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Tüm işlemler başarıyla tamamlandı.")

if __name__ == "__main__":
    # Test klasörü
    data_folder = "data" 
    if os.path.exists(data_folder):
        pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
        if pdf_files:
            main_ingest(pdf_files, "doc_kaggle_v1")
        else:
            print("Klasörde PDF bulunamadı.")
    else:
        # Manuel test
        main_ingest(["test.pdf"], "doc_kaggle_v1")