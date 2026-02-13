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
    IMAGE_DPI_SCALE = 2.0  # 2x Zoom (Yazıların okunabilirliği için)
    
    # Dosya Yolları
    CHECKPOINT_FILE = "ingest_checkpoint.json"
    LOG_FILE = "ingest_status.log"
    ASSETS_DIR = os.path.join("data", "assets") # Görseller buraya kaydedilecek

# Asset klasörünü oluştur
os.makedirs(CFG.ASSETS_DIR, exist_ok=True)

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

# --- 2. YARDIMCI SINIFLAR ---

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
    """Görselleri akıllıca ayıklar."""
    def __init__(self, min_width=CFG.MIN_WIDTH, min_height=CFG.MIN_HEIGHT, vector_spam_limit=600):
        self.min_width = min_width
        self.min_height = min_height
        self.vector_spam_limit = vector_spam_limit 

    def extract_visual_crops(self, page):
        """Sayfadaki önemli görsel alanları (resim/tablo) tespit eder."""
        # Basitlik için sadece resim varlığına bakıyoruz, 
        # çünkü rag_engine artık sayfanın TAMAMINI asset olarak kullanıyor.
        images = page.get_images(full=True)
        if images: return True
        
        # Çizim/Tablo kontrolü
        paths = page.get_drawings()
        if len(paths) > 0 and len(paths) < self.vector_spam_limit:
            return True
            
        return False

# --- 3. GÜVENLİ CAPTION VE ASSET KAYDI ---

def generate_caption_safe(pil_image, model, processor):
    """Görsel için açıklama üretir."""
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
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}]}]
    
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
        logger.error(f"Caption Hatası: {e}")
        return "[Visual Description Failed]"

def save_page_asset(page, filename, page_num):
    """
    Sayfanın tamamını yüksek çözünürlükte (PNG) kaydeder.
    Bu dosya RAG Engine tarafından sorgu anında kullanıcıya gösterilmek üzere okunacaktır.
    """
    try:
        # 2x Zoom (Okunabilirlik için)
        mat = fitz.Matrix(CFG.IMAGE_DPI_SCALE, CFG.IMAGE_DPI_SCALE)
        pix = page.get_pixmap(matrix=mat)
        
        # Dosya adı: belge_sayfa_uuid.png
        safe_filename = os.path.splitext(filename)[0].replace(" ", "_")
        asset_name = f"{safe_filename}_p{page_num}_{uuid.uuid4().hex[:6]}.png"
        save_path = os.path.join(CFG.ASSETS_DIR, asset_name)
        
        pix.save(save_path)
        return save_path
    except Exception as e:
        logger.error(f"Asset Kayıt Hatası ({filename}): {e}")
        return ""

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
    logger.info(f"🚀 Ingestion Başlatılıyor (Asset Store Mode). Koleksiyon: {collection_name}")
    
    checkpoint = CheckpointManager()
    db_manager = DatabaseManager()
    col = db_manager.get_collection(collection_name)
    miner = CompositeVisualMiner()
    
    llm_manager = LLMManager()
    model, processor = llm_manager.load_vision_model()
    embedder = llm_manager.load_embedder()
    
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        
        if checkpoint.is_processed(filename):
            logger.info(f"⏭️ ATLANDI: {filename}")
            continue
            
        if not os.path.exists(pdf_path): continue

        try:
            doc = fitz.open(pdf_path)
        except: continue

        logger.info(f"📂 İşleniyor: {filename} ({doc.page_count} sayfa)")
        
        batch_chunks = []
        batch_metadatas = []
        
        for i, page in enumerate(tqdm(doc, desc=f"Sayfalar ({filename})")):
            try:
                text = page.get_text() or ""
                
                # Görsel Kontrolü
                has_visual = miner.extract_visual_crops(page)
                visual_summary = ""
                asset_path = ""
                
                if has_visual:
                    # 1. Sayfayı Asset olarak kaydet (RAG Engine için)
                    asset_path = save_page_asset(page, filename, i+1)
                    
                    # 2. Sayfanın görsel özetini çıkar (Captioning)
                    # Not: Burada sadece görsel varsa tüm sayfayı captionlıyoruz.
                    # Daha detaylı crop mantığı istenirse eklenebilir.
                    
                    # Performans için: Sayfanın küçük bir önizlemesini modele veriyoruz
                    pix_small = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                    img_data = pix_small.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data))
                    
                    caption = generate_caption_safe(pil_img, model, processor)
                    visual_summary = f"\n[VISUAL SUMMARY]: {caption}\n"

                full_content = f"--- SOURCE: {filename} | PAGE {i+1} ---\n"
                full_content += visual_summary
                full_content += f"[TEXT CONTENT]:\n{text}"

                splitter = RecursiveCharacterTextSplitter(chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP)
                chunks = splitter.split_text(full_content)

                for chunk in chunks:
                    batch_chunks.append(chunk)
                    batch_metadatas.append({
                        "source": filename, 
                        "page": i+1, 
                        "has_visual": str(has_visual),
                        "image_path": asset_path # <-- RAG Engine bunu okuyacak
                    })

                if len(batch_chunks) >= CFG.BATCH_SIZE:
                    save_batch(col, batch_chunks, batch_metadatas, embedder)
                    batch_chunks, batch_metadatas = [], []
                    gc.collect()

            except Exception as e:
                logger.error(f"Sayfa Hatası: {e}")
        
        doc.close()
        if batch_chunks:
            save_batch(col, batch_chunks, batch_metadatas, embedder)
        
        checkpoint.mark_as_done(filename)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("🎉 Tüm işlemler tamamlandı.")

if __name__ == "__main__":
    data_folder = "data" 
    if os.path.exists(data_folder):
        pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
        if pdf_files:
            # DİKKAT: Eski koleksiyon adını değiştirin veya DB'yi silin
            main_ingest(pdf_files, "doc_v2_asset_store") 
        else:
            print("Klasörde PDF bulunamadı.")