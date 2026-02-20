import os
import io
import gc
import json
import logging
import fitz  # PyMuPDF
import torch
import uuid
import pickle
from rank_bm25 import BM25Okapi
from PIL import Image
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qwen_vl_utils import process_vision_info


# --- KENDİ MODÜLLERİMİZ ---
from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.utils import load_prompts

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
class STE100SemanticSplitter:
    """
    PDF içindeki font boyutlarına ve kalınlıklarına (Bold) bakarak
    metni anlamsal (semantic) bütünlük içinde, başlıklarına göre böler.
    """
    def __init__(self, normal_text_size_limit=11.0):
        # Bu değerin üzerindeki fontları "Başlık" kabul edeceğiz.
        # PDF'ine göre bu değeri 10, 11 veya 12 olarak ayarlayabilirsin.
        self.normal_text_size_limit = normal_text_size_limit

    def extract_semantic_chunks(self, page):
        semantic_chunks = []
        current_heading = "Genel Bağlam"
        current_content = ""

        # Sayfadaki tüm elementleri detaylı sözlük (dict) olarak al
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if b['type'] == 0:  # Eğer bu blok bir 'metin' ise
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text: continue
                        
                        font_size = span["size"]
                        font_name = span["font"].lower()

                        # BAŞLIK TESPİTİ: Font boyutu büyükse VEYA font isminde 'bold' geçiyorsa
                        is_heading = font_size > self.normal_text_size_limit or "bold" in font_name

                        if is_heading:
                            # Eski başlığı ve içeriğini kaydet (Eğer içi boş değilse)
                            if current_content.strip() and len(current_content) > 20:
                                chunk_text = f"[{current_heading}]\n{current_content.strip()}"
                                semantic_chunks.append(chunk_text)
                            
                            # Yeni başlığa geç
                            current_heading = text
                            current_content = ""
                        else:
                            # Başlık değilse, mevcut içeriğe ekle
                            current_content += text + " "
                
                # Bloklar (paragraflar) arası boşluk bırak
                current_content += "\n\n"

        # Sayfa bitiminde son kalan içeriği de ekle
        if current_content.strip() and len(current_content) > 20:
            chunk_text = f"[{current_heading}]\n{current_content.strip()}"
            semantic_chunks.append(chunk_text)

        return semantic_chunks

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
    """Görselleri ve alt yazıları akıllıca ayıklar ve birleştirir."""
    def __init__(self, min_width=CFG.MIN_WIDTH, min_height=CFG.MIN_HEIGHT, vector_spam_limit=600):
        self.min_width = min_width
        self.min_height = min_height
        self.vector_spam_limit = vector_spam_limit 

    def _boxes_intersect_or_close(self, box1, box2, margin=150): 
        """İki kutunun birbirine yakın olup olmadığını kontrol eder."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max + margin < x2_min or x1_min - margin > x2_max or
                    y1_max + margin < y2_min or y1_min - margin > y2_max)

    def _merge_boxes(self, boxes):
        """Yakın kutuları tek bir büyük Bounding Box içinde birleştirir."""
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
        """Sayfadaki görselleri ve alt yazıları tespit edip kırpar."""
        all_visual_rects = []

        # 1. Resimleri Tespit Et
        images = page.get_images(full=True)
        for img in images:
            rect = page.get_image_bbox(img)
            if (rect.width > 50 and rect.height > 50):
                all_visual_rects.append(list(rect))
                
        # 2. Çizim/Tablo Tespit Et
        paths = page.get_drawings()
        is_complex_table = len(paths) > self.vector_spam_limit
        if not is_complex_table:
            for path in paths:
                rect = path["rect"]
                if rect.width > 5 or rect.height > 5:
                    all_visual_rects.append(list(rect))

        # 3. Kısa Metinleri (Şekil Alt Yazıları vb.) Tespit Et
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            if len(block[4]) < 200: # 200 karakterden kısaysa muhtemelen başlıktır
                all_visual_rects.append([block[0], block[1], block[2], block[3]])

        if not all_visual_rects: return []

        # 4. Yakın Öğeleri Birleştir
        merged_rects = self._merge_boxes(all_visual_rects)
        
        # 5. Kırp ve PIL Image Olarak Dön
        final_crops = []
        for rect in merged_rects:
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            if w < self.min_width or h < self.min_height: continue
            
            try:
                clip_rect = fitz.Rect(rect[0], rect[1], rect[2], rect[3])
                mat = fitz.Matrix(CFG.IMAGE_DPI_SCALE, CFG.IMAGE_DPI_SCALE)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                img_data = pix.tobytes("png")
                final_crops.append(Image.open(io.BytesIO(img_data)))
            except: 
                pass
                
        return final_crops
# --- 3. GÜVENLİ CAPTION VE ASSET KAYDI ---

def generate_caption_safe(pil_image, model, processor, prompt_text):
    """Görsel için açıklamayı YAML'dan gelen prompt ile üretir."""
    
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

    semantic_splitter = STE100SemanticSplitter(normal_text_size_limit=11.0)
    
    llm_manager = LLMManager()
    model, processor = llm_manager.load_vision_model()
    embedder = llm_manager.load_embedder()

    prompts = load_prompts()
    # Eğer YAML'da bulamazsa güvenlik için kısa bir fallback metni koyuyoruz
    caption_prompt = prompts.get("caption_prompt", "Describe this technical image accurately.")
    
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
                
                detected_images = miner.extract_visual_crops(page)
                visual_summary = ""
                asset_path = ""
                has_visual = False
                
                if detected_images:
                    has_visual = True
                    # 1. Sayfanın TAMAMINI Asset olarak kaydet (RAG Engine için)
                    # Bu kısmı ellemeyin, RAG Engine hala kullanıcıya tam sayfayı gösterecek.
                    asset_path = save_page_asset(page, filename, i+1)
                    
                    # 2. Sadece KIRPILMIŞ görsellerin özetini çıkar (Captioning)
                    visual_summary += "\n[VISUAL/TABLE DETECTED]:\n"
                    # Sayfada birden fazla kırpılmış görsel/tablo olabilir, hepsini tek tek dönüyoruz
                    for idx, img in enumerate(detected_images):
                        try:
                            # Modele artık tüm sayfa değil, sadece 'img' (kırpılmış bölge) gidiyor
                            caption = generate_caption_safe(img, model, processor, caption_prompt)
                            visual_summary += f"- Analysis {idx+1}: {caption}\n"
                        except Exception as img_err:
                            logger.error(f"Görsel analiz hatası: {img_err}")

                full_content = f"--- SOURCE: {filename} | PAGE {i+1} ---\n"
                full_content += visual_summary
                full_content += f"[TEXT CONTENT]:\n{text}"
                
                # ... (Görsel işlemleri aynı kalacak) ...
                
                # Sadece düz 'text' almak yerine, anlamsal chunk'ları alıyoruz
                page_chunks = semantic_splitter.extract_semantic_chunks(page)

                for chunk_text in page_chunks:
                    # Görsel özetleri varsa (visual_summary), ilk veya ilgili chunk'a yedirebiliriz
                    # Şimdilik her chunk'ın başına genel sayfa bağlamını ekliyoruz
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
                logger.error(f"Sayfa Hatası: {e}")
        
        doc.close()
        if batch_chunks:
            save_batch(col, batch_chunks, batch_metadatas, embedder)
        
        checkpoint.mark_as_done(filename)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("BM25 İndeksi oluşturuluyor ve diske kaydediliyor...")
    all_docs_data = col.get()
    
    if all_docs_data and all_docs_data['documents']:
        tokenized_corpus = [doc.lower().split(" ") for doc in all_docs_data['documents']]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Sadece modeli değil, ID ve metadataları da hizalı tutmak için paketliyoruz
        bm25_cache = {
            "bm25": bm25,
            "ids": all_docs_data['ids'],
            "documents": all_docs_data['documents'],
            "metadatas": all_docs_data['metadatas']
        }
        
        cache_path = os.path.join(CFG.ASSETS_DIR, "..", "bm25_cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(bm25_cache, f)
        logger.info(f" BM25 Cache başarıyla kaydedildi: {cache_path}")
    else:
        logger.warning("Veritabanı boş, BM25 indeksi oluşturulamadı.")
    # --- YENİ KODLAR BURADA BİTİYOR ---

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