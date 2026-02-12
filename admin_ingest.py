import os
import fitz
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.database import DatabaseManager
from src.miner import CompositeVisualMiner

# --- AYARLAR ---
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"

def load_captioning_model():
    print("Qwen2.5-VL Modeli Yükleniyor...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",            
        torch_dtype=torch.bfloat16,   
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28
    )
    return model, processor

def generate_caption(image, model, processor):
    # --- YENİ KAPSAYICI PROMPT ---
    system_prompt = """
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
    
    # Kullanıcı promptu
    user_prompt = "Exhaustive analysis."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Modelin ürettiği cevabı temizle (Prompt kısmını at)
    if "assistant\n" in output_text:
        return output_text.split("assistant\n")[-1]
    return output_text

def ingest_pdf(file_path, collection_name):
    print(f"İşleniyor: {file_path}")
    
    # 1. Modelleri Hazırla
    db = DatabaseManager()
    
    # Koleksiyonu temizle (Temiz başlangıç için)
    try: db.delete_collection(collection_name)
    except: pass
    col = db.get_collection(collection_name)
    
    print("Embedding Modeli Yükleniyor...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    
    qwen_model, qwen_processor = load_captioning_model()
    miner = CompositeVisualMiner()
    
    # 2. PDF Oku ve Analiz Et
    doc = fitz.open(file_path)
    full_text_buffer = ""
    
    print(f"Toplam {len(doc)} sayfa taranıyor...")
    
    for i, page in enumerate(doc):
        page_text = page.get_text()
        
        # Resimleri bul ve caption üret
        images = miner.extract_visual_crops(page)
        visual_context = ""
        
        if images:
            print(f"   -> Sayfa {i+1}: {len(images)} görsel bulundu, analiz ediliyor...")
            for idx, img in enumerate(images):
                try:
                    caption = generate_caption(img, qwen_model, qwen_processor)
                    visual_context += f"\n[GÖRSEL {idx+1} ANALİZİ]: {caption}\n"
                except Exception as e:
                    print(f"Caption Hatası: {e}")

        # Metin + Görsel Açıklamasını Birleştir
        page_content = f"--- PAGE {i+1} ---\n{visual_context}\n{page_text}\n"
        full_text_buffer += page_content

    # 3. Chunking (Parçalama)
    print("Metin parçalanıyor...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text_buffer)
    
    # 4. Embedding ve Kayıt
    if chunks:
        print(f"{len(chunks)} parça veritabanına yazılıyor...")
        embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=4).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path, "page": 0} for _ in chunks]
        
        col.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print("Yükleme Başarılı!")
    else:
        print("Kaydedilecek veri bulunamadı.")

if __name__ == "__main__":
    if os.path.exists("test.pdf"):
        ingest_pdf("test.pdf", "doc_kaggle_v1")
    else:
        print("'test.pdf' bulunamadı.")