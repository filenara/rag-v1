import os
import torch
import fitz  
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import cfg
from src.database import VectorStoreManager
from src.miner import PDFMiner, generate_caption
from src.retriever import HybridRetriever

print(f"Sistem Başlatılıyor. Cihaz: {cfg.DEVICE}")

def load_llm():
    """
    Görsel Analiz (Captioning) ve Cevaplama için Qwen-VL modelini yükler.
    """
    print("Qwen-VL Modeli Yükleniyor (Bu işlem zaman alabilir)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=torch.bfloat16, # GPU belleği için optimize edildi
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)
    return model, processor

def ingest_data(folder_path, vector_db, llm_model, llm_processor):
    """
    PDF'leri okur, görselleri Qwen ile analiz eder, metinle birleştirir 
    ve vektör veritabanına kaydeder.
    """
    print("Embedding Modeli (BGE-M3) Yükleniyor...")
    # Metinleri vektöre çevirecek model
    embedder = SentenceTransformer('BAAI/bge-m3', device=cfg.DEVICE)
    
    # Metin parçalayıcı
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Görsel madencisi (src/miner.py içindeki sınıf)
    miner = PDFMiner()

    # Klasördeki dosyaları tara
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print("Klasörde PDF bulunamadı.")
        return

    # Veritabanı koleksiyonunu al (yoksa oluşturur)
    col = vector_db.get_collection("doc_default")

    for file in pdf_files:
        path = os.path.join(folder_path, file)
        print(f"İşleniyor: {file}")
        
        doc = fitz.open(path)
        full_text_buffer = ""
        
        for i, page in enumerate(doc):
            # 1. Ham Metni Al
            page_text = page.get_text()
            
            # 2. Görselleri Bul ve Analiz Et
            images = miner.extract_visual_crops(page)
            visual_context = ""
            
            if images:
                print(f"   -> Sayfa {i+1}: {len(images)} görsel bulundu, Qwen ile analiz ediliyor...")
                for idx, img in enumerate(images):
                    try:
                        # Qwen modele resmi gönderip açıklama istiyoruz
                        caption = generate_caption(img, llm_model, llm_processor)
                        visual_context += f"\n[GÖRSEL {idx+1} ANALİZİ]: {caption}\n"
                    except Exception as e:
                        print(f"   Caption Hatası: {e}")
            
            # 3. Metin ve Görsel Açıklamasını Birleştir
            # Böylece hem yazı hem de resimdeki bilgi aranabilir olur.
            page_content = f"--- PAGE {i+1} ---\n{visual_context}\n{page_text}\n"
            full_text_buffer += page_content

        # 4. Parçalama (Chunking)
        chunks = splitter.split_text(full_text_buffer)
        
        if not chunks:
            print(f"{file} dosyasından anlamlı metin çıkarılamadı.")
            continue

        # 5. Embedding ve Kayıt
        print(f"{len(chunks)} parça vektörleştiriliyor ve kaydediliyor...")
        
        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
        
        # Metadata hazırlığı
        ids = [f"{file}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file, "page": 0} for _ in chunks] # Basitlik için sayfa 0 genel etiketi
        
        col.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"{file} tamamlandı!")

def main():
    # 1. Veritabanını Başlat
    db = VectorStoreManager()
    
    # 2. Modeli Yükle (Hem ingestion hem cevaplama için lazım)
    model, processor = load_llm()
    
    # 3. Veri Yükleme (Ingestion)
    # Veri klasörünü kontrol et, varsa işle
    data_folder = "./data"
    if os.path.exists(data_folder) and os.listdir(data_folder):
        # Kullanıcıya sorabilirsin veya her çalıştırışta kontrol edebilirsin
        # Şimdilik doluysa işlem yap diyelim
        print("Veri klasörü taraniyor...")
        ingest_data(data_folder, db, model, processor)
    else:
        print("Veri klasörü boş veya yok, sadece arama yapılacak.")

    # 4. Arama Motorunu Hazırla
    retriever = HybridRetriever(db)
    
    # 5. Örnek Sorgu (Test)
    query = "Devre kartındaki hasar nerede?"
    print(f"\nSoru: {query}")
    
    # Sadece search fonksiyonunu çağırıyoruz, Qwen ile cevap üretme kısmı 
    # rag_engine.py veya benzeri bir yerde yapılmalı ama burada da basitçe gösterilebilir.
    docs, metas = retriever.search(query)
    
    print("\n--- BULUNAN DÖKÜMANLAR ---")
    for i, (d, m) in enumerate(zip(docs, metas)):
        print(f"\n{i+1}. Sonuç (Kaynak: {m.get('source')}):")
        print(f"{d[:200]}...") # İlk 200 karakter

if __name__ == "__main__":
    main()