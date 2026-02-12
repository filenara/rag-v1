import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config.settings import cfg
from src.database import VectorStoreManager
from src.miner import PDFMiner, generate_caption
from src.retriever import HybridRetriever

# Kaggle'da GPU kontrolü
print(f"Sistem Başlatılıyor. Cihaz: {cfg.DEVICE}")

def load_llm():
    print("Qwen-VL Modeli Yükleniyor (Bu işlem zaman alabilir)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)
    return model, processor

def ingest_data(folder_path, vector_db, model, processor):
    miner = PDFMiner()
    
    for file in os.listdir(folder_path):
        if not file.endswith(".pdf"): continue
        
        path = os.path.join(folder_path, file)
        print(f"İşleniyor: {file}")
        
        # Basit ingestion mantığı (Detaylandırılabilir)
        # Şimdilik sadece PDF varlığını simüle ediyoruz
        # Gerçek kodda burada fitz.open(path) ile sayfaları döneceksin
        pass

def main():
    # 1. Veritabanını Başlat
    db = VectorStoreManager()
    
    # 2. Modeli Yükle (Eğer sadece arama yapacaksan gerekmez, ama ingestion için lazım)
    model, processor = load_llm()
    
    # 3. Arama Motorunu Hazırla
    # (Not: Veritabanı boşsa önce ingestion yapman gerekir)
    retriever = HybridRetriever(db)
    
    # Örnek Sorgu
    query = "Devre kartındaki hasar nerede?"
    docs, metas = retriever.search(query)
    
    print("\n--- SONUÇLAR ---")
    for d, m in zip(docs, metas):
        print(f"Bulunan: {d[:100]}... (Kaynak: {m})")

if __name__ == "__main__":
    main()