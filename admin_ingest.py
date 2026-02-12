import os
import fitz
import torch
from src.database import DatabaseManager
from src.miner import CompositeVisualMiner
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# GPU Kontrolü (Sadece Kaggle'da çalışsın diye)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ingest_pdf_advanced(file_path, collection_name):
    print(f"🚀 İŞLEM BAŞLIYOR: {file_path} (Cihaz: {DEVICE})")
    
    # 1. Veritabanı ve Embedder
    db = DatabaseManager()
    col = db.get_collection(collection_name)
    embedder = SentenceTransformer('BAAI/bge-m3', device=DEVICE)
    
    # 2. Miner (Görsel Kazıyıcı)
    miner = CompositeVisualMiner()
    
    # 3. PDF Okuma ve Analiz
    doc = fitz.open(file_path)
    full_text_content = ""
    
    for i, page in enumerate(doc):
        print(f"📄 Sayfa {i+1} işleniyor...")
        
        # A) Metni Al
        text = page.get_text()
        
        # B) Görselleri Al
        images = miner.extract_visual_crops(page)
        visual_text = ""
        
        if images and DEVICE == "cuda":
            # Burası sadece GPU varsa çalışır! Resimlere caption yazar.
            # Localde hata vermemesi için pass geçiyoruz.
            visual_text = f"\n[Görsel Tespit Edildi: {len(images)} adet. Analiz için Qwen-VL gerekli.]\n"
        
        page_content = f"--- PAGE {i+1} ---\n{visual_text}\n{text}\n"
        full_text_content += page_content

    # 4. Akıllı Bölme (Chunking)
    print("🔪 Metin parçalanıyor...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text_content)
    
    # 5. Embedding ve Kayıt
    print(f"📊 {len(chunks)} parça vektörleştiriliyor...")
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_path, "page": 0} for _ in chunks] # Basitlik için sayfa 0
    
    col.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print("✅ Yükleme Tamamlandı!")

if __name__ == "__main__":
    pdf_path = "test.pdf"
    if os.path.exists(pdf_path):
        ingest_pdf_advanced(pdf_path, "doc_default")
    else:
        print("Dosya bulunamadı.")