import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from src.database import DatabaseManager
from src.utils import load_config

# Ayarları yükle
cfg = load_config()

def ingest_pdf(file_path, collection_name):
    print(f"🔄 İŞLEM BAŞLIYOR: {file_path}")
    
    # 1. Veritabanı Bağlantısı
    db = DatabaseManager()
    
    # Koleksiyon varsa silip baştan oluşturalım (Temiz başlangıç için)
    # Gerçek hayatta append (ekleme) yapmak isteyebilirsin
    try:
        db.delete_collection(collection_name)
        print(f"🗑️ Eski '{collection_name}' koleksiyonu silindi.")
    except:
        pass
        
    col = db.get_collection(collection_name)
    
    # 2. Embedding Modeli (CPU için hafif model)
    # Production'da burası BGE-M3 olacak
    print("🧠 Embedding modeli yükleniyor (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. PDF Okuma
    doc = fitz.open(file_path)
    documents = []
    metadatas = []
    ids = []
    
    print(f"📄 Toplam {len(doc)} sayfa okunuyor...")
    
    for i, page in enumerate(doc):
        text = page.get_text()
        
        # Sadece dolu sayfaları al
        if len(text.strip()) > 50:
            # Basit chunking (Sayfa bazlı)
            # İlerde burayı RecursiveCharacterTextSplitter ile yapacağız
            documents.append(text)
            metadatas.append({"source": file_path, "page": i + 1})
            ids.append(f"{collection_name}_p{i}")

    # 4. Vektörleştirme ve Kayıt
    if documents:
        print(f"📊 {len(documents)} parça vektörleştiriliyor...")
        embeddings = embedder.encode(documents).tolist()
        
        col.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ BAŞARILI! '{collection_name}' koleksiyonuna {len(documents)} parça eklendi.")
    else:
        print("❌ HATA: PDF'ten anlamlı metin çıkarılamadı.")

if __name__ == "__main__":
    # Test için burayı elle değiştirip çalıştırabilirsin
    # Örnek: python admin_ingest.py
    
    # Kullanıcıya soralım
    pdf_path = input("Yüklenecek PDF yolunu girin (örn: test.pdf): ")
    col_name = input("Koleksiyon adı ne olsun? (örn: cihaz_bakim): ")
    
    if os.path.exists(pdf_path):
        ingest_pdf(pdf_path, col_name)
    else:
        print("Dosya bulunamadı!")