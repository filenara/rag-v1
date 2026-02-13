import os
from config.settings import cfg # Eğer config dosyan varsa
from src.database import DatabaseManager # İsmi VectorStoreManager değil DatabaseManager yapmıştık
from src.retriever import HybridRetriever

# Ingestion işlemi artık "admin_ingest.py" üzerinden yapılıyor.
# Bu dosya sadece sistemin çalışıp çalışmadığını test etmek içindir.

def main():
    print("Sistem Arama Modu Başlatılıyor...")

    # 1. Veritabanını Bağla
    db = DatabaseManager()
    
    # 2. Arama Motorunu Hazırla
    # Not: Retriever içinde embedding modeli yüklenecektir.
    retriever = HybridRetriever(db)
    
    # 3. Örnek Sorgu (Test)
    query = "Devre kartındaki hasar nerede?"
    print(f"\nSoru: {query}")
    
    # Koleksiyon adını admin_ingest.py'de ne verdiysen o olmalı (örn: doc_kaggle_v1)
    # Retriever kodunu incelemedik ama genellikle bir collection_name parametresi ister.
    # Varsayılan olarak kodunda nasılsa öyle bırakıyorum.
    docs, metas = retriever.search(query) 
    
    print("\n--- BULUNAN DÖKÜMANLAR ---")
    if docs:
        for i, (d, m) in enumerate(zip(docs, metas)):
            print(f"\n{i+1}. Sonuç (Kaynak: {m.get('source')} - Sayfa: {m.get('page')}):")
            print(f"{d[:200]}...") 
    else:
        print("Sonuç bulunamadı.")

if __name__ == "__main__":
    main()