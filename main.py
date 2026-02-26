import os
from src.rag_engine import RAGEngine
from src.utils import load_config


def main():
    print("Sistem Arama Modu Baslatiliyor...")

    # 1. Ayarlari ve Motoru Hazirla
    config = load_config()
    collection_name = config.get("vector_db", {}).get("collection_name", "doc_v2_asset_store")
    
    engine = RAGEngine()
    
    # 2. Ornek Sorgu (Test)
    query = "Devre kartindaki hasar nerede?"
    print(f"\nSoru: {query}")
    
    # 3. Aramayi Yap
    print("\nArama ve uretim islemi basladi, lutfen bekleyin...")
    final_answer, context_text = engine.search_and_answer(
        query=query, 
        collection_name=collection_name,
        history=[],
        user_image=None,
        use_ste100=False
    ) 
    
    print("\n--- URETILEN CEVAP ---")
    print(final_answer)
    
    print("\n--- BULUNAN BAGLAM (CONTEXT) ---")
    if context_text:
        # Metin cok uzun olabilecegi icin ilk 500 karakteri yazdiriyoruz
        print(f"{context_text[:500]}...")
    else:
        print("Baglam bulunamadi.")


if __name__ == "__main__":
    main()