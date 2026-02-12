import time
import random
from src.utils import load_config
from src.database import DatabaseManager

cfg = load_config()

class RAGEngine:
    def __init__(self):
        self.mock_mode = cfg['system']['use_mock_llm']
        self.db_manager = DatabaseManager()
        
        if not self.mock_mode:
            print("🟢 [Sistem] Gerçek GPU Modelleri Yükleniyor... (Bu biraz sürebilir)")
            # PRODUCTION: Buraya Qwen ve Embedding model yükleme kodları gelecek.
            # Şimdilik sadece yer tutucu, Phase 3'te burayı dolduracağız.
            self.model = None 
            self.processor = None
        else:
            print("🛠️ [Sistem] MOCK Modu Aktif. GPU kullanılmıyor.")

    def search_and_answer(self, query, collection_name, history=[]):
        """
        Sorguyu alır, dökümanda arar ve cevap üretir.
        """
        # 1. Koleksiyonu Seç
        if not collection_name:
            return "Lütfen önce bir döküman seçin.", [], []

        # 2. Arama Yap (Retrieval)
        # Mock modunda bile veritabanından veri çekmeye çalışalım
        col = self.db_manager.get_collection(collection_name)
        
        # Basit embedding taklidi (Gerçek embedding entegre edilene kadar)
        # Production'da burası 'embedding_model.encode(query)' olacak
        results = col.query(
            query_embeddings=[[0.1] * 384], # Rastgele vektör (Mock)
            n_results=3
        )
        
        # 3. Cevap Üret (Generation)
        if self.mock_mode:
            # --- MOCK CEVAP SİMÜLASYONU ---
            time.sleep(1.5) # Yapay zeka düşünüyor efekti
            
            # Rastgele bir STE100 hatası sıkıştıralım ki Guard'ı test edelim
            response_text = (
                f"MOCK CEVAP: '{collection_name}' dökümanına bakarak söylüyorum.\n\n"
                f"Sorduğunuz '{query}' hakkında teknik veriler incelendi.\n"
                f"Sistem şu an stabil çalışıyor. Ancak, please utilize the emergency button." 
                # Not: 'utilize' kelimesi yasaklı, bunu bilerek koydum.
            )
            
            sources = ["Sayfa 1 (Giriş)", "Sayfa 5 (Teknik Veriler)"]
            return response_text, sources
        
        else:
            # --- GERÇEK CEVAP ---
            # Burası Kaggle/GPU makinesi için kodlanacak
            return "Gerçek model henüz bağlı değil.", []