import chromadb
import os
from src.utils import load_config

class DatabaseManager:
    def __init__(self):
        # Ayarları yükle
        self.cfg = load_config()
        
        # Veritabanı yolu
        persist_path = self.cfg['vector_db']['persist_path']
        
        # Klasör yoksa oluştur
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
            
        self.client = chromadb.PersistentClient(path=persist_path)

    def list_collections(self):
        """Sistemdeki tüm döküman setlerini listeler."""
        try:
            cols = self.client.list_collections()
            return [c.name for c in cols]
        except Exception as e:
            print(f"DB Hatası: {e}")
            return []

    def get_collection(self, name):
        """
        Belirli bir döküman koleksiyonunu getirir veya oluşturur.
        Mesafe metriği artık ayarlardan dinamik olarak okunuyor.
        """
        # Varsayılan olarak 'cosine' kullanır, ayar yoksa hata vermez.
        metric = self.cfg['vector_db'].get('distance_metric', 'cosine')
        
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": metric}
        )
    
    def delete_collection(self, name):
        """Koleksiyonu siler (Admin/Yönetici işlemleri için)."""
        try:
            self.client.delete_collection(name)
            return True
        except:
            return False