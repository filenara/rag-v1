import chromadb
from chromadb.config import Settings
import os
from src.utils import load_config

cfg = load_config()

class DatabaseManager:
    def __init__(self):
        # Ayarlardaki yola göre veritabanını başlat
        persist_path = cfg['vector_db']['persist_path']
        
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
        """Belirli bir döküman koleksiyonunu getirir."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def delete_collection(self, name):
        """Koleksiyonu siler (Admin için)."""
        try:
            self.client.delete_collection(name)
            return True
        except:
            return False