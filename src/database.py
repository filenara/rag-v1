import chromadb
import os
import logging
from src.utils import load_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.cfg = load_config()
        
        persist_path = self.cfg.get('vector_db', {}).get('persist_path', 'data/chroma_db')
        
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
            
        self.client = chromadb.PersistentClient(path=persist_path)

    def list_collections(self):
        try:
            cols = self.client.list_collections()
            return [c.name for c in cols]
        except Exception as e:
            logger.error(f"DB Hatasi (list_collections): {e}", exc_info=True)
            return []

    def get_collection(self, name):
        metric = self.cfg.get('vector_db', {}).get('distance_metric', 'cosine')
        
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": metric}
        )
    
    def delete_collection(self, name):
        try:
            self.client.delete_collection(name)
            return True
        except Exception as e:
            logger.error(f"Koleksiyon silinirken hata olustu ({name}): {e}", exc_info=True)
            return False