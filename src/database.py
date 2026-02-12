import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import cfg

class VectorStoreManager:
    def __init__(self, persist_dir="./chroma_db_data", reset_db=False):
        self.persist_dir = persist_dir
        
        # Eğer sıfırdan başlamak isteniyorsa klasörü sil
        if reset_db and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            
        # Embedding modeli (CPU/GPU ayarı config'den gelir)
        print(f"Embedding Modeli Yükleniyor: {cfg.DEVICE}")
        self.embedder = SentenceTransformer(cfg.EMBED_MODEL, device=cfg.DEVICE)
        
        # ChromaDB İstemcisi
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks, metadatas):
        if not chunks:
            return 0
            
        # Embedding oluştur
        embeddings = self.embedder.encode(chunks, batch_size=32, show_progress_bar=True).tolist()
        
        # ID oluştur (Mevcut sayı üzerinden devam et)
        current_count = self.collection.count()
        ids = [str(current_count + i) for i in range(len(chunks))]
        
        # Veritabanına ekle
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def query_similar(self, query_text, n_results=5):
        query_vec = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=n_results
        )
        return results