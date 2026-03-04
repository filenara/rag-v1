import os
import uuid
import pickle
import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi

from src.database import DatabaseManager
from src.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class VectorIndexer:
    def __init__(self, collection_name: str, bm25_path: str):
        self.collection_name = collection_name
        self.bm25_path = bm25_path
        
        self.db_manager = DatabaseManager()
        self.collection = self.db_manager.get_collection(self.collection_name)
        
        self.llm_manager = LLMManager()
        self.embedder = self.llm_manager.load_embedder()

    def save_batch(self, batch_chunks: List[str], batch_metadatas: List[Dict]) -> None:
        if not batch_chunks:
            return

        logger.info(f"{len(batch_chunks)} adet metin parcasi vektorlestiriliyor...")
        
        embeddings = self.embedder.encode(batch_chunks, normalize_embeddings=True).tolist()
        
        ids = [str(uuid.uuid4()) for _ in range(len(batch_chunks))]
        
        self.collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            ids=ids
        )
        logger.info("Batch basariyla ChromaDB'ye kaydedildi.")

    def build_and_save_bm25(self) -> None:
        logger.info("BM25 indeksi olusturuluyor...")
        
        all_data = self.collection.get()
        documents = all_data.get("documents", [])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        
        if not documents:
            logger.warning("Veritabaninda dokuman bulunamadi. BM25 olusturulamadi.")
            return

        tokenized_corpus = [str(doc).lower().split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        
        cache_data = {
            "bm25": bm25,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas
        }
        
        with open(self.bm25_path, "wb") as f:
            pickle.dump(cache_data, f)
            
        logger.info(f"BM25 indeksi basariyla kaydedildi: {self.bm25_path}")