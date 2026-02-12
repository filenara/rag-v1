import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import cfg

class HybridRetriever:
    def __init__(self, vector_store):
        print("Retriever Başlatılıyor...")
        self.vector_db = vector_store
        
        # Veritabanındaki metinleri BM25 için RAM'e çekiyoruz
        # Production Notu: Çok büyük veride bunu Elasticsearch/Opensearch yapar.
        data = self.vector_db.collection.get(include=["documents", "metadatas"])
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        
        if self.documents:
            tokenized_corpus = [doc.split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
            print("UYARI: Veritabanı boş! Önce döküman ekleyin.")

        print(f"Reranker Yükleniyor ({cfg.DEVICE})...")
        self.reranker = CrossEncoder(
            cfg.RERANK_MODEL, 
            device=cfg.DEVICE,
            trust_remote_code=True
        )

    def search(self, query, top_k=None):
        if top_k is None: top_k = cfg.INITIAL_TOP_K
        if not self.bm25: return [], []

        # 1. Vektör Araması
        vec_res = self.vector_db.query_similar(query, n_results=top_k)
        vec_docs = vec_res['documents'][0] if vec_res['documents'] else []
        vec_metas = vec_res['metadatas'][0] if vec_res['metadatas'] else []

        # 2. BM25 Araması
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Adayları Birleştir (Basit havuzlama)
        candidates = {} # {doc_content: metadata}
        
        for d, m in zip(vec_docs, vec_metas):
            candidates[d] = m
            
        for idx in bm25_top_n:
            doc = self.documents[idx]
            candidates[doc] = self.metadatas[idx]

        unique_docs = list(candidates.keys())
        unique_metas = list(candidates.values())
        
        if not unique_docs: return [], []

        # 3. Reranking (Yeniden Sıralama)
        pairs = [[query, doc] for doc in unique_docs]
        scores = self.reranker.predict(pairs)
        
        sorted_indices = np.argsort(scores)[::-1][:cfg.FINAL_TOP_K]
        
        final_docs = [unique_docs[i] for i in sorted_indices]
        final_metas = [unique_metas[i] for i in sorted_indices]
        
        return final_docs, final_metas