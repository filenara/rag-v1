import torch
import gc
import numpy as np
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info
from src.database import DatabaseManager

# Singleton Model Yükleyici (Tekrar tekrar yüklemesin diye)
class ModelLoader:
    _instance = None
    
    @classmethod
    def get_models(cls):
        if cls._instance is None:
            print("🚀 [Sistem] Dev Modeller Yükleniyor (GPU)...")
            
            # 1. Qwen2.5-VL (Vision Language Model)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True, min_pixels=256*28*28, max_pixels=1280*28*28)
            
            # 2. Reranker (Hakem Model)
            reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device="cuda", trust_remote_code=True)
            
            cls._instance = (model, processor, reranker)
            print("✅ Modeller Hazır!")
            
        return cls._instance

class RAGEngine:
    def __init__(self):
        self.db_manager = DatabaseManager()
        
        # Modelleri yükle
        self.model, self.processor, self.reranker = ModelLoader.get_models()
        
        # BM25 (Kelime Bazlı Arama) Hazırlığı
        print("📊 [Sistem] BM25 İndeksi oluşturuluyor...")
        self.col = self.db_manager.get_collection("doc_default") # Varsayılan koleksiyon
        data = self.col.get() # Tüm veriyi çek (Dikkat: Çok büyük veride optimize edilmeli)
        self.documents = data['documents']
        self.metadatas = data['metadatas']
        
        if self.documents:
            tokenized_corpus = [doc.split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def hybrid_search(self, query, top_k=15, final_k=3):
        if not self.bm25: return [], []

        # 1. Vektör Arama (Anlamsal)
        # Not: Burada embedder'ı db_manager içinden çağırmamız lazım veya tekrar tanımlamalıyız.
        # Basitlik için veritabanı sınıfına güveniyoruz.
        vec_results = self.col.query(query_texts=[query], n_results=top_k)
        
        doc_scores = {}
        doc_metas = {}
        
        # Vektör Sonuçlarını Puanla (RRF)
        if vec_results['documents']:
            for rank, (doc, meta) in enumerate(zip(vec_results['documents'][0], vec_results['metadatas'][0])):
                if doc not in doc_scores:
                    doc_scores[doc] = 0
                    doc_metas[doc] = meta
                doc_scores[doc] += 1 / (60 + rank)

        # 2. BM25 Arama (Kelime)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n = np.argsort(bm25_scores)[::-1][:top_k]
        
        for rank, idx in enumerate(bm25_top_n):
            doc = self.documents[idx]
            if doc not in doc_scores:
                doc_scores[doc] = 0
                doc_metas[doc] = self.metadatas[idx]
            doc_scores[doc] += 1 / (60 + rank)
            
        # 3. Reranking (Sıralama)
        sorted_candidates = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        candidates = [item[0] for item in sorted_candidates]
        candidate_metas = [doc_metas[doc] for doc in candidates]
        
        if not candidates: return [], []
        
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1][:final_k]
        
        final_docs = [candidates[i] for i in sorted_indices]
        final_metas = [candidate_metas[i] for i in sorted_indices]
        
        return final_docs, final_metas

    def search_and_answer(self, query, collection_name=None):
        # Hibrit Arama Yap
        docs, metas = self.hybrid_search(query)
        
        if not docs:
            return "Üzgünüm, dökümanlarda bu bilgiye rastlayamadım.", []

        context_text = "\n\n".join(docs)
        best_meta = metas[0]
        
        # Prompt Hazırla
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer strictly based on the context provided."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        # Eğer en alakalı parçada resim varsa, onu da modele ver (Gelişmiş özellik)
        # Not: Bu kısım için PDF'ten anlık render gerekir, şimdilik metin tabanlı ilerliyoruz.
        
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], padding=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Temizlik (System promptu vs sil)
        response = response.split("assistant\n")[-1]
        
        sources = [f"Sayfa {m['page']}" for m in metas]
        return response, sources