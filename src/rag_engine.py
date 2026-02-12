import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from src.database import DatabaseManager

class RAGEngine:
    def __init__(self):
        print("Sistem Başlatılıyor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db = DatabaseManager()
        
        # 1. Embedding Modeli (Sorgu İçin)
        self.embedder = SentenceTransformer("BAAI/bge-m3", device=self.device)
        
        # 2. Reranker (Hakem)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=self.device, trust_remote_code=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

    def is_feedback_intent(self, query):
        """Kullanıcının bir hata bildirdiğini veya 'tekrar bak' dediğini anlar."""
        feedback_words = ["yanlış", "hatalı", "tekrar bak", "düzelt", "wrong", "incorrect", "re-examine", "look again"]
        return any(word in query.lower() for word in feedback_words)

    def search_and_answer(self, query, collection_name, history=[]):
     
        # --- 1. ADIM: NİYET ANALİZİ (FEEDBACK LOOP) ---
        is_feedback = self.is_feedback_intent(query)
        
        if is_feedback and len(history) > 0:
            print("[Niyet] Feedback algılandı. Önceki bağlam üzerinden analiz yapılıyor...")
            # Geçmişten en son kullanılan context'i bul
            context = ""
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context" in turn:
                    context = turn["context"]
                    break
            
            intent_instruction = "The user indicated the previous answer was incorrect. Carefully re-examine the provided context for missed technical details or subtle information."
        
        else:
            print(f"[Niyet] Yeni arama başlatılıyor: {query}")
            # --- 2. ADIM: HİBRİT ARAMA VE RERANKING ---
            col = self.db.get_collection(collection_name)
            
            # A) Vektör Arama
            query_vec = self.embedder.encode([query]).tolist()
            vec_results = col.query(query_embeddings=query_vec, n_results=20)
            
            # B) BM25 Arama
            all_docs_data = col.get()
            documents = all_docs_data['documents']
            if not documents: return "Veritabanı boş.", ""
            
            tokenized_corpus = [doc.lower().split(" ") for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.lower().split(" "))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:20]
            
            # Adayları Birleştir
            candidates = []
            seen = set()
            if vec_results['documents']:
                for doc in vec_results['documents'][0]:
                    if doc not in seen: candidates.append(doc); seen.add(doc)
            for idx in top_bm25_indices:
                doc = documents[idx]
                if doc not in seen: candidates.append(doc); seen.add(doc)
                
            # C) Reranking
            pairs = [[query, doc] for doc in candidates]
            scores = self.reranker.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1][:3] # En iyi 3 parça
            context = "\n\n".join([candidates[i] for i in sorted_indices])
            
            intent_instruction = "Answer strictly based on the technical context provided."

        # --- 3. ADIM: GELİŞMİŞ MULTI-MODAL PROMPT ENTEGRASYONU ---
        prompt = f"""Role: Senior Technical Systems Engineer & Documentation Analyst.

        Instruction: {intent_instruction}

        Core Instructions:
        1. Language & Protocol: 
        - ALWAYS respond in ENGLISH.
        - Provide direct, technical answers. No "Hello," "Based on the text," or meta-commentary.
        - Use professional, engineering-grade terminology.

        2. Multi-Modal Reasoning:
        - You are provided with [TEXT CONTEXT] and [IMAGE ANALYSIS/CAPTIONS].
        - Treat descriptions of diagrams, graphs, and tables as immutable facts.
        - If there is a conflict between text and image analysis, prioritize visual evidence from the image for physical descriptions (colors, damage, layouts).
        - Link text labels to image regions (e.g., if text mentions 'J1 connector' and image analysis shows a label 'J1', treat them as the same entity).

        3. Technical Accuracy:
        - Tables: Analyze data row-by-row and column-by-column. Maintain relationships between headers and values.
        - Diagrams/Schematics: Follow signal paths or connections mentioned in text or visible in captions (e.g., arrows, lines).
        - Maintenance/Safety: Strictly follow any warnings or procedures found in the fragments.

        4. Guardrails & Integrity:
        - Zero-Knowledge Rule: If the combined context (text + image captions) does not contain the answer, state: "Information not found in provided fragment."
        - No Hallucinations: Do not assume connections not explicitly stated or visible.
        - STE100 Compliance: Use simple, direct verbs (e.g., "Use" instead of "Utilize") to ensure technical clarity.

        Input Handling:
        - Primarily analyze the provided Image Pixel data (if available) or Visual Captions.
        - Do not state "the text does not mention this" if the information is present in the visual analysis.
         
        Context:
        {context}
        
        Question: {query}
        Answer:"""
        
        # Qwen-VL Inference
        messages = [{"role": "user", "content": prompt}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], padding=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_answer = response.split("assistant\n")[-1] if "assistant\n" in response else response

        return final_answer, context