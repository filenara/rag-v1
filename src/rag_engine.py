import torch
import numpy as np
from rank_bm25 import BM25Okapi
from src.database import DatabaseManager
from src.llm_manager import LLMManager

class RAGEngine:
    def __init__(self):
        """
        Lazy Mode: Modeller burada YÜKLENMEZ. Sadece yönetici çağrılır.
        """
        print("⚡ RAGEngine Başlatılıyor (Conversation Aware Mode)...")
        self.db = DatabaseManager()
        self.llm_manager = LLMManager() 

    def _is_feedback_intent(self, query):
        """Kullanıcının bir hata bildirdiğini veya 'tekrar bak' dediğini anlar."""
        feedback_words = ["yanlış", "hatalı", "tekrar bak", "düzelt", "wrong", "incorrect", "re-examine", "look again"]
        return any(word in query.lower() for word in feedback_words)

    def _format_history(self, history):
        """Sohbet geçmişini Prompt'a uygun metne dönüştürür."""
        if not history:
            return "No previous conversation."
        
        formatted = ""
        for turn in history:
            role = "User" if turn['role'] == 'user' else "Assistant"
            content = turn['content'].replace('\n', ' ') # Satır sonlarını temizle
            formatted += f"{role}: {content}\n"
        return formatted

    def search_and_answer(self, query, collection_name, history=[]):
        """
        Arama ve Cevaplama fonksiyonu.
        """
        
        # --- 1. MODELLERİ ÇAĞIR ---
        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()
        
        # --- 2. NİYET VE BAĞLAM ANALİZİ ---
        is_feedback = self._is_feedback_intent(query)
        intent_instruction = "Answer based on the context and previous conversation."
        
        context = ""
        
        # Geçmişi formatla
        history_text = self._format_history(history)

        if is_feedback and len(history) > 0:
            print("[Niyet] Feedback algılandı. Önceki bağlam tekrar inceleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context" in turn:
                    context = turn["context"]
                    break
            
            if not context:
                print("[Uyarı] Geçmişte context bulunamadı, yeni arama yapılıyor...")
                is_feedback = False 
            else:
                intent_instruction = "The user indicated the previous answer was incorrect. Carefully re-examine the provided context."
        
        # Feedback değilse standart arama yap
        if not is_feedback:
            print(f"[Niyet] Yeni arama: {query}")
            
            # --- 3. HİBRİT ARAMA ---
            col = self.db.get_collection(collection_name)
            
            # A) Vektör
            query_vec = embedder.encode([query]).tolist()
            vec_results = col.query(query_embeddings=query_vec, n_results=20)
            
            # B) BM25
            all_docs_data = col.get()
            documents = all_docs_data['documents']
            
            if not documents: return "Veritabanı boş.", ""
            
            tokenized_corpus = [doc.lower().split(" ") for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.lower().split(" "))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:20]
            
            # Birleştir
            candidates = []
            seen = set()
            if vec_results['documents']:
                for doc in vec_results['documents'][0]:
                    if doc not in seen: candidates.append(doc); seen.add(doc)
            for idx in top_bm25_indices:
                doc = documents[idx]
                if doc not in seen: candidates.append(doc); seen.add(doc)
            
            if not candidates: return "İlgili sonuç bulunamadı.", ""

            # C) Reranking
            pairs = [[query, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1][:3] 
            context = "\n\n".join([candidates[i] for i in sorted_indices])
            
        # --- 4. CEVAP ÜRETİMİ (PROMPT ENTEGRASYONU) ---
        
        model, processor = self.llm_manager.load_vision_model()
        
        # Prompt: History + Context + Question
        prompt = f"""Role: Senior Technical Systems Engineer.
        Instruction: {intent_instruction}

        [PREVIOUS CONVERSATION]
        {history_text}
        [END CONVERSATION]

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
    
        [CONTEXT FROM DATABASE]
        {context}
        
        Current Question: {query}
        Answer:"""
        
        # Inference
        messages = [{"role": "user", "content": prompt}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_answer = response.split("assistant\n")[-1] if "assistant\n" in response else response

        return final_answer, context