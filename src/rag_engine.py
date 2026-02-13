import torch
import numpy as np
import os
from PIL import Image
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info
from src.database import DatabaseManager
from src.llm_manager import LLMManager

class RAGEngine:
    def __init__(self):
        """
        Lazy Mode: Modeller burada YÜKLENMEZ. Sadece yönetici çağrılır.
        """
        print("⚡ RAGEngine Başlatılıyor (Conversation Aware + Multi-Query RRF Mode)...")
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
            content = str(turn['content']).replace('\n', ' ') # Satır sonlarını temizle
            formatted += f"{role}: {content}\n"
        return formatted

    def _generate_multi_queries(self, original_query, model, processor, n=3):
        """
        Soruyu farklı açılardan tekrar sorarak arama kapsamını genişletir.
        """
        prompt = f"Question: '{original_query}'. To search for this question in a technical database, write {n} different alternative questions in Turkish. Just list the questions, no numbering."
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], padding=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Cevabı temizle ve listeye çevir
            raw_queries = output.split(prompt)[-1].strip().split('\n')
            queries = [q.strip() for q in raw_queries if len(q) > 5]
            return [original_query] + queries[:n] # Orijinal soruyu da ekle
        except Exception as e:
            print(f"⚠️ Sorgu çoğaltma hatası: {e}")
            return [original_query]

    def search_and_answer(self, query, collection_name, history=[]):
        """
        Arama ve Cevaplama fonksiyonu (RRF + Multi-Query + History Aware).
        """
        
        # --- 1. MODELLERİ ÇAĞIR ---
        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()
        model, processor = self.llm_manager.load_vision_model()
        
        # --- 2. NİYET VE BAĞLAM ANALİZİ ---
        is_feedback = self._is_feedback_intent(query)
        intent_instruction = "Answer based on the context and previous conversation."
        
        context_text = ""
        best_meta = {}
        input_image = None
        
        # Geçmişi formatla
        history_text = self._format_history(history)

        # A) Feedback (Geri Bildirim) Durumu: Eski bağlamı kullan
        if is_feedback and len(history) > 0:
            print("[Niyet] Feedback algılandı. Önceki bağlam tekrar inceleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context" in turn:
                    context_text = turn["context"]
                    # Eğer önceki cevapta görsel varsa onu bulmaya çalışabiliriz (burası opsiyonel geliştirilebilir)
                    break
            
            if not context_text:
                print("[Uyarı] Geçmişte context bulunamadı, yeni arama yapılıyor...")
                is_feedback = False 
            else:
                intent_instruction = "The user indicated the previous answer was incorrect. Carefully re-examine the provided context."
        
        # B) Yeni Arama Durumu: Multi-Query + RRF
        if not is_feedback:
            print(f"[Niyet] Yeni arama: {query}")
            col = self.db.get_collection(collection_name)
            
            # 1. Multi-Query Üretimi
            search_queries = self._generate_multi_queries(query, model, processor)
            print(f"🔍 Genişletilmiş Sorgular: {search_queries}")

            # 2. Verileri Hazırla (BM25 için)
            all_docs_data = col.get() 
            documents = all_docs_data['documents']
            ids = all_docs_data['ids']
            metadatas = all_docs_data['metadatas']
            
            if not documents: return "Veritabanı boş.", ""

            tokenized_corpus = [doc.lower().split(" ") for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)

            # 3. RRF (Reciprocal Rank Fusion) Döngüsü
            doc_scores = {} # {doc_id: score}
            doc_meta_map = {} # {doc_id: meta}
            doc_text_map = {} # {doc_id: text}
            k_constant = 60
            
            for q in search_queries:
                # Vektör Arama
                q_vec = embedder.encode([q]).tolist()
                vec_res = col.query(query_embeddings=q_vec, n_results=10)
                
                if vec_res['ids']:
                    for rank, doc_id in enumerate(vec_res['ids'][0]):
                        if doc_id not in doc_scores: 
                            doc_scores[doc_id] = 0
                            # Map'leri doldur
                            try:
                                idx = ids.index(doc_id)
                                doc_meta_map[doc_id] = metadatas[idx]
                                doc_text_map[doc_id] = documents[idx]
                            except: continue
                        
                        doc_scores[doc_id] += 1 / (k_constant + rank)
                
                # BM25 Arama
                bm25_scores = bm25.get_scores(q.lower().split(" "))
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
                
                for rank, idx in enumerate(top_bm25_indices):
                    doc_id = ids[idx]
                    if doc_id not in doc_scores: 
                        doc_scores[doc_id] = 0
                        doc_meta_map[doc_id] = metadatas[idx]
                        doc_text_map[doc_id] = documents[idx]
                        
                    doc_scores[doc_id] += 1 / (k_constant + rank)

            # 4. En iyi adayları seç
            sorted_candidates = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:5]
            
            if not sorted_candidates: return "İlgili sonuç bulunamadı.", ""

            candidates_text = [doc_text_map[item[0]] for item in sorted_candidates]
            candidates_meta = [doc_meta_map[item[0]] for item in sorted_candidates]

            # 5. Reranking (Cross-Encoder ile son sıralama)
            pairs = [[query, txt] for txt in candidates_text]
            scores = reranker.predict(pairs)
            best_idx = np.argmax(scores)
            
            context_text = candidates_text[best_idx]
            best_meta = candidates_meta[best_idx]
            
            print(f"🎯 En iyi eşleşme skoru: {scores[best_idx]:.4f} | Kaynak: {best_meta.get('source')}")

            # 6. Görsel Yükleme (Asset Store)
            image_path = best_meta.get("image_path", "")
            if image_path and os.path.exists(image_path):
                print(f"🖼️ Görsel Bağlam Yükleniyor: {image_path}")
                try:
                    input_image = Image.open(image_path)
                except Exception as e:
                    print(f"Resim yükleme hatası: {e}")

        # --- 4. CEVAP ÜRETİMİ (SİZİN ORİJİNAL PROMPT) ---
        
        # Prompt metnini değiştirmeden koruyoruz.
        # Sadece {context} ve {query} değişkenleri yukarıdaki yeni mantıkla doldu.
        prompt_text = f"""Role: Senior Technical Systems Engineer.
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
        {context_text}
        
        Current Question: {query}
        Answer:"""
        
        # Mesaj içeriğini oluştur (Görsel varsa ekle, yoksa sadece metin)
        user_content = []
        
        if input_image:
            user_content.append({"type": "image", "image": input_image})
            # Görsel olduğunda promptun başına ufak bir işaret koyuyoruz ki model görseli dikkate alsın
            final_prompt_text = "[IMAGE ATTACHED TO THIS MESSAGE]\n" + prompt_text
        else:
            final_prompt_text = prompt_text
            
        user_content.append({"type": "text", "text": final_prompt_text})

        messages = [{"role": "user", "content": user_content}]
        
        # Inference
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Qwen bazen 'assistant\n' tag'ini de output'a dahil eder, onu temizleyelim
        final_answer = response.split("assistant\n")[-1] if "assistant\n" in response else response

        # Kaynak bilgisini cevabın sonuna ekle (Kullanıcı arayüzü için)
        source_info = ""
        if best_meta:
            src_name = os.path.basename(best_meta.get('source', 'Unknown'))
            pg_num = best_meta.get('page', '?')
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image: source_info += " [Görsel Analiz Yapıldı]"

        return final_answer + source_info, context_text