import torch
import numpy as np
import os
from PIL import Image
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info
from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.utils import load_prompts

class RAGEngine:
    def __init__(self):
        print("⚡ RAGEngine Başlatılıyor (Conversation Aware + Multi-Query RRF Mode)...")
        self.db = DatabaseManager()
        self.llm_manager = LLMManager()
        # YAML'dan promptları yükle
        self.prompts = load_prompts()

    def _construct_system_prompt(self):
        """YAML'dan gelen Persona ve STE100 Kurallarını birleştirir."""
        persona = self.prompts.get('system_persona', 'You are a helpful assistant.')
        rules = self.prompts.get('ste100_rules', '')
        return f"{persona}\n\n---\n{rules}"

    def _is_feedback_intent(self, query):
        feedback_words = ["yanlış", "hatalı", "tekrar bak", "düzelt", "wrong", "incorrect", "re-examine", "look again"]
        return any(word in query.lower() for word in feedback_words)

    def _format_history(self, history):
        if not history:
            return "No previous conversation."
        formatted = ""
        for turn in history:
            role = "User" if turn['role'] == 'user' else "Assistant"
            content = str(turn['content']).replace('\n', ' ')
            formatted += f"{role}: {content}\n"
        return formatted

    def _generate_multi_queries(self, original_query, model, processor, n=3):
        prompt = f"Question: '{original_query}'. To search for this question in a technical database, write {n} different alternative questions in English. Just list the questions, no numbering."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            raw_queries = output.split(prompt)[-1].strip().split('\n')
            queries = [q.strip() for q in raw_queries if len(q) > 5]
            return [original_query] + queries[:n]
        except Exception as e:
            print(f"⚠️ Sorgu çoğaltma hatası: {e}")
            return [original_query]

    def search_and_answer(self, query, collection_name, history=[]):
        # 1. MODELLERİ ÇAĞIR
        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()
        model, processor = self.llm_manager.load_vision_model()
        
        # 2. NİYET VE BAĞLAM ANALİZİ
        is_feedback = self._is_feedback_intent(query)
        intent_instruction = "Answer based on the context and previous conversation." # Varsayılan instruction
        
        context_text = ""
        best_meta = {}
        input_image = None
        
        # Geçmişi formatla
        history_text = self._format_history(history)

        # A) Feedback Durumu
        if is_feedback and len(history) > 0:
            print("[Niyet] Feedback algılandı. Önceki bağlam tekrar inceleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context" in turn:
                    context_text = turn["context"]
                    break
            
            if not context_text:
                print("[Uyarı] Geçmişte context bulunamadı, yeni arama yapılıyor...")
                is_feedback = False 
            else:
                intent_instruction = "The user indicated the previous answer was incorrect. Carefully re-examine the provided context."
        
        # B) Yeni Arama Durumu
        if not is_feedback:
            print(f"[Niyet] Yeni arama: {query}")
            col = self.db.get_collection(collection_name)
            search_queries = self._generate_multi_queries(query, model, processor)
            
            # ... (Buradaki RRF ve Search mantığınız aynı kalıyor) ...
            # Kısalık olması için RRF kod bloğunu özet geçiyorum, sizin önceki kodunuz buraya gelecek
            # --- RRF SİSTEMİ BAŞLANGICI ---
            all_docs_data = col.get() 
            documents = all_docs_data['documents']
            ids = all_docs_data['ids']
            metadatas = all_docs_data['metadatas']
            
            if not documents: return "Veritabanı boş.", ""
            tokenized_corpus = [doc.lower().split(" ") for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = {}
            doc_meta_map = {} 
            doc_text_map = {}
            k_constant = 60
            for q in search_queries:
                q_vec = embedder.encode([q]).tolist()
                vec_res = col.query(query_embeddings=q_vec, n_results=10)
                if vec_res['ids']:
                    for rank, doc_id in enumerate(vec_res['ids'][0]):
                        if doc_id not in doc_scores: 
                            doc_scores[doc_id] = 0
                            try:
                                idx = ids.index(doc_id)
                                doc_meta_map[doc_id] = metadatas[idx]
                                doc_text_map[doc_id] = documents[idx]
                            except: continue
                        doc_scores[doc_id] += 1 / (k_constant + rank)
                bm25_scores = bm25.get_scores(q.lower().split(" "))
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
                for rank, idx in enumerate(top_bm25_indices):
                    doc_id = ids[idx]
                    if doc_id not in doc_scores: 
                        doc_scores[doc_id] = 0
                        doc_meta_map[doc_id] = metadatas[idx]
                        doc_text_map[doc_id] = documents[idx]
                    doc_scores[doc_id] += 1 / (k_constant + rank)
            sorted_candidates = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:5]
            if not sorted_candidates: return "İlgili sonuç bulunamadı.", ""
            candidates_text = [doc_text_map[item[0]] for item in sorted_candidates]
            candidates_meta = [doc_meta_map[item[0]] for item in sorted_candidates]
            pairs = [[query, txt] for txt in candidates_text]
            scores = reranker.predict(pairs)
            best_idx = np.argmax(scores)
            context_text = candidates_text[best_idx]
            best_meta = candidates_meta[best_idx]
            # --- RRF SONU ---

            # Görsel Yükleme
            image_path = best_meta.get("image_path", "")
            if image_path and os.path.exists(image_path):
                print(f" Görsel Bağlam Yükleniyor: {image_path}")
                try:
                    input_image = Image.open(image_path)
                except Exception as e:
                    print(f"Resim yükleme hatası: {e}")

        # --- CEVAP ÜRETİMİ (Şablon Doldurma) ---
        
        # 1. Şablonu Yükle
        template = self.prompts.get('response_template', "Context: {context_text}\nQuestion: {query}")
        
        # 2. Değişkenleri Şablona Göm
        # Burada template içindeki {intent_instruction}, {history_text} vb. yerlerini dolduruyoruz.
        user_prompt_text = template.format(
            intent_instruction=intent_instruction,
            history_text=history_text,
            context_text=context_text,
            query=query
        )

        # 3. Mesaj Yapısını Kur
        user_content = []
        if input_image:
            user_content.append({"type": "image", "image": input_image})
            # Sizin istediğiniz görsel uyarısı:
            user_prompt_text = "[IMAGE ATTACHED TO THIS MESSAGE]\n" + user_prompt_text
            
        user_content.append({"type": "text", "text": user_prompt_text})

        # 4. Sistem Promptunu Hazırla (STE100 Rules)
        system_instruction = self._construct_system_prompt()

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
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
        final_answer = response.split("assistant\n")[-1] if "assistant\n" in response else response

        # Kaynak Bilgisi Ekle
        source_info = ""
        if best_meta:
            src_name = os.path.basename(best_meta.get('source', 'Unknown'))
            pg_num = best_meta.get('page', '?')
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image: source_info += " [Görsel Analiz Yapıldı]"

        return final_answer + source_info, context_text