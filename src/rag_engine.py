import torch
import numpy as np
import os
import pickle
from PIL import Image
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info
from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.utils import load_prompts, load_config


class RAGEngine:
    def refine_answer(self, draft_text, feedback_list, model, processor):
        """
        Guard'dan gelen hata raporunu ve YAML'daki promptu kullanarak
        Qwen-VL'e metni yeniden ve hatasiz yazdirir.
        """
        print("[Self-Correction] STE100 ihlalleri duzeltiliyor...")
        
        feedback_str = "\n".join(feedback_list)
        template = self.prompts.get(
            "self_correction_prompt",
            "Fix this:\n{draft_answer}\nErrors:\n{feedback_report}"
        )
        
        prompt_text = template.format(
            draft_answer=draft_text,
            feedback_report=feedback_str
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_answer = response.split("assistant\n")[-1] if "assistant\n" in response else response
        
        return final_answer
    
    def __init__(self):
        print("RAGEngine Baslatiliyor (Conversation Aware + RRF Mode)...")
        self.db = DatabaseManager()
        self.llm_manager = LLMManager()
        self.prompts = load_prompts()

    def _construct_system_prompt(self, use_ste100=False):
        if use_ste100:
            persona = self.prompts.get("system_persona", "You are a helpful assistant.")
            rules = self.prompts.get("ste100_rules", "")
            return f"{persona}\n\n---\n{rules}"
        else:
            persona = self.prompts.get("system_persona_standard", "You are a technical assistant.")
            return persona

    def _is_feedback_intent(self, query):
        feedback_words = ["yanlis", "hatali", "tekrar bak", "duzelt", "wrong", "incorrect", "re-examine", "look again"]
        return any(word in query.lower() for word in feedback_words)

    def _format_history(self, history):
        if not history:
            return "No previous conversation."
        formatted = ""
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content")).replace("\n", " ")
            formatted += f"{role}: {content}\n"
        return formatted

    def search_and_answer(self, query, collection_name, history=None, user_image=None, use_ste100=False):
        if history is None:
            history = []

        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()
        model, processor = self.llm_manager.load_vision_model()
        
        is_feedback = self._is_feedback_intent(query)
        intent_instruction = "Answer based on the context and previous conversation."
        
        context_text = ""
        best_meta = {}
        input_image = None
        
        history_text = self._format_history(history)

        if is_feedback and len(history) > 0:
            print("[Niyet] Feedback algilandi. Onceki baglam tekrar inceleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context_text" in turn:
                    context_text = turn["context_text"]
                    break
            
            if not context_text:
                print("[Uyari] Gecmiste context bulunamadi, yeni arama yapiliyor...")
                is_feedback = False 
            else:
                intent_instruction = "The user indicated the previous answer was incorrect. Carefully re-examine the provided context."
        
        if not is_feedback:
            print(f"[Niyet] Yeni arama: {query}")
            col = self.db.get_collection(collection_name)
            
            search_queries = [query]
            
            config_data = load_config()
            bm25_cache_path = config_data.get("vector_db", {}).get("bm25_cache_path", "data/bm25_cache.pkl")
            
            if os.path.exists(bm25_cache_path):
                print("BM25 Indeksi diskten (cache) yukleniyor...")
                with open(bm25_cache_path, "rb") as f:
                    cache = pickle.load(f)
                bm25 = cache["bm25"]
                ids = cache["ids"]
                documents = cache["documents"]
                metadatas = cache["metadatas"]
            else:
                print("[Uyari] BM25 Cache bulunamadi, anlik indeks olusturuluyor...")
                all_docs_data = col.get() 
                documents = all_docs_data["documents"]
                ids = all_docs_data["ids"]
                metadatas = all_docs_data["metadatas"]
                
                if not documents:
                    return "Veritabani bos.", ""
                tokenized_corpus = [doc.lower().split(" ") for doc in documents]
                bm25 = BM25Okapi(tokenized_corpus)

            doc_scores = {}
            doc_meta_map = {} 
            doc_text_map = {}
            k_constant = 60
            
            for q in search_queries:
                q_vec = embedder.encode([q], normalize_embeddings=True).tolist()
                vec_res = col.query(query_embeddings=q_vec, n_results=10)
                
                if vec_res["ids"]:
                    for rank, doc_id in enumerate(vec_res["ids"][0]):
                        if doc_id not in doc_scores: 
                            doc_scores[doc_id] = 0
                            try:
                                idx = ids.index(doc_id)
                                doc_meta_map[doc_id] = metadatas[idx]
                                doc_text_map[doc_id] = documents[idx]
                            except ValueError:
                                continue
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
            if not sorted_candidates:
                return "Ilgili sonuc bulunamadi.", ""
            
            candidates_text = [doc_text_map[item[0]] for item in sorted_candidates]
            candidates_meta = [doc_meta_map[item[0]] for item in sorted_candidates]
            pairs = [[query, txt] for txt in candidates_text]
            scores = reranker.predict(pairs)
            best_idx = np.argmax(scores)
            context_text = candidates_text[best_idx]
            best_meta = candidates_meta[best_idx]

            if user_image is not None:
                print("Kullanici Gorseli Yukleniyor (Veritabani gorseli ezildi)...")
                input_image = user_image
            else:
                image_path = best_meta.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    print(f"Gorsel Baglam Yukleniyor: {image_path}")
                    try:
                        input_image = Image.open(image_path)
                    except Exception as e:
                        print(f"Resim yukleme hatasi: {e}")

        template = self.prompts.get("response_template", "Context: {context_text}\nQuestion: {query}")
        
        user_prompt_text = template.format(
            intent_instruction=intent_instruction,
            history_text=history_text,
            context_text=context_text,
            query=query
        )

        user_content = []
        if input_image:
            user_content.append({"type": "image", "image": input_image})
            user_prompt_text = "[IMAGE ATTACHED TO THIS MESSAGE]\n" + user_prompt_text
            
        user_content.append({"type": "text", "text": user_prompt_text})

        system_instruction = self._construct_system_prompt(use_ste100)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
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

        source_info = ""
        if best_meta:
            src_name = os.path.basename(best_meta.get("source", "Unknown"))
            pg_num = best_meta.get("page", "?")
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image:
                source_info += " [Gorsel Analiz Yapildi]"

        return final_answer + source_info, context_text