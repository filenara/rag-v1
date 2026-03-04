import os
import re
import pickle
import logging
import threading
import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info

from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.ste100_guard import STE100Guard
from src.utils import load_prompts, load_config

logger = logging.getLogger(__name__)


class RAGEngine:
    _inference_lock = threading.Lock()

    def __init__(self):
        logger.info("RAGEngine baslatiliyor (Conversation Aware + RRF Mode)...")
        self.db = DatabaseManager()
        self.llm_manager = LLMManager()
        self.guard = STE100Guard()
        self.prompts = load_prompts()
        self.config = load_config()

        self.retrieval_cfg = self.config.get("retrieval", {})
        self.n_results = self.retrieval_cfg.get("n_results", 10)
        self.k_constant = self.retrieval_cfg.get("k_constant", 60)
        self.top_k_rerank = self.retrieval_cfg.get("top_k_rerank", 5)

        self.bm25: Optional[BM25Okapi] = None
        self.ids: List[str] = []
        self.doc_text_map: Dict[str, str] = {}
        self.doc_meta_map: Dict[str, dict] = {}
        
        self._load_bm25_cache()

    def _load_bm25_cache(self) -> None:
        bm25_cache_path = self.config.get("vector_db", {}).get(
            "bm25_cache_path", "data/bm25_cache.pkl"
        )
        if os.path.exists(bm25_cache_path):
            logger.info("BM25 indeksi diskten RAM'e yukleniyor...")
            try:
                with open(bm25_cache_path, "rb") as f:
                    cache = pickle.load(f)
                
                self.bm25 = cache.get("bm25")
                self.ids = cache.get("ids", [])
                documents = cache.get("documents", [])
                metadatas = cache.get("metadatas", [])
                
                self.doc_text_map = {
                    doc_id: doc for doc_id, doc in zip(self.ids, documents)
                }
                self.doc_meta_map = {
                    doc_id: meta for doc_id, meta in zip(self.ids, metadatas)
                }
                logger.info("BM25 Cache basariyla hafizaya alindi.")
            except Exception as e:
                logger.error(f"BM25 cache okuma hatasi: {e}")
        else:
            logger.warning("BM25 cache dosyasi bulunamadi.")

    def reload_cache(self) -> None:
        self._load_bm25_cache()

    def refine_answer(
        self, draft_text: str, feedback_list: list, model: Any, processor: Any
    ) -> str:
        logger.info("[Self-Correction] STE100 ihlalleri duzeltiliyor...")
        
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
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], padding=True, return_tensors="pt"
        ).to(model.device)
        
        with self._inference_lock:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        raw_text = (
            response.split("assistant\n")[-1] if "assistant\n" in response else response
        )
        
        final_answer = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", final_answer, flags=re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            
        return final_answer

    def _construct_system_prompt(self, use_ste100: bool = False) -> str:
        if use_ste100:
            persona = self.prompts.get("system_persona", "You are a helpful assistant.")
            rules = self.prompts.get("ste100_rules", "")
            return f"{persona}\n\n---\n{rules}"
        
        return self.prompts.get("system_persona_standard", "You are a technical assistant.")

    def _is_feedback_intent(self, query: str) -> bool:
        feedback_words = [
            "yanlis", "hatali", "tekrar bak", "duzelt", 
            "wrong", "incorrect", "re-examine", "look again"
        ]
        return any(word in query.lower() for word in feedback_words)

    def _format_history(self, history: list) -> str:
        if not history:
            return "No previous conversation."
        formatted = ""
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content")).replace("\n", " ")
            formatted += f"{role}: {content}\n"
        return formatted

    def search_and_answer(
        self, 
        query: str, 
        collection_name: str, 
        history: list = None, 
        use_ste100: bool = False, 
        strict_mode: bool = False
    ) -> Tuple[str, str, bool, bool, list]:
        
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
            logger.info("[Niyet] Feedback algilandi. Onceki baglam inceleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and "context_text" in turn:
                    context_text = turn["context_text"]
                    break
            
            if not context_text:
                logger.warning("[Uyari] Gecmiste context bulunamadi, yeni arama yapiliyor...")
                is_feedback = False 
            else:
                intent_instruction = (
                    "The user indicated the previous answer was incorrect. "
                    "Carefully re-examine the provided context."
                )
        
        if not is_feedback:
            logger.info(f"[Niyet] Yeni arama: {query}")
            col = self.db.get_collection(collection_name)
            
            doc_scores = {}
            
            with self._inference_lock:
                q_vec = embedder.encode([query], normalize_embeddings=True).tolist()
                
            vec_res = col.query(query_embeddings=q_vec, n_results=self.n_results)
            
            if vec_res["ids"] and vec_res["ids"][0]:
                for rank, doc_id in enumerate(vec_res["ids"][0]):
                    if doc_id not in doc_scores: 
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1 / (self.k_constant + rank)
                    
            if self.bm25:
                bm25_scores = self.bm25.get_scores(query.lower().split(" "))
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:self.n_results]
                
                for rank, idx in enumerate(top_bm25_indices):
                    doc_id = self.ids[idx]
                    if doc_id not in doc_scores: 
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1 / (self.k_constant + rank)
                    
            sorted_candidates = sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[:self.top_k_rerank]
            
            if not sorted_candidates:
                return "Ilgili sonuc bulunamadi.", "", True, False, []
            
            candidates_text = [
                self.doc_text_map.get(item[0], "") for item in sorted_candidates
            ]
            candidates_meta = [
                self.doc_meta_map.get(item[0], {}) for item in sorted_candidates
            ]
            
            pairs = [[query, txt] for txt in candidates_text if txt and str(txt).strip()]
            
            if not pairs:
                logger.warning("Reranker icin gecerli metin cifti olusturulamadi.")
                return "Icerik analizine uygun metin bulunamadi.", "", True, False, []
                
            with self._inference_lock:
                scores = reranker.predict(pairs)
            
            if scores is None or len(scores) == 0:
                logger.warning("Reranker gecerli bir skor uretmedi.")
                return "Arama sonuclari siralanamadi.", "", True, False, []

            best_idx = np.argmax(scores)
            context_text = candidates_text[best_idx]
            best_meta = candidates_meta[best_idx]

            image_path = best_meta.get("image_path", "")
            if image_path and os.path.exists(image_path):
                logger.info(f"Gorsel baglam yukleniyor: {image_path}")
                try:
                    input_image = Image.open(image_path)
                except Exception as e:
                    logger.error(f"Resim yukleme hatasi: {e}")

        template = self.prompts.get(
            "response_template", "Context: {context_text}\nQuestion: {query}"
        )
        
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
        
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with self._inference_lock:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        raw_text = (
            response.split("assistant\n")[-1] if "assistant\n" in response else response
        )

        final_text = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", final_text, flags=re.DOTALL)
        if answer_match:
            final_text = answer_match.group(1).strip()

        is_compliant = True
        was_corrected = False
        feedback_report = []

        if use_ste100:
            logger.info("STE100 kurallari denetleniyor...")
            is_compliant, feedback_report = self.guard.analyze_and_report(final_text)
            
            if not is_compliant and strict_mode:
                max_retries = 2
                retries = 0
                current_text = final_text
                
                while not is_compliant and retries < max_retries:
                    retries += 1
                    logger.info(f"Ihlaller duzeltiliyor... (Deneme {retries}/{max_retries})")
                    
                    previous_text = current_text
                    current_text = self.refine_answer(
                        current_text, feedback_report, model, processor
                    )
                    
                    if current_text.strip() == previous_text.strip():
                        break
                        
                    is_compliant, feedback_report = self.guard.analyze_and_report(
                        current_text
                    )
                    
                final_text = current_text
                was_corrected = True

        source_info = ""
        if best_meta:
            src_name = os.path.basename(best_meta.get("source", "Unknown"))
            pg_num = best_meta.get("page", "?")
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image:
                source_info += " [Gorsel Analiz Yapildi]"

        return final_text + source_info, context_text, is_compliant, was_corrected, feedback_report