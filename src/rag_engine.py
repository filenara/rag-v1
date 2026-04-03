import os
import re
import json
import pickle
import logging
import base64
from io import BytesIO
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image
from rank_bm25 import BM25Okapi

from src.database import DatabaseManager
from src.llm_manager import LLMManager
from src.ste100_guard import STE100Guard
from src.utils import load_prompts, load_config

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        logger.info("RAGEngine baslatiliyor (API Client Mode)...")
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
        self.source_to_indices: Dict[str, List[int]] = {}
        
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
                
                self.source_to_indices = {}
                for idx, doc_id in enumerate(self.ids):
                    meta = self.doc_meta_map.get(doc_id, {})
                    source = meta.get("source", "").lower()
                    if source not in self.source_to_indices:
                        self.source_to_indices[source] = []
                    self.source_to_indices[source].append(idx)
                    
                logger.info("BM25 Cache basariyla hafizaya alindi.")
            except Exception as e:
                logger.error("BM25 cache okuma hatasi: %s", e, exc_info=True)
        else:
            logger.warning("BM25 cache dosyasi bulunamadi.")

    def reload_cache(self) -> None:
        self._load_bm25_cache()

    def _generate_api(self, messages: list, max_tokens: int) -> str:
        api_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, list):
                api_content = []
                for item in content:
                    if item.get("type") == "image" and "image" in item:
                        img = item["image"]
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        api_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        })
                    else:
                        api_content.append(item)
                api_messages.append({"role": role, "content": api_content})
            else:
                api_messages.append(msg)
                
        vision_client = self.llm_manager.get_vision_client()
        return vision_client.generate(api_messages, max_tokens=max_tokens)

    def _clean_output(self, raw_text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL).strip()
        match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _analyze_and_rewrite_query(self, query: str, history: list) -> Tuple[str, str]:
        history_text = ""
        if history:
            for msg in history[-3:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                if isinstance(content, str):
                    history_text += f"{role}: {content}\n"
                
        prompt_template = self.prompts.get(
            "query_rewrite_prompt", 
            "Task:\n1. Rewrite the query.\nFormat:\n{{\"standalone_query\": \"string\", \"source_filter\": \"string\"}}\nHistory:\n{history_text}\nLatest Query: {query}"
        )
        prompt = prompt_template.format(history_text=history_text, query=query)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            raw_response = self._generate_api(messages, max_tokens=200)
            cleaned_response = self._clean_output(raw_response)
            
            cleaned_response = cleaned_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
                
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            cleaned_response = cleaned_response.strip()
            
            try:
                parsed = json.loads(cleaned_response)
                standalone = parsed.get("standalone_query", query)
                filter_val = parsed.get("source_filter", "")
                return standalone, filter_val
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*?\}", cleaned_response, flags=re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        standalone = parsed.get("standalone_query", query)
                        filter_val = parsed.get("source_filter", "")
                        return standalone, filter_val
                    except json.JSONDecodeError:
                        logger.warning("Regex ile bulunan JSON ayristirilamadi. Ham metin kullanilacak.")
                        return query, ""
                else:
                    logger.warning("JSON yapisi bulunamadi. Ham metin kullanilacak.")
                    return query, ""
        except Exception as e:
            logger.error("Sorgu analizi hatasi: %s", e, exc_info=True)
            return query, ""

    def _route_intent(self, query: str, history: list, use_ste100: bool) -> str:
        history_text = ""
        if history:
            for msg in history[-3:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                if isinstance(content, str):
                    history_text += f"{role}: {content}\n"

        if use_ste100:
            prompt_template = self.prompts.get(
                "intent_routing_ste100",
                "Classify category.\nHistory:\n{history_text}\nMessage: {query}\nCategory:"
            )
            valid_intents = ["PROCEDURE", "DESCRIPTIVE", "SAFETY", "REVISION"]
            default_intent = "PROCEDURE"
        else:
            prompt_template = self.prompts.get(
                "intent_routing_standard",
                "Classify category.\nHistory:\n{history_text}\nMessage: {query}\nCategory:"
            )
            valid_intents = ["QA", "REVISION", "CHITCHAT"]
            default_intent = "QA"
        
        prompt = prompt_template.format(history_text=history_text, query=query)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        try:
            raw_intent = self._generate_api(messages, max_tokens=10)
            intent = self._clean_output(raw_intent).upper().strip()
            
            for vi in valid_intents:
                if vi in intent:
                    return vi
            return default_intent
        except Exception as e:
            logger.error("Niyet analizi hatasi: %s", e, exc_info=True)
            return default_intent

    def refine_answer(self, draft_text: str, feedback_list: list) -> str:
        logger.info("[Self-Correction] STE100 ihlalleri API uzerinden duzeltiliyor...")
        
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
        
        try:
            raw_text = self._generate_api(messages, max_tokens=512)
            return self._clean_output(raw_text)
        except Exception as e:
            logger.error("Duzeltme sirasinda API hatasi: %s", e, exc_info=True)
            return draft_text

    def search_and_answer(
        self, 
        query: str, 
        collection_name: str, 
        history: list = None, 
        use_ste100: bool = False, 
        strict_mode: bool = False,
        template_type: str = "General"
    ) -> Tuple[str, str, bool, bool, list]:
        
        if history is None:
            history = []

        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()
        
        logger.info("Sorgu baglam ve metaveri tespiti icin analiz ediliyor...")
        standalone_query, source_filter = self._analyze_and_rewrite_query(query, history)
        logger.info("Yeniden yazilan sorgu: %s", standalone_query)
        if source_filter:
            logger.info("Tespit edilen hedef dokuman: %s", source_filter)
            
        logger.info("Niyet analizi yapiliyor...")
        intent = self._route_intent(standalone_query, history, use_ste100)
        logger.info("Tespit edilen niyet: %s", intent)
        
        if intent in ["PROCEDURE", "DESCRIPTIVE", "SAFETY"]:
            use_ste100 = True
            template_type = intent.capitalize()
            logical_intent = "SEARCH"
        elif intent == "QA":
            logical_intent = "SEARCH"
        elif intent == "REVISION":
            logical_intent = "REVISION"
        else:
            logical_intent = "CHAT"
            
        context_text = ""
        best_meta = {}
        input_image = None
        last_assistant_text = ""
        
        if (logical_intent in ["CHAT", "REVISION"]) and history:
            logger.info("Arama atlandi. Onceki baglam ve metin yukleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant":
                    context_text = turn.get("context_text", "")
                    last_assistant_text = turn.get("content", "")
                    break
            
            if not last_assistant_text and logical_intent == "REVISION":
                logger.warning("Gecmis asistan metni bulunamadi. QA moduna donuluyor.")
                logical_intent = "SEARCH"
        
        if logical_intent == "SEARCH" or not history:
            col = self.db.get_collection(collection_name)
            doc_scores = {}
            q_vec = embedder.encode([standalone_query], normalize_embeddings=True)
            
            where_clause = None
            if source_filter:
                where_clause = {"source": {"$contains": source_filter}}
                
            def execute_retrieval(current_where):
                scores_dict = {}
                vec_res = col.query(
                    query_embeddings=q_vec, 
                    n_results=self.n_results, 
                    where=current_where
                )
                
                if vec_res["ids"] and vec_res["ids"][0]:
                    for rank, doc_id in enumerate(vec_res["ids"][0]):
                        if doc_id not in scores_dict: 
                            scores_dict[doc_id] = 0
                        scores_dict[doc_id] += 1 / (self.k_constant + rank)
                        
                if self.bm25:
                    bm25_scores = np.array(self.bm25.get_scores(standalone_query.lower().split(" ")))
                    if current_where and source_filter:
                        filter_lower = source_filter.lower()
                        valid_indices = []
                        for src, indices in self.source_to_indices.items():
                            if filter_lower in src:
                                valid_indices.extend(indices)
                        
                        mask = np.ones(len(bm25_scores), dtype=bool)
                        if valid_indices:
                            mask[valid_indices] = False
                        bm25_scores[mask] = -1.0
                                
                    top_bm25_indices = np.argsort(bm25_scores)[::-1]
                    rank = 0
                    for idx in top_bm25_indices:
                        if rank >= self.n_results or bm25_scores[idx] < 0:
                            break
                        doc_id = self.ids[idx]
                        if doc_id not in scores_dict: 
                            scores_dict[doc_id] = 0
                        scores_dict[doc_id] += 1 / (self.k_constant + rank)
                        rank += 1
                return scores_dict
                
            doc_scores = execute_retrieval(where_clause)
            
            sorted_candidates = sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[:self.top_k_rerank]
            
            if not sorted_candidates:
                return "Ilgili sonuc bulunamadi.", "", True, False, []
            
            valid_candidates = []
            for item in sorted_candidates:
                doc_id = item[0]
                txt = self.doc_text_map.get(doc_id, "")
                meta = self.doc_meta_map.get(doc_id, {})
                if txt and str(txt).strip():
                    valid_candidates.append({
                        "text": txt,
                        "meta": meta
                    })
            
            pairs = [[standalone_query, cand["text"]] for cand in valid_candidates]
            
            if pairs:
                scores = reranker.predict(pairs)
                
                top_indices = np.argsort(scores)[::-1][:3]
                
                context_texts = []
                best_meta = valid_candidates[top_indices[0]]["meta"]
                image_path = best_meta.get("image_path", "")
                
                for idx in top_indices:
                    context_texts.append(valid_candidates[idx]["text"])
                
                context_text = "\n\n".join(context_texts)

                if image_path and os.path.exists(image_path):
                    try:
                        input_image = Image.open(image_path)
                    except Exception as e:
                        logger.error("Resim yukleme hatasi: %s", e, exc_info=True)

        messages = []
        
        history_text_formatted = ""
        for msg in history[-3:]:
            role_name = "User" if msg.get("role") == "user" else "Assistant"
            content_val = msg.get("content", "")
            if isinstance(content_val, str) and content_val.strip():
                history_text_formatted += f"{role_name}: {content_val}\n"
                
        if not history_text_formatted.strip():
            history_text_formatted = "No previous conversation."

        if logical_intent == "REVISION":
            logger.info("Revizyon promptu hazirlaniyor...")
            system_instruction = self.prompts.get(
                "system_persona_revision", 
                "You are a technical assistant. Revise the document."
            )
            messages.append({"role": "system", "content": system_instruction})
            
            revision_template = self.prompts.get(
                "self_correction_prompt", 
                "Fix this:\n{draft_answer}\nErrors/Feedback:\n{feedback_report}"
            )
            user_prompt_text = revision_template.format(
                draft_answer=last_assistant_text,
                feedback_report=query
            )
        else:
            if use_ste100 and logical_intent == "SEARCH":
                logger.info("Dinamik STE100 promptu uretiliyor. Format: %s", template_type)
                persona = self.prompts.get("system_persona", "You are a technical assistant.")
                dynamic_rules = self.guard.build_injection_prompt(context_text, template_type)
                system_instruction = f"{persona}\n\n{dynamic_rules}"
            else:
                system_instruction = self.prompts.get("system_persona_standard", "You are a technical assistant.")

            messages.append({"role": "system", "content": system_instruction})
            
            base_template = self.prompts.get(
                "response_template", 
                "Context Information:\n{context_text}\n\nUser Question/Request:\n{query}"
            )
            user_prompt_text = base_template.format(
                history_text=history_text_formatted.strip(),
                context_text=context_text,
                query=query
            )

        user_content = []
        if input_image:
            user_content.append({"type": "image", "image": input_image})
            user_prompt_text = "[IMAGE ATTACHED TO THIS MESSAGE]\n" + user_prompt_text
            
        user_content.append({"type": "text", "text": user_prompt_text})
        messages.append({"role": "user", "content": user_content})
        
        try:
            raw_text = self._generate_api(messages, max_tokens=2048)
            final_text = self._clean_output(raw_text)
        except Exception as e:
            logger.error("API cagirilirken hata: %s", e, exc_info=True)
            final_text = "Cevap uretilirken yerel modelde bir hata olustu."

        is_compliant = True
        was_corrected = False
        feedback_report = []

        if use_ste100 and logical_intent == "SEARCH":
            logger.info("STE100 kurallari denetleniyor...")
            is_compliant, feedback_report = self.guard.analyze_and_report(final_text)
            
            if not is_compliant and strict_mode:
                max_retries = 2
                retries = 0
                current_text = final_text
                
                while not is_compliant and retries < max_retries:
                    retries += 1
                    logger.info("Ihlaller duzeltiliyor... (Deneme %s/%s)", retries, max_retries)
                    previous_text = current_text
                    current_text = self.refine_answer(current_text, feedback_report)
                    
                    if current_text.strip() == previous_text.strip():
                        break
                        
                    is_compliant, feedback_report = self.guard.analyze_and_report(current_text)
                    
                final_text = current_text
                was_corrected = True

        source_info = ""
        if best_meta and logical_intent == "SEARCH":
            src_name = os.path.basename(best_meta.get("source", "Unknown"))
            pg_num = best_meta.get("page", "?")
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image:
                source_info += " [Gorsel Analiz Yapildi]"

        return final_text + source_info, context_text, is_compliant, was_corrected, feedback_report