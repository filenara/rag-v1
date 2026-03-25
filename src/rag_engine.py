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
                logger.error("BM25 cache okuma hatasi: %s", e)
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
            for msg in history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
                
        prompt = (
            "You are a query analysis assistant. Based on the conversation history and the latest user query, "
            "perform two tasks:\n"
            "1. Rewrite the user query into a standalone, fully understandable question.\n"
            "2. Extract the specific document name or file name if the user explicitly mentions one to search within. "
            "If no document is mentioned, leave it empty.\n\n"
            "You MUST respond ONLY with a valid JSON object in the following format:\n"
            "{\n"
            '  "standalone_query": "The rewritten query here",\n'
            '  "source_filter": "extracted document name or empty string"\n'
            "}\n\n"
            f"History:\n{history_text}\n"
            f"Latest Query: {query}\n"
            "JSON Output:"
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            raw_response = self._generate_api(messages, max_tokens=150)
            cleaned_response = self._clean_output(raw_response)
            
            json_match = re.search(r"\{.*\}", cleaned_response, flags=re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                standalone = parsed.get("standalone_query", query)
                filter_val = parsed.get("source_filter", "")
                return standalone, filter_val
            return query, ""
        except Exception as e:
            logger.error("Sorgu analizi (JSON) hatasi: %s", e)
            return query, ""

    def _route_intent(self, query: str) -> str:
        prompt = (
            "Classify the following user message into one of two categories:\n"
            "SEARCH: The user is asking a technical question that requires searching external documents.\n"
            "CHAT: The user is providing feedback, asking to modify a previous answer (e.g. 'make it longer', 'are you sure?'), or chatting.\n"
            "Respond ONLY with the word SEARCH or CHAT.\n\n"
            f"Message: {query}\n"
            "Category:"
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            raw_intent = self._generate_api(messages, max_tokens=10)
            intent = self._clean_output(raw_intent).upper()
            if "CHAT" in intent:
                return "CHAT"
            return "SEARCH"
        except Exception as e:
            logger.error("Niyet analizi hatasi: %s", e)
            return "SEARCH"

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
            logger.error("Duzeltme sirasinda API hatasi: %s", e)
            return draft_text

    def _construct_system_prompt(self, use_ste100: bool = False) -> str:
        if use_ste100:
            persona = self.prompts.get("system_persona", "You are a helpful assistant.")
            rules = self.prompts.get("ste100_rules", "")
            return f"{persona}\n\n---\n{rules}"
        
        return self.prompts.get("system_persona_standard", "You are a technical assistant.")

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
        intent = self._route_intent(standalone_query)
        logger.info("Tespit edilen niyet: %s", intent)
        
        context_text = ""
        best_meta = {}
        input_image = None
        
        if intent == "CHAT" and history:
            logger.info("Arama atlandi. Onceki baglam yukleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant" and turn.get("context_text"):
                    context_text = turn["context_text"]
                    break
            
            if not context_text:
                logger.warning("Gecmis baglam bulunamadi. SEARCH moduna donuluyor.")
                intent = "SEARCH"
        
        if intent == "SEARCH" or not history:
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
                    bm25_scores = self.bm25.get_scores(standalone_query.lower().split(" "))
                    top_bm25_indices = np.argsort(bm25_scores)[::-1]
                    
                    rank = 0
                    for idx in top_bm25_indices:
                        if rank >= self.n_results:
                            break
                        doc_id = self.ids[idx]
                        
                        if current_where and source_filter:
                            meta_source = self.doc_meta_map.get(doc_id, {}).get("source", "").lower()
                            if source_filter.lower() not in meta_source:
                                continue
                                
                        if doc_id not in scores_dict: 
                            scores_dict[doc_id] = 0
                        scores_dict[doc_id] += 1 / (self.k_constant + rank)
                        rank += 1
                        
                return scores_dict
                
            doc_scores = execute_retrieval(where_clause)
            
            if not doc_scores and where_clause:
                logger.warning("'%s' filtresi sonuc dondurmedi. Fallback (Filtresiz) arama baslatiliyor.", source_filter)
                doc_scores = execute_retrieval(None)
                
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
            
            pairs = [[standalone_query, txt] for txt in candidates_text if txt and str(txt).strip()]
            
            if pairs:
                scores = reranker.predict(pairs)
                best_idx = int(np.argmax(scores))
                context_text = candidates_text[best_idx]
                best_meta = candidates_meta[best_idx]
                image_path = best_meta.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    try:
                        input_image = Image.open(image_path)
                    except Exception as e:
                        logger.error("Resim yukleme hatasi: %s", e)

        if use_ste100:
            logger.info("Dinamik STE100 promptu uretiliyor. Format: %s", template_type)
            system_instruction = self.guard.build_injection_prompt(context_text, template_type)
        else:
            system_instruction = self.prompts.get("system_persona_standard", "You are a technical assistant.")

        messages = [{"role": "system", "content": system_instruction}]
        
        for msg in history[-4:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})

        user_prompt_text = (
            f"Context Information:\n{context_text}\n\n"
            f"User Question/Request:\n{query}"
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
            logger.error("API cagirilirken hata: %s", e)
            final_text = "Cevap uretilirken yerel modelde bir hata olustu."

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
                    logger.info("Ihlaller duzeltiliyor... (Deneme %s/%s)", retries, max_retries)
                    previous_text = current_text
                    current_text = self.refine_answer(current_text, feedback_report)
                    
                    if current_text.strip() == previous_text.strip():
                        break
                        
                    is_compliant, feedback_report = self.guard.analyze_and_report(current_text)
                    
                final_text = current_text
                was_corrected = True

        source_info = ""
        if best_meta and intent == "SEARCH":
            src_name = os.path.basename(best_meta.get("source", "Unknown"))
            pg_num = best_meta.get("page", "?")
            source_info = f"\n\n*(Kaynak: {src_name}, Sayfa {pg_num})*"
            if input_image:
                source_info += " [Gorsel Analiz Yapildi]"

        return final_text + source_info, context_text, is_compliant, was_corrected, feedback_report