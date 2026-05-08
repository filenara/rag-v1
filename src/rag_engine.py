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
from src.tokenization import technical_tokenize
from src.utils import load_prompts, load_config
from src.ste100_style_validator import validate_ste100_style

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
        self.min_rerank_score = self.retrieval_cfg.get("min_rerank_score", 0.25)
        self.enable_low_score_fallback = self.retrieval_cfg.get(
            "enable_low_score_fallback",
            False,
        )
        self.fallback_min_rrf_score = self.retrieval_cfg.get(
            "fallback_min_rrf_score",
            0.015,
        )
        self.verification_cfg = self.config.get("verification", {})
        self.verification_enabled = self.verification_cfg.get("enabled", True)
        self.verification_mode = self.verification_cfg.get("mode", "reject")
        self.verification_max_tokens = self.verification_cfg.get("max_tokens", 512)

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
        text = str(raw_text or "").strip()

        if not text:
            return ""

        answer_match = re.search(
            r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if answer_match:
            text = answer_match.group(1).strip()
        else:
            text = self._remove_reasoning_blocks(text)

        text = self._remove_stray_model_tags(text)
        text = self._remove_prompt_fragments(text)
        text = self._remove_internal_plan_fragments(text)
        text = text.strip()

        if self._contains_reasoning_leak(text) or self._contains_prompt_leak(text):
            logger.warning(
                "Internal reasoning or prompt fragment detected in model output. "
                "Output was suppressed."
            )
            return ""

        return text
    
    def _remove_reasoning_blocks(self, text: str) -> str:
        cleaned = str(text or "")

        for tag in ["thinking", "think"]:
            closed_block_pattern = (
                rf"<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>"
            )
            cleaned = re.sub(
                closed_block_pattern,
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )

            unclosed_block_pattern = rf"<\s*{tag}\b[^>]*>.*$"
            cleaned = re.sub(
                unclosed_block_pattern,
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )

        return cleaned.strip()


    def _remove_stray_model_tags(self, text: str) -> str:
        cleaned = str(text or "")

        cleaned = re.sub(
            r"<\s*/?\s*(answer|thinking|think)\s*>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        return cleaned.strip()


    def _contains_reasoning_leak(self, text: str) -> bool:
        normalized = str(text or "").lower()

        reasoning_markers = [
            "<thinking",
            "</thinking",
            "<think",
            "</think",
            "analyze document context",
            "synthesize and resolve conflict",
            "final plan:",
            "final plan",
            "the user is asking",
            "i will construct",
            "i will provide",
            "wait,",
            "let's re-read",
            "i will draft",
            "i will ensure",
            "i will structure",
            "i will use",
            "i will not use",
            "i need to",
            "key information to include",
            "xml structure",
            "` tags",
            "sentence 1:",
            "sentence 2:",
            "sentence 3:",
            "the procedure should",
            "the descriptive text must",
            "the safety text must",
            "tags.",
            "plan:",
            "i must ignore",
            "image provided in the context",
            "the image provided in the context",
            "provided in the context",
            "conflicts with the text source",
            "primary authority",
            "text source is the primary authority",
        ]

        return any(marker in normalized for marker in reasoning_markers)
    
    def _remove_prompt_fragments(self, text: str) -> str:
        lines = str(text or "").splitlines()
        cleaned_lines = []

        prompt_fragment_patterns = [
            r"^\s*`+\s*\.?\s*$",
            r"^\s*\.?\s*$",
            r"no conversational fillers",
            r"ensure the answer is direct",
            r"answer directly",
            r"omit conversational fillers",
            r"ensure the answer is direct and professional",
        ]

        def is_prompt_fragment(line: str) -> bool:
            normalized = line.strip().lower()

            if not normalized:
                return True

            return any(
                re.search(pattern, normalized, flags=re.IGNORECASE)
                for pattern in prompt_fragment_patterns
            )

        skipping_leading_fragments = True

        for line in lines:
            if skipping_leading_fragments and is_prompt_fragment(line):
                continue

            skipping_leading_fragments = False
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
    
    def _remove_internal_plan_fragments(self, text: str) -> str:
        cleaned = str(text or "").strip()

        if not cleaned:
            return ""

        plan_markers = [
            "i will draft",
            "i will ensure",
            "i will structure",
            "i will use",
            "i will not use",
            "i need to",
            "key information to include",
            "xml structure",
            "` tags",
            "tags.",
            "plan:",
            "i must ignore",
            "image provided in the context",
            "the image provided in the context",
            "provided in the context",
            "conflicts with the text source",
            "primary authority",
            "text source is the primary authority",
            "sentence 1:",
            "sentence 2:",
            "sentence 3:",
            "the procedure should",
            "the descriptive text must",
            "the safety text must",
        ]

        def has_plan_marker(value: str) -> bool:
            normalized = value.strip().lower()
            return any(marker in normalized for marker in plan_markers)

        paragraphs = re.split(r"\n\s*\n", cleaned)
        remaining_paragraphs = []
        skipping_leading_plan = True

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if skipping_leading_plan and has_plan_marker(paragraph):
                continue

            skipping_leading_plan = False
            remaining_paragraphs.append(paragraph)

        if remaining_paragraphs:
            cleaned = "\n\n".join(remaining_paragraphs).strip()

        lines = cleaned.splitlines()
        cleaned_lines = []
        skipping_leading_plan = True
        inside_leading_plan_block = False

        plan_list_pattern = re.compile(
            r"^\s*(?:\d+[\).\s-]+|[-*•]\s+)",
            flags=re.IGNORECASE,
        )

        for line in lines:
            stripped = line.strip()

            if skipping_leading_plan:
                if not stripped:
                    continue

                if has_plan_marker(stripped):
                    if re.match(r"^\s*plan\s*:\s*$", stripped, flags=re.IGNORECASE):
                        inside_leading_plan_block = True
                    continue

                if inside_leading_plan_block and plan_list_pattern.match(stripped):
                    continue

                skipping_leading_plan = False

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()


    def _contains_prompt_leak(self, text: str) -> bool:
        normalized = str(text or "").lower()

        prompt_leak_markers = [
            "no conversational fillers",
            "ensure the answer is direct",
            "answer directly",
            "omit conversational fillers",
            "[image attached to this message]",
            "i will draft",
            "i will ensure",
            "i will structure",
            "i need to",
            "key information to include",
            "xml structure",
            "` tags",
            "sentence 1:",
            "the procedure should",
            "the descriptive text must",
            "the safety text must",
            "tags.",
            "plan:",
            "i must ignore",
            "image provided in the context",
            "the image provided in the context",
            "provided in the context",
            "conflicts with the text source",
            "primary authority",
            "text source is the primary authority",
        ]

        return any(marker in normalized for marker in prompt_leak_markers)
    
    def _is_information_not_found(self, text: str) -> bool:
        normalized = str(text or "").lower()

        not_found_markers = [
            "information not found",
            "not found in provided documents",
            "not found in provided fragment",
        ]

        return any(marker in normalized for marker in not_found_markers)

    def _preview_debug_text(
        self,
        value: object,
        limit: int = 1200,
    ) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > limit:
            return (
                f"{text[:limit]}... "
                f"[truncated, total_chars={len(text)}]"
            )

        return text

    def _parse_json_response(self, raw_text: str) -> dict:
        cleaned_text = self._clean_output(raw_text).strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        start_idx = cleaned_text.find("{")
        end_idx = cleaned_text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError("Verifier response does not contain a valid JSON object.")

        json_text = cleaned_text[start_idx:end_idx + 1]
        return json.loads(json_text)

    def _preprocess_ste100_retrieval_query(
        self,
        query: str,
        template_type: str,
    ) -> str:
        raw_query = str(query or "").strip()

        if not raw_query:
            return ""

        cleaned_query = raw_query

        leading_patterns = [
            r"^\s*(write|create|generate|produce|provide)\s+"
            r"(an?\s+)?asd[-\s]?ste100\s+"
            r"(procedure|descriptive\s+text|safety\s+warning|"
            r"safety\s+caution)\s+(for|about|on)\s+",
            r"^\s*(write|create|generate|produce|provide)\s+"
            r"(an?\s+)?"
            r"(procedure|descriptive\s+text|safety\s+warning|"
            r"safety\s+caution)\s+(for|about|on)\s+",
        ]

        for pattern in leading_patterns:
            cleaned_query = re.sub(
                pattern,
                "",
                cleaned_query,
                flags=re.IGNORECASE,
            ).strip()

        cleanup_patterns = [
            r"\basd[-\s]?ste100\b",
            r"\bprocedure\b",
            r"\bdescriptive\s+text\b",
            r"\bsafety\s+warning\b",
            r"\bsafety\s+caution\b",
        ]

        for pattern in cleanup_patterns:
            cleaned_query = re.sub(
                pattern,
                " ",
                cleaned_query,
                flags=re.IGNORECASE,
            )

        cleaned_query = re.sub(r"[^A-Za-z0-9./_-]+", " ", cleaned_query)
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

        normalized = cleaned_query.lower()
        template = str(template_type or "").lower()

        expansion_rules = [
            {
                "triggers": ["rplidar", "a2m12", "power", "interface"],
                "expansion": (
                    "5V MOTOCTL TX RX power supply "
                    "communication interface"
                ),
            },
            {
                "triggers": ["rplidar", "a2m12", "power", "supply"],
                "expansion": "5V 4.9V 5.2V 2500mA 600mA power supply",
            },
            {
                "triggers": ["rplidar", "a2m12", "scanning"],
                "expansion": (
                    "12 meter 10Hz 16kHz 0.225 angular resolution"
                ),
            },
            {
                "triggers": ["os2", "laser"],
                "expansion": (
                    "Class 1 865nm OS2-128 OS2-64 OS2-32 laser safety"
                ),
            },
            {
                "triggers": ["metal", "52", "ac"],
                "expansion": (
                    "waterproof Gigabit Ethernet 802.11ac 2.4GHz 5GHz"
                ),
            },
            {
                "triggers": ["emergency", "shutdown"],
                "expansion": (
                    "Red Button Cable A supervisor ext 440 "
                    "emergency shutdown"
                ),
            },
        ]

        expansions = []

        for rule in expansion_rules:
            if all(trigger in normalized for trigger in rule["triggers"]):
                expansions.append(rule["expansion"])
                break

        if template == "safety":
            expansions.append("warning caution danger safety")

        retrieval_terms = [cleaned_query]
        retrieval_terms.extend(expansions)

        retrieval_query = " ".join(
            term.strip()
            for term in retrieval_terms
            if term and term.strip()
        )

        retrieval_query = re.sub(r"\s+", " ", retrieval_query).strip()

        logger.info(
            "STE100 retrieval query preprocessed. original=%r, retrieval=%r",
            raw_query,
            retrieval_query,
        )

        return retrieval_query or raw_query    

    def _analyze_and_rewrite_query(self, query: str, history: list) -> str:
        history_text = ""
        if history:
            for msg in history[-3:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                if isinstance(content, str):
                    history_text += f"{role}: {content}\n"
                
        prompt_template = self.prompts.get(
            "query_rewrite_prompt", 
            "Rewrite the query to be standalone.\nHistory:\n{history_text}\nQuery: {query}\nStandalone Query:"
        )
        prompt = prompt_template.format(history_text=history_text, query=query)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            raw_response = self._generate_api(messages, max_tokens=100)
            return self._clean_output(raw_response).strip()
        except Exception as e:
            logger.error("Sorgu analizi hatasi: %s", e, exc_info=True)
            return query

    def _expand_revision_query(self, query: str, history: list) -> str:
        history_text = ""
        if history:
            for msg in history[-6:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                if isinstance(content, str):
                    history_text += f"{role}: {content}\n"
                    
        prompt_template = self.prompts.get(
            "revision_expansion_prompt", 
            "Find root query from history and merge with feedback.\nHistory:\n{history_text}\nFeedback: {query}\nMerged:"
        )
        prompt = prompt_template.format(history_text=history_text, query=query)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            raw_response = self._generate_api(messages, max_tokens=150)
            return self._clean_output(raw_response).strip()
        except Exception as e:
            logger.error("Kok sorgu cikarimi hatasi: %s", e, exc_info=True)
            return query

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

    def refine_answer(self, draft_text: str, feedback_list: list, context_text: str, core_rules: list) -> str:
        logger.info("[Self-Correction] STE100 ihlalleri dinamik baglam ve kurallarla duzeltiliyor...")
        
        feedback_str = "\n".join(feedback_list)
        rules_str = "\n".join([f"- {r}" for r in core_rules])
        
        template = self.prompts.get(
            "self_correction_prompt",
            "Fix the errors in the draft using the context and rules.\nContext:\n{context_text}\nRules:\n{core_rules}\nDraft:\n{draft_answer}\nErrors:\n{feedback_report}\nFixed Text:"
        )
        
        prompt_text = template.format(
            context_text=context_text,
            core_rules=rules_str,
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
        
    def _verify_answer_grounding(
        self,
        final_answer: str,
        context_text: str,
    ) -> bool:
        if not self.verification_enabled:
            return True

        if not context_text.strip():
            return False

        if not final_answer.strip():
            return False

        normalized_answer = final_answer.strip().lower()

        if "information not found" in normalized_answer:
            return True

        prompt_template = self.prompts.get(
            "answer_verification_prompt",
            (
                "Verify whether the final answer is fully supported by the context. "
                "Return only valid JSON with verdict SUPPORTED or UNSUPPORTED.\n\n"
                "Context:\n{context_text}\n\n"
                "Final answer:\n{final_answer}\n\n"
                "{{\"verdict\": \"SUPPORTED\", \"unsupported_claims\": []}}"
            ),
        )

        prompt_text = prompt_template.format(
            context_text=context_text,
            final_answer=final_answer,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
            }
        ]

        try:
            raw_response = self._generate_api(
                messages,
                max_tokens=self.verification_max_tokens,
            )
            verification_result = self._parse_json_response(raw_response)
        except Exception as e:
            logger.error("Answer verification failed: %s", e, exc_info=True)
            return False

        verdict = str(verification_result.get("verdict", "")).upper().strip()

        if verdict == "SUPPORTED":
            return True

        unsupported_claims = verification_result.get("unsupported_claims", [])
        logger.warning(
            "Answer rejected by verifier. verdict=%s, unsupported_claims=%s",
            verdict,
            unsupported_claims,
        )

        return False
    
    def retrieve_context(
        self,
        query: str,
        collection_name: str,
        source_filter: str = "",
        top_k_context: int = 3,
    ) -> Tuple[str, List[Dict], Optional[Image.Image]]:
        embedder = self.llm_manager.load_embedder()
        reranker = self.llm_manager.load_reranker()

        col = self.db.get_collection(collection_name)
        q_vec = embedder.encode([query], normalize_embeddings=True)

        where_clause = None
        if source_filter:
            where_clause = {"source": {"$contains": source_filter}}

        def execute_retrieval(current_where):
            scores_dict = {}

            vec_res = col.query(
                query_embeddings=q_vec,
                n_results=self.n_results,
                where=current_where,
            )

            if vec_res["ids"] and vec_res["ids"][0]:
                for rank, doc_id in enumerate(vec_res["ids"][0]):
                    if doc_id not in scores_dict:
                        scores_dict[doc_id] = 0

                    scores_dict[doc_id] += 1 / (self.k_constant + rank)

            if self.bm25:
                query_tokens = technical_tokenize(query)
                bm25_scores = np.array(self.bm25.get_scores(query_tokens))

                if current_where and source_filter:
                    filter_lower = source_filter.lower()
                    valid_indices = []

                    for src, indices in self.source_to_indices.items():
                        if filter_lower in src:
                            valid_indices.extend(indices)

                    if valid_indices:
                        mask = np.ones(len(bm25_scores), dtype=bool)
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

        logger.info(
            "[RetrievalDebug] query=%r source_filter=%r "
            "doc_scores_count=%s n_results=%s top_k_rerank=%s",
            query,
            source_filter,
            len(doc_scores),
            self.n_results,
            self.top_k_rerank,
        )

        sorted_candidates = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:self.top_k_rerank]

        logger.info(
            "[RetrievalDebug] sorted_candidates_count=%s",
            len(sorted_candidates),
        )

        for debug_rank, (debug_doc_id, debug_rrf_score) in enumerate(
            sorted_candidates[:10],
            start=1,
        ):
            debug_meta = self.doc_meta_map.get(debug_doc_id, {})
            debug_source = os.path.basename(
                debug_meta.get("source", "Unknown")
            )
            debug_page = debug_meta.get("page", "?")
            debug_parent = debug_meta.get("parent_context", "")

            logger.info(
                "[RetrievalDebug] candidate rank=%s source=%r page=%r "
                "context=%r rrf_score=%.6f doc_id=%r",
                debug_rank,
                debug_source,
                debug_page,
                debug_parent,
                float(debug_rrf_score),
                debug_doc_id,
            )

        if not sorted_candidates:
            logger.info(
                "[RetrievalDebug] no sorted candidates. Returning empty "
                "context."
            )
            return "", [], None

        valid_candidates = []

        for candidate_rank, (doc_id, rrf_score) in enumerate(sorted_candidates):
            text = self.doc_text_map.get(doc_id, "")
            meta = self.doc_meta_map.get(doc_id, {})

            if text and str(text).strip():
                valid_candidates.append(
                    {
                        "text": text,
                        "meta": meta,
                        "rrf_score": float(rrf_score),
                        "candidate_rank": candidate_rank,
                    }
                )

        logger.info(
            "[RetrievalDebug] valid_candidates_count=%s",
            len(valid_candidates),
        )

        if not valid_candidates:
            logger.info(
                "[RetrievalDebug] no valid candidates after text lookup. "
                "Returning empty context."
            )
            return "", [], None

        pairs = [
            [query, candidate["text"]]
            for candidate in valid_candidates
        ]

        if not pairs:
            return "", [], None

        scores = reranker.predict(pairs)
        best_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        best_rrf_score = max(
            candidate.get("rrf_score", 0.0)
            for candidate in valid_candidates
        )

        logger.info(
            "[RetrievalDebug] reranker_scores_count=%s best_score=%.6f "
            "min_rerank_score=%.6f best_rrf_score=%.6f "
            "fallback_enabled=%s fallback_min_rrf_score=%.6f",
            len(scores),
            best_score,
            float(self.min_rerank_score),
            best_rrf_score,
            self.enable_low_score_fallback,
            float(self.fallback_min_rrf_score),
        )

        for debug_idx, debug_score in enumerate(scores):
            debug_candidate = valid_candidates[debug_idx]
            debug_meta = debug_candidate.get("meta", {})
            debug_source = os.path.basename(
                debug_meta.get("source", "Unknown")
            )
            debug_page = debug_meta.get("page", "?")
            debug_parent = debug_meta.get("parent_context", "")

            logger.info(
                "[RetrievalDebug] rerank rank=%s source=%r page=%r "
                "context=%r rerank_score=%.6f rrf_score=%.6f",
                debug_idx + 1,
                debug_source,
                debug_page,
                debug_parent,
                float(debug_score),
                float(debug_candidate.get("rrf_score", 0.0)),
            )

        if best_score < self.min_rerank_score:
            if (
                not self.enable_low_score_fallback
                or best_rrf_score < self.fallback_min_rrf_score
            ):
                logger.info(
                    "[RetrievalDebug] rejected by rerank threshold. "
                    "best_score=%s, threshold=%s, best_rrf_score=%s, "
                    "fallback_enabled=%s",
                    best_score,
                    self.min_rerank_score,
                    best_rrf_score,
                    self.enable_low_score_fallback,
                )
                return "", [], None

            logger.warning(
                "[RetrievalDebug] low-score retrieval fallback active. "
                "best_score=%s, "
                "threshold=%s, best_rrf_score=%s, fallback_min_rrf_score=%s",
                best_score,
                self.min_rerank_score,
                best_rrf_score,
                self.fallback_min_rrf_score,
            )

        top_indices = np.argsort(scores)[::-1][:top_k_context]

        logger.info(
            "[RetrievalDebug] selected_top_indices=%s top_k_context=%s",
            [int(index) for index in top_indices],
            top_k_context,
        )

        context_texts = []
        sources = []
        seen_sources = set()
        first_image_path = ""

        for idx in top_indices:
            candidate = valid_candidates[idx]
            text = candidate["text"]
            meta = candidate["meta"]

            context_texts.append(text)

            source_name = os.path.basename(meta.get("source", "Unknown"))
            page_num = meta.get("page", "?")
            parent_context = meta.get("parent_context", "")
            has_visual = str(meta.get("has_visual", "False"))
            image_path = meta.get("image_path", "")

            source_key = (
                source_name,
                str(page_num),
                parent_context,
                image_path,
            )

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(
                    {
                        "source": source_name,
                        "page": page_num,
                        "parent_context": parent_context,
                        "has_visual": has_visual.lower() == "true",
                        "image_path": image_path,
                        "rerank_score": float(scores[idx]),
                        "rrf_score": float(candidate.get("rrf_score", 0.0)),
                        "candidate_rank": int(candidate.get("candidate_rank", -1)),
                    }
                )

            if not first_image_path and image_path:
                first_image_path = image_path.split(",")[0].strip()

        context_text = "\n\n".join(context_texts)
        input_image = None

        if first_image_path and os.path.exists(first_image_path):
            try:
                input_image = Image.open(first_image_path)
            except Exception as e:
                logger.error("Resim yukleme hatasi: %s", e, exc_info=True)

        return context_text, sources, input_image

    def search_and_answer(
        self, 
        query: str, 
        collection_name: str, 
        history: list = None, 
        use_ste100: bool = False, 
        strict_mode: bool = False,
        template_type: str = "General"
    ) -> Tuple[str, str, bool, bool, list, list]:
        
        if history is None:
            history = []
        
        # 1. Regex ile Slash Command Tespiti ve Guvenlik Kontrolu
        slash_match = re.search(r'(?:^|\s)/([\w.-]+)', query)
        source_filter = ""
        
        if slash_match:
            potential_source = slash_match.group(1).lower()
            matched_source = None
            
            for src_key in self.source_to_indices.keys():
                if potential_source in src_key:
                    matched_source = potential_source
                    break
                    
            if matched_source:
                source_filter = matched_source
                logger.info("Gecerli slash command tespit edildi. Filtre: %s", source_filter)
            else:
                logger.warning("Gecersiz slash command tespit edildi (%s), yok sayiliyor.", potential_source)
                
            # Sorguyu anlamsal gurultuden temizle
            query = re.sub(r'(?:^|\s)/[\w.-]+', '', query).strip()

        # 2. Intent Tespiti ve Arayuz Ezmesi (Override)
        explicit_template = str(template_type).strip().capitalize()
        
        if use_ste100 and explicit_template in ["Procedure", "Descriptive", "Safety"]:
            intent = explicit_template.upper()
            logger.info("Arayuzden secilen format (Explicit Intent) uygulaniyor: %s", intent)
        else:
            logger.info("Niyet analizi yapiliyor...")
            intent = self._route_intent(query, history, use_ste100)
            logger.info("Tespit edilen niyet: %s", intent)

        # 3. Revizyon Mantigi ve Akis Yonlendirmesi
        if intent == "REVISION":
            logger.info("Revizyon algilandi. Kok sorgu cikarimi yapiliyor...")
            standalone_query = self._expand_revision_query(query, history)
            logical_intent = "SEARCH"
            template_type = explicit_template if explicit_template != "General" else "Procedure"
        elif intent in ["PROCEDURE", "DESCRIPTIVE", "SAFETY", "QA"]:
            logical_intent = "SEARCH"
            template_type = explicit_template if use_ste100 else "General"

            if use_ste100:
                retrieval_query = self._preprocess_ste100_retrieval_query(
                    query=query,
                    template_type=template_type,
                )

                if history:
                    standalone_query = self._analyze_and_rewrite_query(
                        retrieval_query,
                        history,
                    )
                else:
                    standalone_query = retrieval_query
            else:
                standalone_query = self._analyze_and_rewrite_query(
                    query,
                    history,
                )
        else:
            logical_intent = "CHAT"
            standalone_query = query
            
        context_text = ""
        sources = []
        input_image = None
        
        if logical_intent == "CHAT" and history:
            logger.info("Arama atlandi. Onceki baglam yukleniyor...")
            for turn in reversed(history):
                if turn.get("role") == "assistant":
                    context_text = turn.get("context_text", "")
                    break

        if logical_intent == "SEARCH":
            context_text, sources, input_image = self.retrieve_context(
                query=standalone_query,
                collection_name=collection_name,
                source_filter=source_filter,
            )

            logger.info(
                "[GenerationDebug] retrieval completed. "
                "standalone_query=%r context_length=%s sources_count=%s",
                standalone_query,
                len(context_text or ""),
                len(sources or []),
            )

            logger.info(
                "[GenerationDebug] context_preview=%r",
                self._preview_debug_text(context_text, limit=2000),
            )

            for source_index, source in enumerate(sources or [], start=1):
                logger.info(
                    "[GenerationDebug] source index=%s source=%r page=%r "
                    "context=%r rerank_score=%s rrf_score=%s",
                    source_index,
                    source.get("source", "Unknown"),
                    source.get("page", "?"),
                    source.get("parent_context", ""),
                    source.get("rerank_score", None),
                    source.get("rrf_score", None),
                )

            if not context_text.strip():
                return (
                    "Information not found in provided documents.",
                    "",
                    True,
                    False,
                    [],
                    [],
                )
        messages = []
        
        history_text_formatted = ""
        for msg in history[-3:]:
            role_name = "User" if msg.get("role") == "user" else "Assistant"
            content_val = msg.get("content", "")
            if isinstance(content_val, str) and content_val.strip():
                history_text_formatted += f"{role_name}: {content_val}\n"
                
        if not history_text_formatted.strip():
            history_text_formatted = "No previous conversation."

        if use_ste100 and logical_intent == "SEARCH":
            logger.info(
                "Dinamik STE100 promptu uretiliyor. Format: %s",
                template_type,
            )
            persona = self.prompts.get(
                "system_persona",
                "You are a technical assistant.",
            )
            dynamic_rules = self.guard.build_injection_prompt(
                context_text,
                template_type,
            )

            logger.info(
                "[GenerationDebug] ste100_prompt_state "
                "template_type=%r strict_mode=%s context_length=%s "
                "sources_count=%s",
                template_type,
                strict_mode,
                len(context_text or ""),
                len(sources or []),
            )

            logger.info(
                "[GenerationDebug] dynamic_rules_preview=%r",
                self._preview_debug_text(dynamic_rules, limit=2000),
            )

            system_instruction = f"{persona}\n\n{dynamic_rules}"
        else:
            system_instruction = self.prompts.get("system_persona_standard", "You are a technical assistant.")

        messages.append({"role": "system", "content": system_instruction})
        
        base_template = self.prompts.get(
            "response_template", 
            "Context Information:\n{context_text}\n\nUser Question/Request:\n{query}"
        )
        
        # Orijinal temizlenmis query ile yanit uretimi yapilir
        user_prompt_text = base_template.format(
            history_text=history_text_formatted.strip(),
            context_text=context_text,
            query=query,
        )

        logger.info(
            "[GenerationDebug] user_prompt_state "
            "original_query=%r generation_query=%r "
            "user_prompt_length=%s",
            query,
            query,
            len(user_prompt_text or ""),
        )

        logger.info(
            "[GenerationDebug] user_prompt_preview=%r",
            self._preview_debug_text(user_prompt_text, limit=2500),
        )

        user_content = []
        if input_image:
            user_content.append({"type": "image", "image": input_image})
            user_prompt_text = "[IMAGE ATTACHED TO THIS MESSAGE]\n" + user_prompt_text
            
        user_content.append({"type": "text", "text": user_prompt_text})
        messages.append({"role": "user", "content": user_content})
        
        try:
            raw_text = self._generate_api(messages, max_tokens=2048)
            logger.info(
                "[GenerationDebug] raw_model_output_preview=%r",
                self._preview_debug_text(raw_text, limit=2500),
            )
            final_text = self._clean_output(raw_text)
            logger.info(
                "[GenerationDebug] cleaned_model_output_preview=%r",
                self._preview_debug_text(final_text, limit=2500),
            )

        except Exception as e:
            logger.error("API cagirilirken hata: %s", e, exc_info=True)
            final_text = "Cevap uretilirken yerel modelde bir hata olustu."

        if logical_intent == "SEARCH" and self.verification_mode == "reject":
            is_grounded = self._verify_answer_grounding(
                final_answer=final_text,
                context_text=context_text,
            )

            if not is_grounded:
                final_text = "Information not found in provided documents."
                sources = []

        is_compliant = True
        was_corrected = False
        feedback_report = []

        if use_ste100 and logical_intent == "SEARCH":
            logger.info("STE100 kurallari denetleniyor...")

            is_not_found = self._is_information_not_found(final_text)
            dictionary_compliant, feedback_report = self.guard.analyze_and_report(
                final_text
            )

            style_result = validate_ste100_style(
                answer=final_text,
                template_type=template_type,
                require_safety_marker=str(template_type).lower() == "safety",
            )

            style_feedback = style_result.get("feedback", [])

            if is_not_found:
                logger.warning(
                    "[GenerationDebug] final answer is not-found before return. "
                    "context_length=%s sources_count=%s final_answer=%r",
                    len(context_text or ""),
                    len(sources or []),
                    self._preview_debug_text(final_text, limit=1000),
                )
                is_compliant = dictionary_compliant
            else:
                feedback_report = feedback_report + style_feedback
                is_compliant = dictionary_compliant and style_result["passed"]

            if not is_compliant and strict_mode and not is_not_found:
                max_retries = 2
                retries = 0
                current_text = final_text

                while not is_compliant and retries < max_retries:
                    retries += 1
                    logger.info(
                        "Ihlaller duzeltiliyor... (Deneme %s/%s)",
                        retries,
                        max_retries,
                    )
                    previous_text = current_text

                    current_text = self.refine_answer(
                        draft_text=current_text,
                        feedback_list=feedback_report,
                        context_text=context_text,
                        core_rules=self.guard.core_rules,
                    )

                    if current_text.strip() == previous_text.strip():
                        break

                    dictionary_compliant, feedback_report = (
                        self.guard.analyze_and_report(current_text)
                    )

                    style_result = validate_ste100_style(
                        answer=current_text,
                        template_type=template_type,
                        require_safety_marker=str(template_type).lower() == "safety",
                    )

                    style_feedback = style_result.get("feedback", [])
                    feedback_report = feedback_report + style_feedback
                    is_compliant = dictionary_compliant and style_result["passed"]

                final_text = current_text
                was_corrected = retries > 0

            logger.info(
                "[GenerationDebug] ste100_validation_result "
                "is_compliant=%s was_corrected=%s feedback_count=%s "
                "not_found=%s",
                is_compliant,
                was_corrected,
                len(feedback_report or []),
                self._is_information_not_found(final_text),
            )

            if feedback_report:
                logger.info(
                    "[GenerationDebug] ste100_feedback=%s",
                    feedback_report,
                )    

        return final_text, context_text, is_compliant, was_corrected, feedback_report, sources