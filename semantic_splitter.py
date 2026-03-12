import re
import numpy as np
from typing import List

from src.llm_manager import LLMManager


class STE100SemanticSplitter:
    def __init__(
        self, 
        similarity_threshold: float = 0.45, 
        max_chunk_length: int = 1500
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_length = max_chunk_length
        self.llm_manager = LLMManager()
        self.embedder = self.llm_manager.load_embedder()
        self.carry_over_buffer = ""

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def extract_semantic_chunks(self, raw_text: str, is_last_page: bool = False) -> List[str]:
        combined_text = f"{self.carry_over_buffer} {raw_text}".strip()
        self.carry_over_buffer = ""

        if not combined_text:
            return []

        ends_with_terminator = bool(re.search(r'(?:[.!?]|\n{2,})\s*$', combined_text))

        split_pattern = r'(?<=[.!?])\s+|\n{2,}'
        raw_sentences = re.split(split_pattern, combined_text)

        sentences = [
            s.replace('\n', ' ').strip() 
            for s in raw_sentences 
            if s and len(s.strip()) > 5
        ]

        if not sentences:
            return []

        if not ends_with_terminator and not is_last_page:
            self.carry_over_buffer = sentences.pop()

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self.embedder.encode(sentences)

        semantic_chunks = []
        current_chunk = sentences[0]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sim_score = self._cosine_similarity(embeddings[i-1], embeddings[i])

            is_semantic_shift = sim_score < self.similarity_threshold
            is_length_exceeded = len(current_chunk) + len(sentence) > self.max_chunk_length

            if is_semantic_shift or is_length_exceeded:
                semantic_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk:
            semantic_chunks.append(current_chunk.strip())

        return semantic_chunks

    def reset_buffer(self) -> None:
        self.carry_over_buffer = ""